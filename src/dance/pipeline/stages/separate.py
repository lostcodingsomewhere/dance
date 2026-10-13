"""
Stem separation stage (Demucs).

Writes one StemFile row per (track, kind). Demucs's source names map to our
StemKind enum 1:1: drums, bass, vocals, other.

Hygiene guarantees (no model or schema change):

- **Atomic writes.** Every stem is written to a hidden same-directory temp
  file (``.<kind>.<random>.partial.wav``) and only ``os.replace``-d onto its
  final ``<kind>.wav`` name once *all four* wrote cleanly. A Python-level
  failure removes the temps and leaves any previous stems untouched; a hard
  native crash (SIGBUS / exit 138 — see ``pipeline/heal_ledger.py``) can
  only strand hidden temps, which the next run sweeps. Live therefore never
  references a truncated ``<kind>.wav``.
- **Residual QC.** ``StemFile.separation_quality`` holds the reconstruction
  residual of the whole separation, ``20*log10(rms(mix - sum(stems)) /
  rms(mix))`` in dB, computed on the stems *as written* (clipped to the PCM
  range). More negative = the four stems sum back to the mix more faithfully.
  It is a track-level number copied onto each of the track's stem rows.
  ``htdemucs_ft`` takes each stem from a different fine-tuned model (diagonal
  bag weights), so nothing forces the sum to equal the mix: about -20 dB is
  normal (measured -20.7 dB on a real techno excerpt); a value drifting
  toward 0 dB flags a broken separation.
- **Provenance.** ``StemFile.model_used`` records the settings that shaped
  the audio, e.g. ``htdemucs_ft|s2|pcm24|demucs4.0.1`` (model, shifts, PCM
  subtype, Demucs version), so a shifts=0 stem is distinguishable from a
  shifts=2 one.
- **MPS fallback.** A Python-level ``RuntimeError`` from Demucs on MPS is
  retried once on CPU (logged). A native SIGBUS cannot be caught in-process;
  that path is handled by the dispatcher's heal ledger instead.

Input scaling: the Demucs CLI standardises the mix (``(wav - mean) / std``)
before ``apply_model`` and undoes it afterwards. We deliberately don't.
``HTDemucs.forward`` already normalises every segment it sees (both the
spectrogram branch and the waveform branch, ``(x - mean) / (1e-5 + std)``)
and de-normalises its output, and everything around it in ``apply_model``
(shift averaging, overlap-add) is linear — so the network is scale- and
offset-equivariant and the CLI's outer standardisation is a no-op up to
float rounding and the 1e-5 epsilon. Matching it would add a full-track
copy for no audible change.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import time
from collections.abc import Sequence
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import StemFile, StemKind, Track, TrackState
from dance.pipeline.utils.device import pick_device

logger = logging.getLogger(__name__)

try:
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    _DEMUCS_OK = True
except ImportError:
    _DEMUCS_OK = False
    logger.warning("Demucs not available — stem separation disabled")


# Demucs source name -> our StemKind (htdemucs_ft happens to use the same names,
# but be explicit so future model swaps don't break us).
_DEMUCS_SOURCE_MAP: dict[str, StemKind] = {
    "drums": StemKind.DRUMS,
    "bass": StemKind.BASS,
    "vocals": StemKind.VOCALS,
    "other": StemKind.OTHER,
}

# Suffix of in-progress stem temp files. Hidden (dot-prefixed) and never a
# ``<kind>.wav`` name, so nothing that looks for stems can pick one up.
_PARTIAL_SUFFIX = ".partial.wav"

# ``StemFile.model_used`` is ``String(64)``.
_MODEL_USED_MAX = 64

# Permissions for finished stems. ``mkstemp`` creates 0600 files; stems are
# ordinary user media (Live, Finder, backups), so widen to the usual 0644.
# (Reading the real umask means ``os.umask`` set+reset, which is process-wide
# and racy under the dispatcher's worker threads.)
_STEM_FILE_MODE = 0o644


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_stems_hygiene.py)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def demucs_version() -> str:
    """Installed Demucs version, or ``"unknown"``."""
    try:
        return _dist_version("demucs")
    except PackageNotFoundError:
        pass
    try:
        import demucs

        return str(getattr(demucs, "__version__", "unknown"))
    except ImportError:
        return "unknown"


def stem_provenance(
    model_name: str, shifts: int, bit_depth: str, version: str | None = None
) -> str:
    """Build the ``StemFile.model_used`` provenance string.

    ``htdemucs_ft|s2|pcm24|demucs4.0.1`` — model, ``shifts``, PCM subtype,
    Demucs version. Capped at 64 chars (the column width); if it would
    overflow, the model name is trimmed so the knobs survive.
    """
    ver = version if version is not None else demucs_version()
    subtype = bit_depth.lower().replace("_", "")
    suffix = f"|s{shifts}|{subtype}|demucs{ver}"
    room = max(0, _MODEL_USED_MAX - len(suffix))
    return (model_name[:room] + suffix)[:_MODEL_USED_MAX]


def reconstruction_residual_db(
    mix: Any,
    stems: Sequence[Any],
    *,
    clip: float | None = 1.0,
    chunk: int = 1 << 20,
) -> float | None:
    """Reconstruction residual of a separation, in dB relative to the mix.

    ``20 * log10(rms(mix - sum(stems)) / rms(mix))``. ``mix`` and every stem
    are ``(channels, samples)`` arrays. ``clip`` mirrors what the PCM writer
    does to out-of-range floats (soundfile clips to ±1.0) so the number
    describes the stems Live actually plays; pass ``None`` to skip it.

    Sums are accumulated chunk-wise in float64 so a 6-minute stereo track
    doesn't need a full-length float64 copy. Returns ``None`` for a silent
    (or empty) mix, where the ratio is undefined. A perfect reconstruction
    is floored at -200 dB rather than ``-inf`` (the value goes to SQLite and
    JSON).
    """
    mix = np.asarray(mix)
    if mix.ndim == 1:
        mix = mix[None, :]
    length = mix.shape[-1]
    arrs = []
    for s in stems:
        a = np.asarray(s)
        if a.ndim == 1:
            a = a[None, :]
        arrs.append(a)
        length = min(length, a.shape[-1])
    if length == 0:
        return None

    sum_mix = 0.0
    sum_res = 0.0
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        m = mix[:, start:end].astype(np.float64)
        r = m.copy()
        for a in arrs:
            part = a[:, start:end].astype(np.float64)
            if clip is not None:
                np.clip(part, -clip, clip, out=part)
            r -= part
        sum_mix += float(np.sum(m * m))
        sum_res += float(np.sum(r * r))

    n = length * mix.shape[0]
    rms_mix = (sum_mix / n) ** 0.5
    if rms_mix < 1e-9:
        return None
    rms_res = (sum_res / n) ** 0.5
    if rms_res <= 0.0:
        return -200.0
    return max(-200.0, 20.0 * float(np.log10(rms_res / rms_mix)))


def sweep_partials(out_dir: Path) -> int:
    """Delete stem temp files stranded by a hard crash. Returns the count."""
    removed = 0
    for p in out_dir.glob(f".*{_PARTIAL_SUFFIX}"):
        with contextlib.suppress(OSError):
            p.unlink()
            removed += 1
    if removed:
        logger.info("Removed %d stranded partial stem file(s) in %s", removed, out_dir)
    return removed


def write_stems_atomic(
    out_dir: Path,
    stems: Sequence[tuple[str, Any]],
    samplerate: int,
    subtype: str,
) -> dict[str, Path]:
    """Write ``(name, (samples, channels) array)`` stems as ``<name>.wav``.

    Two phases: every stem is written to its own hidden temp file in
    ``out_dir`` (same filesystem, so the rename is atomic), and only after
    all of them succeed is each one ``os.replace``-d onto its final name.
    If any write raises, every temp is removed and no final path is touched,
    so a previous complete set of stems survives intact.
    """
    temps: list[tuple[str, Path]] = []
    try:
        for name, data in stems:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{name}.", suffix=_PARTIAL_SUFFIX, dir=str(out_dir)
            )
            try:
                os.fchmod(fd, _STEM_FILE_MODE)
            finally:
                os.close(fd)
            temps.append((name, Path(tmp_name)))
            sf.write(tmp_name, data, samplerate, subtype=subtype)

        final: dict[str, Path] = {}
        for name, tmp in temps:
            dest = out_dir / f"{name}.wav"
            os.replace(tmp, dest)
            final[name] = dest
        return final
    except BaseException:
        # Already-replaced temps no longer exist; unlink() just no-ops them.
        for _, tmp in temps:
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class StemSeparationStage:
    """Separates each track into 4 stems via Demucs."""

    name = "separate"
    input_state = TrackState.ANALYZED
    inflight_state = TrackState.SEPARATING
    output_state = TrackState.SEPARATED
    error_state = TrackState.ERROR
    # 2 workers + GPU semaphore lets two Demucs runs overlap on MPS.
    # Bump down to 1 if you OOM, or DANCE_GPU_CONCURRENCY higher to widen.
    concurrency = 2
    uses_gpu = True

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str | None = None
        self._model_name: str | None = None
        # Separate CPU copy for the MPS->CPU retry (see _cpu_fallback_model).
        self._cpu_model: Any = None
        self._cpu_model_name: str | None = None
        self._cpu_lock = threading.Lock()
        # Device that actually produced the last separation (for logs/tests).
        self.last_device: str | None = None

    # ------------------------------------------------------------------

    def _ensure_model(self, settings: Settings) -> None:
        if self._model is not None and self._model_name == settings.demucs_model:
            return
        if not _DEMUCS_OK:
            raise RuntimeError("Demucs is not installed")

        device = pick_device(settings.demucs_device)
        logger.info("Loading Demucs model %s on %s", settings.demucs_model, device)
        model = get_model(settings.demucs_model)
        try:
            model.to(device)
        except RuntimeError as exc:  # NotImplementedError is a subclass
            if device != "mps":
                raise
            logger.warning("MPS load failed (%s); falling back to CPU", exc)
            device = "cpu"
            model.to(device)
        model.eval()
        self._device = device
        self._model = model
        self._model_name = settings.demucs_model

    def _cpu_fallback_model(self, settings: Settings) -> Any:
        """A CPU-resident copy of the model, loaded on first need.

        Not ``self._model``: ``apply_model`` moves (sub-)models onto the
        target device for the call, and that instance is shared by the
        stage's worker threads — re-homing it to CPU here would yank it out
        from under a concurrent MPS separation. Loading from the local
        weight cache is a few seconds and only happens after a failure.
        """
        with self._cpu_lock:
            if self._cpu_model is None or self._cpu_model_name != settings.demucs_model:
                logger.info("Loading CPU copy of Demucs model %s", settings.demucs_model)
                model = get_model(settings.demucs_model)
                model.to("cpu")
                model.eval()
                self._cpu_model = model
                self._cpu_model_name = settings.demucs_model
            return self._cpu_model

    def _apply(self, model: Any, audio_np: Any, device: str | None, settings: Settings) -> Any:
        wav = torch.from_numpy(audio_np)
        if device is not None:
            wav = wav.to(device)
        with torch.no_grad():
            return apply_model(
                model,
                wav[None],
                device=device,
                progress=False,
                num_workers=0,
                # shifts>0 averages N randomly time-shifted passes to reduce
                # separation artifacts (≈ N× compute). Tunable via settings.
                shifts=settings.demucs_shifts,
            )[0]

    def _separate(self, audio_np: Any, settings: Settings) -> tuple[Any, str | None]:
        """Run Demucs; on an MPS ``RuntimeError`` retry once on CPU.

        Returns ``(sources, device_used)``. Only Python-level errors reach
        this ``except`` — a native SIGBUS kills the process outright.
        """
        try:
            return self._apply(self._model, audio_np, self._device, settings), self._device
        except RuntimeError as exc:
            if self._device != "mps":
                raise
            logger.warning(
                "Demucs failed on MPS (%s: %s); retrying once on CPU",
                type(exc).__name__,
                exc,
            )
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()
            cpu_model = self._cpu_fallback_model(settings)
            return self._apply(cpu_model, audio_np, "cpu", settings), "cpu"

    # ------------------------------------------------------------------

    def process(self, session: Session, track: Track, settings: Settings) -> bool:
        if not _DEMUCS_OK:
            track.state = self.error_state.value
            track.error_message = "Demucs not installed"
            session.commit()
            return False

        # Idempotent: if all 4 stems already exist on disk and in DB, skip.
        existing = session.query(StemFile).filter_by(track_id=track.id).all()
        if len(existing) == 4 and all(Path(s.path).exists() for s in existing):
            track.state = self.output_state.value
            session.commit()
            logger.info("Stems already present for track %s", track.id)
            return True

        track.state = TrackState.SEPARATING.value
        session.commit()

        self._ensure_model(settings)

        path = Path(track.file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        logger.info("Separating: %s — %s", track.artist, track.title)
        # Load via librosa as stereo at Demucs's native sample rate, then
        # convert to a torch tensor. We use librosa rather than torchaudio
        # because torchaudio 2.9 dropped its built-in loader in favor of
        # torchcodec, which would be an extra dep + a system FFmpeg.
        # No CLI-style standardisation — see the module docstring.
        audio_np, _ = librosa.load(str(path), sr=self._model.samplerate, mono=False)
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np], axis=0)
        elif audio_np.shape[0] > 2:
            audio_np = audio_np[:2]

        t0 = time.monotonic()
        sources, device_used = self._separate(audio_np, settings)
        elapsed = time.monotonic() - t0
        self.last_device = device_used

        named: list[tuple[StemKind, Any]] = []
        for i, source_name in enumerate(self._model.sources):
            kind = _DEMUCS_SOURCE_MAP.get(source_name)
            if kind is None:
                logger.warning("Skipping unknown demucs source: %s", source_name)
                continue
            named.append((kind, sources[i].cpu().numpy()))  # (channels, samples)

        # QC against the stems we are about to write (a skipped unknown
        # source correctly shows up as residual — Live won't have it).
        residual_db = reconstruction_residual_db(audio_np, [a for _, a in named])
        clipped = {kind.value: int(np.count_nonzero(np.abs(a) > 1.0)) for kind, a in named}
        if any(clipped.values()):
            logger.warning(
                "Track %s: stem samples beyond full scale (hard-clipped on write): %s",
                track.id,
                {k: v for k, v in clipped.items() if v},
            )

        # Drop stale rows so we don't accumulate duplicates on a retry.
        # Committed before the files change so a crash between here and
        # the final commit leaves "no stems" (→ re-separate), never old rows
        # describing new audio.
        if existing:
            for s in existing:
                session.delete(s)
            session.commit()

        out_dir = settings.stems_dir / track.file_hash[:8]
        out_dir.mkdir(parents=True, exist_ok=True)
        sweep_partials(out_dir)

        # soundfile expects shape (samples, channels), torch tensors are
        # (channels, samples) — transpose before write.
        paths = write_stems_atomic(
            out_dir,
            [(kind.value, a.T) for kind, a in named],
            self._model.samplerate,
            settings.demucs_bit_depth,
        )

        provenance = stem_provenance(
            self._model_name or settings.demucs_model,
            settings.demucs_shifts,
            settings.demucs_bit_depth,
        )
        logger.info(
            "Separated track %s on %s in %.1fs: residual %s dB, %s",
            track.id,
            device_used,
            elapsed,
            "n/a" if residual_db is None else f"{residual_db:.1f}",
            provenance,
        )

        for kind, _ in named:
            session.add(
                StemFile(
                    track_id=track.id,
                    kind=kind.value,
                    path=str(paths[kind.value]),
                    model_used=provenance,
                    # Track-level reconstruction residual in dB (see module
                    # docstring); same value on every stem row of the track.
                    separation_quality=residual_db,
                )
            )

        track.state = self.output_state.value
        session.commit()
        return True
