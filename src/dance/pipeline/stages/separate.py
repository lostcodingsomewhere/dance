"""
Stem separation stage (Demucs).

Writes one StemFile row per (track, kind). Demucs's source names map to our
StemKind enum 1:1: drums, bass, vocals, other.
"""

from __future__ import annotations

import logging
from pathlib import Path

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


class StemSeparationStage:
    """Separates each track into 4 stems via Demucs."""

    name = "separate"
    input_state = TrackState.ANALYZED
    output_state = TrackState.SEPARATED
    error_state = TrackState.ERROR

    def __init__(self) -> None:
        self._model = None
        self._device: str | None = None
        self._model_name: str | None = None

    # ------------------------------------------------------------------

    def _ensure_model(self, settings: Settings) -> None:
        if self._model is not None and self._model_name == settings.demucs_model:
            return
        if not _DEMUCS_OK:
            raise RuntimeError("Demucs is not installed")

        self._device = pick_device(settings.demucs_device)
        logger.info("Loading Demucs model %s on %s", settings.demucs_model, self._device)
        model = get_model(settings.demucs_model)
        model.to(self._device)
        model.eval()
        self._model = model
        self._model_name = settings.demucs_model

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
        audio_np, _ = librosa.load(
            str(path), sr=self._model.samplerate, mono=False
        )
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np], axis=0)
        elif audio_np.shape[0] > 2:
            audio_np = audio_np[:2]
        wav = torch.from_numpy(audio_np).to(self._device)

        with torch.no_grad():
            sources = apply_model(
                self._model,
                wav[None],
                device=self._device,
                progress=False,
                num_workers=0,
            )[0]

        # Drop stale rows so we don't accumulate duplicates on a retry.
        if existing:
            for s in existing:
                session.delete(s)
            session.commit()

        out_dir = settings.stems_dir / track.file_hash[:8]
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, source_name in enumerate(self._model.sources):
            kind = _DEMUCS_SOURCE_MAP.get(source_name)
            if kind is None:
                logger.warning("Skipping unknown demucs source: %s", source_name)
                continue
            stem_path = out_dir / f"{kind.value}.wav"
            # soundfile expects shape (samples, channels), torch tensors are
            # (channels, samples) — transpose before write.
            stem_np = sources[i].cpu().numpy().T
            sf.write(str(stem_path), stem_np, self._model.samplerate, subtype="PCM_16")
            session.add(
                StemFile(
                    track_id=track.id,
                    kind=kind.value,
                    path=str(stem_path),
                    model_used=self._model_name,
                )
            )

        track.state = self.output_state.value
        session.commit()
        return True
