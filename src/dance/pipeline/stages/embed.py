"""CLAP audio embedding stage.

Computes one embedding for the full mix and one per stem (drums/bass/vocals/
other) and stores them in ``track_embeddings``. This is the FINAL per-track
pipeline stage — on success the track transitions to ``COMPLETE``.

Each embedding is deterministic and covers the whole audio: the audio is cut
into the model's native 10 s windows (never randomly cropped), near-silent
windows are skipped, up to ``settings.clap_max_windows`` are taken evenly
spread, and their embeddings are mean-pooled (see
:mod:`dance.pipeline.utils.clap_windows`). Rows are written under
:data:`~dance.core.embedding_scheme.CLAP_EMBED_VERSION`; rows from any other
scheme for the same (track, stem, model) are deleted so readers never mix
schemes. ``dance embed`` re-embeds COMPLETE tracks still on an old scheme.

Heavy ML imports (torch, transformers, librosa) are gated behind a flag so
this module can still be imported without them; the dispatcher's lazy
registration will catch the resulting ImportError if the gate is False at
process time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import Track, TrackEmbedding, TrackState, now_utc
from dance.core.embedding_scheme import CLAP_EMBED_VERSION, current_scheme
from dance.core.serialization import encode_embedding
from dance.pipeline.utils.clap_windows import (
    extract_windows,
    pool_embeddings,
    select_windows,
)
from dance.pipeline.utils.db import get_stems_for_track, upsert
from dance.pipeline.utils.device import pick_device

logger = logging.getLogger(__name__)

try:
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    _CLAP_OK = True
except ImportError:  # pragma: no cover — exercised only on installs missing deps
    _CLAP_OK = False
    logger.warning("CLAP dependencies unavailable — embedding stage disabled")


# CLAP's expected sample rate.
CLAP_SAMPLE_RATE = 48000
# Fallback window length when the processor doesn't say (10 s at 48 kHz —
# laion/clap-htsat-unfused's ``nb_max_samples``).
_DEFAULT_WINDOW = 10 * CLAP_SAMPLE_RATE
# Windows per forward pass: bounds peak activation memory on MPS while still
# amortising per-call overhead.
_BATCH = 8


class EmbeddingStage:
    """Embed the full mix + each stem with a CLAP model."""

    name = "embed"
    input_state = TrackState.REGIONS_DETECTED
    inflight_state = TrackState.EMBEDDING
    output_state = TrackState.COMPLETE
    error_state = TrackState.ERROR
    # 2 workers + GPU semaphore. Same MPS pool as separate.
    concurrency = 2
    uses_gpu = True

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._model_name: str | None = None
        # Samples per window — the processor's max_length, so it never crops.
        self._window = _DEFAULT_WINDOW
        self._max_windows = int(Settings.model_fields["clap_max_windows"].default)
        self._silence_db = float(Settings.model_fields["clap_silence_db"].default)

    def _ensure_model(self, settings: Settings, *, local_files_only: bool = False) -> None:
        """Lazy-load the CLAP model + processor, cached on the instance."""
        if self._model is not None and self._model_name == settings.clap_model:
            return
        if not _CLAP_OK:
            raise RuntimeError("transformers/torch/librosa not installed")

        device = pick_device(settings.clap_device)
        logger.info("Loading CLAP model %s on %s", settings.clap_model, device)

        processor = ClapProcessor.from_pretrained(
            settings.clap_model, local_files_only=local_files_only
        )
        model = ClapModel.from_pretrained(settings.clap_model, local_files_only=local_files_only)

        # MPS occasionally rejects HF model casts; fall back to CPU cleanly.
        if device == "mps":
            try:
                model = model.to(device)
            except (RuntimeError, NotImplementedError) as exc:
                logger.warning("MPS load failed (%s); falling back to CPU", exc)
                device = "cpu"
                model = model.to(device)
        else:
            model = model.to(device)
        model.eval()

        extractor = getattr(processor, "feature_extractor", None)
        sr = int(getattr(extractor, "sampling_rate", 0) or CLAP_SAMPLE_RATE)
        if sr != CLAP_SAMPLE_RATE:
            raise RuntimeError(
                f"{settings.clap_model} expects {sr} Hz audio; this stage loads "
                f"{CLAP_SAMPLE_RATE} Hz"
            )

        self._device = device
        self._model = model
        self._processor = processor
        self._model_name = settings.clap_model
        self._window = int(getattr(extractor, "nb_max_samples", 0) or _DEFAULT_WINDOW)

    def _configure(self, settings: Settings) -> None:
        """Pick up the windowing knobs (cheap; called per track)."""
        self._max_windows = int(settings.clap_max_windows)
        self._silence_db = float(settings.clap_silence_db)

    # ------------------------------------------------------------------

    def _encode(self, audio: np.ndarray) -> np.ndarray | None:
        """Embed one 48 kHz mono float32 audio: windowed, pooled, unit-norm.

        Returns ``None`` when every window is near-silent (below
        ``settings.clap_silence_db``) — there is nothing to describe.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("model not loaded — call _ensure_model first")

        starts = select_windows(
            audio,
            self._window,
            max_windows=self._max_windows,
            silence_db=self._silence_db,
        )
        if not starts:
            return None
        windows = extract_windows(audio, starts, self._window)
        parts = [
            self._encode_windows(windows[i : i + _BATCH]) for i in range(0, len(windows), _BATCH)
        ]
        return pool_embeddings(np.concatenate(parts, axis=0))

    def _encode_windows(self, windows: list[np.ndarray]) -> np.ndarray:
        """One forward pass over equal-length windows -> ``(n, dim)`` float32."""
        if any(w.shape[0] != self._window for w in windows):
            # Longer and the extractor random-crops; shorter and it pads.
            raise ValueError("every window must be exactly max_length samples")
        inputs = self._processor(audio=windows, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)
        # `get_audio_features` in transformers >=4.45 returns a
        # ``BaseModelOutputWithPooling`` (the pooled embedding lives in
        # ``pooler_output``). Older versions returned the tensor directly.
        if hasattr(features, "pooler_output"):
            tensor = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            # Some variants only expose last_hidden_state — mean-pool it.
            tensor = features.last_hidden_state.mean(dim=1)
        else:
            tensor = features
        out: np.ndarray = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return out

    @staticmethod
    def _load_audio(path: Path) -> np.ndarray:
        """Load a file as 48kHz mono float32."""
        audio, _ = librosa.load(str(path), sr=CLAP_SAMPLE_RATE, mono=True)
        return audio.astype(np.float32, copy=False)

    def encode_text(self, text: str) -> np.ndarray:
        """Embed a text query into the same 512-d space as audio embeddings.

        CLAP is a joint audio/text model — text and audio embeddings live in
        the same space, so cosine similarity between a text query and stored
        audio embeddings ranks tracks by semantic match.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("model not loaded — call _ensure_model first")

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        if hasattr(features, "pooler_output"):
            tensor = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            tensor = features.last_hidden_state.mean(dim=1)
        else:
            tensor = features
        out: np.ndarray = tensor.cpu().numpy()[0].astype(np.float32, copy=False)
        return out

    # ------------------------------------------------------------------

    def process(self, session: Session, track: Track, settings: Settings) -> bool:
        track.state = TrackState.EMBEDDING.value
        session.commit()

        try:
            written, silent = self.embed_track(session, track, settings)
            track.state = self.output_state.value
            session.commit()
            logger.info(
                "Embedded track %s (%d rows, %d silent audio skipped)",
                track.id,
                written,
                silent,
            )
            return True

        except Exception as exc:  # noqa: BLE001 — stage reports via state
            # Drop anything half-written before recording the failure.
            session.rollback()
            logger.error("Embedding failed for track %s: %s", track.id, exc)
            track.state = self.error_state.value
            track.error_message = f"{self.name}: {exc}"[:500]
            session.commit()
            return False

    def embed_track(self, session: Session, track: Track, settings: Settings) -> tuple[int, int]:
        """Embed the mix + every stem of ``track`` under the current scheme.

        Every audio is embedded before anything is written, so a failure
        part-way leaves the stored rows exactly as they were. Does not commit
        and does not touch ``track.state`` — :meth:`process` and
        ``dance embed`` own that. Returns ``(rows_written, silent_audio)``.
        """
        self._ensure_model(settings)
        self._configure(settings)

        mix_path = Path(track.file_path)
        if not mix_path.exists():
            raise FileNotFoundError(f"Track audio missing: {mix_path}")
        targets: list[tuple[int | None, Path]] = [(None, mix_path)]
        for stem in get_stems_for_track(session, int(track.id)):
            stem_path = Path(stem.path)
            if not stem_path.exists():
                raise FileNotFoundError(f"Stem audio missing: {stem_path}")
            targets.append((int(stem.id), stem_path))

        vectors = [(stem_id, self._encode(self._load_audio(path))) for stem_id, path in targets]

        now = now_utc()
        written = silent = 0
        for stem_id, vec in vectors:
            if vec is None:
                silent += 1
                logger.info(
                    "Track %s %s: every window below %.0f dBFS — no embedding stored",
                    track.id,
                    "mix" if stem_id is None else f"stem {stem_id}",
                    self._silence_db,
                )
            else:
                written += 1
            self._store(session, track, stem_file_id=stem_id, embedding=vec, when=now)
        return written, silent

    # ------------------------------------------------------------------

    def _store(
        self,
        session: Session,
        track: Track,
        *,
        stem_file_id: int | None,
        embedding: np.ndarray | None,
        when: datetime,
    ) -> None:
        """Make the current-scheme row the only row for (track, stem, model).

        Rows written by any other scheme (including the legacy NULL version)
        are deleted, so no reader can pick up a stale vector. ``embedding``
        ``None`` (every window silent) deletes the current row too.
        """
        same_slot = session.query(TrackEmbedding).filter(
            TrackEmbedding.track_id == track.id,
            TrackEmbedding.stem_file_id.is_(None)
            if stem_file_id is None
            else TrackEmbedding.stem_file_id == stem_file_id,
            TrackEmbedding.model == self._model_name,
        )
        if embedding is not None:
            same_slot = same_slot.filter(
                or_(
                    TrackEmbedding.model_version.is_(None),
                    TrackEmbedding.model_version != CLAP_EMBED_VERSION,
                )
            )
        same_slot.delete(synchronize_session="fetch")
        if embedding is None:
            return

        upsert(
            session,
            TrackEmbedding,
            where={
                "track_id": track.id,
                "stem_file_id": stem_file_id,
                "model": self._model_name,
                "model_version": CLAP_EMBED_VERSION,
            },
            dim=int(embedding.shape[0]),
            embedding=encode_embedding(embedding),
            created_at=when,
        )


def stale_track_ids(session: Session, model: str) -> list[int]:
    """COMPLETE tracks whose ``model`` embeddings are not all current.

    Stale means: any row for ``model`` written by another scheme (including
    the legacy NULL version), or no current full-mix row. ``dance embed``
    re-embeds these after a scheme change.
    """
    old_scheme = select(TrackEmbedding.track_id).where(
        TrackEmbedding.model == model,
        or_(
            TrackEmbedding.model_version.is_(None),
            TrackEmbedding.model_version != CLAP_EMBED_VERSION,
        ),
    )
    has_current_mix = select(TrackEmbedding.track_id).where(
        TrackEmbedding.model == model,
        TrackEmbedding.stem_file_id.is_(None),
        current_scheme(),
    )
    rows = (
        session.query(Track.id)
        .filter(
            Track.state == TrackState.COMPLETE.value,
            or_(Track.id.in_(old_scheme), Track.id.notin_(has_current_mix)),
        )
        .order_by(Track.id)
        .all()
    )
    return [int(r[0]) for r in rows]
