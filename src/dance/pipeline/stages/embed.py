"""CLAP audio embedding stage.

Computes one embedding for the full mix and one per stem (drums/bass/vocals/
other), upserting into ``track_embeddings``. This is the FINAL per-track
pipeline stage — on success the track transitions to ``COMPLETE``.

Heavy ML imports (torch, transformers, librosa) are gated behind a flag so
this module can still be imported without them; the dispatcher's lazy
registration will catch the resulting ImportError if the gate is False at
process time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import now_utc, StemFile, Track, TrackEmbedding, TrackState
from dance.core.serialization import encode_embedding
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


class EmbeddingStage:
    """Embed the full mix + each stem with a CLAP model."""

    name = "embed"
    input_state = TrackState.REGIONS_DETECTED
    output_state = TrackState.COMPLETE
    error_state = TrackState.ERROR

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._model_name: str | None = None
        self._model_version: str | None = None

    def _ensure_model(self, settings: Settings) -> None:
        """Lazy-load the CLAP model + processor, cached on the instance."""
        if self._model is not None and self._model_name == settings.clap_model:
            return
        if not _CLAP_OK:
            raise RuntimeError("transformers/torch/librosa not installed")

        device = pick_device(settings.clap_device)
        logger.info("Loading CLAP model %s on %s", settings.clap_model, device)

        processor = ClapProcessor.from_pretrained(settings.clap_model)
        model = ClapModel.from_pretrained(settings.clap_model)

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

        self._device = device
        self._model = model
        self._processor = processor
        self._model_name = settings.clap_model
        self._model_version = getattr(
            getattr(model, "config", None), "transformers_version", None
        )

    # ------------------------------------------------------------------

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        """Convert one chunk of 48kHz mono float32 audio into a 512-d embedding."""
        if self._model is None or self._processor is None:
            raise RuntimeError("model not loaded — call _ensure_model first")

        inputs = self._processor(
            audio=audio, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        )
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
        return tensor.cpu().numpy()[0].astype(np.float32, copy=False)

    @staticmethod
    def _load_audio(path: Path) -> np.ndarray:
        """Load a file as 48kHz mono float32."""
        audio, _ = librosa.load(str(path), sr=CLAP_SAMPLE_RATE, mono=True)
        return audio.astype(np.float32, copy=False)

    # ------------------------------------------------------------------

    def process(self, session: Session, track: Track, settings: Settings) -> bool:
        track.state = TrackState.EMBEDDING.value
        session.commit()

        try:
            self._ensure_model(settings)

            now = now_utc()

            # Full mix.
            mix_path = Path(track.file_path)
            if not mix_path.exists():
                raise FileNotFoundError(f"Track audio missing: {mix_path}")
            audio = self._load_audio(mix_path)
            self._upsert(session, track, stem_file_id=None, audio=audio, when=now)

            # Per-stem.
            stems = get_stems_for_track(session, track.id)
            for stem in stems:
                stem_path = Path(stem.path)
                if not stem_path.exists():
                    raise FileNotFoundError(f"Stem audio missing: {stem_path}")
                audio = self._load_audio(stem_path)
                self._upsert(
                    session, track, stem_file_id=stem.id, audio=audio, when=now
                )

            track.state = self.output_state.value
            session.commit()
            logger.info(
                "Embedded track %s (1 mix + %d stems)", track.id, len(stems)
            )
            return True

        except Exception as exc:  # noqa: BLE001 — stage reports via state
            logger.error("Embedding failed for track %s: %s", track.id, exc)
            track.state = self.error_state.value
            track.error_message = f"{self.name}: {exc}"[:500]
            session.commit()
            return False

    # ------------------------------------------------------------------

    def _upsert(
        self,
        session: Session,
        track: Track,
        *,
        stem_file_id: int | None,
        audio: np.ndarray,
        when: datetime,
    ) -> None:
        """Compute the embedding and upsert one TrackEmbedding row."""
        embedding = self._encode(audio)
        upsert(
            session,
            TrackEmbedding,
            where={
                "track_id": track.id,
                "stem_file_id": stem_file_id,
                "model": self._model_name,
                "model_version": self._model_version,
            },
            dim=int(embedding.shape[0]),
            embedding=encode_embedding(embedding),
            created_at=when,
        )
