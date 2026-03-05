"""
Stem separation stage using Demucs.

Separates tracks into drums, bass, vocals, and other (synths/pads).
Uses htdemucs_ft model for best quality.
Supports MPS acceleration on Apple Silicon.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import Stems, Track, TrackState

logger = logging.getLogger(__name__)

# Try to import demucs
try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    logger.warning("Demucs not available - stem separation disabled")


class StemSeparationStage:
    """
    Stem separation using Demucs.

    Separates tracks into:
    - drums: Kick, snare, hats, percussion
    - bass: Bass lines
    - vocals: Vocals and vocal-like sounds
    - other: Synths, pads, FX

    Uses MPS acceleration on Apple Silicon Macs.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stems_dir = settings.stems_dir
        self.model_name = "htdemucs_ft"  # Best quality model
        self.model = None
        self.device = None

    def _get_device(self) -> str:
        """Get the best available device for inference."""
        if not DEMUCS_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            # Apple Silicon MPS acceleration
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        """Lazy load the Demucs model."""
        if not DEMUCS_AVAILABLE:
            raise RuntimeError(
                "Demucs not installed. Install with: pip install demucs"
            )

        if self.model is None:
            logger.info(f"Loading Demucs model: {self.model_name}")
            self.device = self._get_device()
            logger.info(f"Using device: {self.device}")

            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def _get_stem_dir(self, track: Track) -> Path:
        """Get directory for track's stems."""
        # Use first 8 chars of hash for directory name
        stem_dir = self.stems_dir / track.file_hash[:8]
        stem_dir.mkdir(parents=True, exist_ok=True)
        return stem_dir

    def separate_track(self, session: Session, track: Track) -> bool:
        """
        Separate a track into stems.

        Args:
            session: Database session.
            track: Track to separate.

        Returns:
            True if separation succeeded.
        """
        # Check if already separated
        existing = session.query(Stems).filter_by(track_id=track.id).first()
        if existing:
            logger.info(f"Stems already exist for: {track.title}")
            return True

        if not DEMUCS_AVAILABLE:
            logger.warning("Skipping stem separation - Demucs not available")
            return False

        # Update state
        track.state = TrackState.SEPARATING.value
        session.commit()

        try:
            self._load_model()

            file_path = Path(track.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Track file not found: {file_path}")

            logger.info(f"Separating stems: {track.artist} - {track.title}")

            # Load audio
            wav, sr = torchaudio.load(str(file_path))

            # Resample if needed
            if sr != self.model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
                wav = resampler(wav)

            # Ensure stereo
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]

            # Move to device
            wav = wav.to(self.device)

            # Apply model
            with torch.no_grad():
                # Add batch dimension
                sources = apply_model(
                    self.model,
                    wav[None],
                    device=self.device,
                    progress=True,
                    num_workers=0,
                )[0]

            # Save stems
            stem_dir = self._get_stem_dir(track)
            stem_paths = {}

            for i, source_name in enumerate(self.model.sources):
                stem_path = stem_dir / f"{source_name}.wav"
                # Move to CPU for saving
                stem_audio = sources[i].cpu()
                torchaudio.save(
                    str(stem_path),
                    stem_audio,
                    self.model.samplerate,
                )
                stem_paths[source_name] = str(stem_path)
                logger.debug(f"Saved stem: {stem_path}")

            # Store in database
            stems = Stems(
                track_id=track.id,
                drums_path=stem_paths.get("drums"),
                bass_path=stem_paths.get("bass"),
                vocals_path=stem_paths.get("vocals"),
                other_path=stem_paths.get("other"),
                model_used=self.model_name,
            )
            session.add(stems)

            # Update track state
            track.state = TrackState.SEPARATED.value
            session.commit()

            logger.info(f"Separated: {track.artist} - {track.title}")
            return True

        except Exception as e:
            logger.error(f"Stem separation failed for {track.title}: {e}")
            track.state = TrackState.ERROR.value
            track.error_message = f"Stem separation failed: {str(e)[:400]}"
            session.commit()
            return False

    def get_stem_paths(self, session: Session, track: Track) -> Optional[dict]:
        """
        Get paths to separated stems for a track.

        Returns:
            Dict with stem paths, or None if not separated.
        """
        stems = session.query(Stems).filter_by(track_id=track.id).first()
        if not stems:
            return None

        return {
            "drums": stems.drums_path,
            "bass": stems.bass_path,
            "vocals": stems.vocals_path,
            "other": stems.other_path,
        }


def separate_pending_tracks(
    session: Session,
    settings: Settings,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """
    Separate all tracks in ANALYZED state.

    Args:
        session: Database session.
        settings: Application settings.
        limit: Maximum number of tracks to process.

    Returns:
        Tuple of (success_count, error_count).
    """
    if settings.skip_stems:
        logger.info("Stem separation disabled in settings")
        return 0, 0

    query = session.query(Track).filter(
        Track.state == TrackState.ANALYZED.value
    )

    if limit:
        query = query.limit(limit)

    tracks = query.all()

    if not tracks:
        return 0, 0

    stage = StemSeparationStage(settings)
    success = 0
    errors = 0

    for track in tracks:
        if stage.separate_track(session, track):
            success += 1
        else:
            errors += 1

    return success, errors
