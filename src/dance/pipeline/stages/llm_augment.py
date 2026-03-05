"""
LLM augmentation stage using Qwen2-Audio.

Adds rich metadata to tracks:
- Subgenre classification
- Mood tags
- Notable musical elements
- Contextual cue point names
- Quality validation (BPM/key verification)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from dance.core.database import Analysis, Track, TrackState

logger = logging.getLogger(__name__)

# Lazy import to avoid loading heavy deps if not needed
_llm_model = None


def _get_llm_model():
    """Lazily initialize the LLM model."""
    global _llm_model
    if _llm_model is None:
        try:
            from dance.llm import QwenAudioModel
            from dance.config import get_settings

            settings = get_settings()
            _llm_model = QwenAudioModel(
                model_name=settings.llm_model,
                device=settings.llm_device,
                quantization=settings.llm_quantize,
            )
        except ImportError as e:
            logger.error(f"LLM dependencies not installed: {e}")
            raise
    return _llm_model


class LLMAugmentStage:
    """
    LLM-based audio understanding for enhanced metadata.

    Uses Qwen2-Audio to:
    1. Classify subgenre (tech house, melodic techno, etc.)
    2. Generate mood tags
    3. Identify notable musical elements
    4. Create contextual cue point names
    5. Validate Essentia analysis (BPM, key)
    """

    def __init__(self, skip_validation: bool = False):
        """
        Initialize the LLM augmentation stage.

        Args:
            skip_validation: If True, skip BPM/key validation
        """
        self.skip_validation = skip_validation
        self._model = None

    def _ensure_model(self):
        """Ensure the model is loaded."""
        if self._model is None:
            self._model = _get_llm_model()
        return self._model

    def augment_track(self, session: Session, track: Track) -> bool:
        """
        Run LLM augmentation on a track.

        Args:
            session: Database session.
            track: Track to augment (must have Analysis record).

        Returns:
            True if augmentation succeeded, False otherwise.
        """
        # Update state
        track.state = TrackState.LLM_AUGMENTING.value
        session.commit()

        try:
            # Get existing analysis
            analysis = session.query(Analysis).filter_by(track_id=track.id).first()
            if analysis is None:
                raise ValueError(f"Track {track.id} has no analysis record")

            file_path = Path(track.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Track file not found: {file_path}")

            logger.info(f"LLM analyzing: {track.artist} - {track.title}")

            # Get the model and run analysis
            model = self._ensure_model()
            result = model.analyze(
                audio_path=file_path,
                bpm=analysis.bpm,
                key=analysis.key_standard,
                camelot_key=analysis.key_camelot,
                energy=analysis.floor_energy,
            )

            # Check for errors
            if result.error and not result.subgenre:
                # Complete failure
                raise RuntimeError(f"LLM analysis failed: {result.error}")

            # Update analysis with LLM results
            self._update_analysis(analysis, result)

            # Update track state
            track.state = TrackState.LLM_AUGMENTED.value
            session.commit()

            # Log summary
            tags_str = ", ".join(result.mood_tags[:3]) if result.mood_tags else "none"
            logger.info(
                f"LLM augmented: {track.artist} - {track.title} | "
                f"Subgenre: {result.subgenre or 'unknown'}, Mood: {tags_str}"
            )
            return True

        except Exception as e:
            logger.error(f"LLM augmentation failed for {track.title}: {e}")
            # Don't set ERROR state - allow pipeline to continue without LLM
            # Just mark as LLM_AUGMENTED with null fields
            track.state = TrackState.LLM_AUGMENTED.value
            if analysis:
                analysis.llm_model = "error"
                analysis.llm_analyzed_at = datetime.utcnow()
            session.commit()
            return False

    def _update_analysis(self, analysis: Analysis, result) -> None:
        """Update analysis record with LLM results."""
        # Rich tagging
        analysis.llm_subgenre = result.subgenre
        analysis.llm_mood_tags = json.dumps(result.mood_tags) if result.mood_tags else None
        analysis.llm_notable_elements = json.dumps(result.notable_elements) if result.notable_elements else None
        analysis.llm_energy_curve = result.energy_curve
        analysis.llm_dj_notes = result.dj_notes

        # Cue contexts
        analysis.llm_cue_contexts = json.dumps(result.cue_contexts) if result.cue_contexts else None

        # Quality validation
        analysis.llm_bpm_validated = result.bpm_validated
        analysis.llm_bpm_suggestion = result.bpm_suggestion
        analysis.llm_key_validated = result.key_validated
        analysis.llm_key_suggestion = result.key_suggestion
        analysis.llm_quality_issues = json.dumps(result.quality_issues) if result.quality_issues else None
        analysis.llm_is_dj_track = result.is_dj_track

        # Metadata
        analysis.llm_model = result.model_name
        analysis.llm_analyzed_at = datetime.utcnow()


def is_llm_available() -> bool:
    """Check if LLM augmentation is available."""
    try:
        from dance.llm import QwenAudioModel
        model = QwenAudioModel()
        return model.is_available()
    except ImportError:
        return False


def get_llm_status() -> dict:
    """Get LLM model status information."""
    try:
        model = _get_llm_model()
        return model.get_status()
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def augment_separated_tracks(
    session: Session,
    limit: Optional[int] = None,
    skip_if_unavailable: bool = True,
) -> tuple[int, int, int]:
    """
    Augment all tracks in SEPARATED state with LLM.

    Args:
        session: Database session.
        limit: Maximum number of tracks to process.
        skip_if_unavailable: If True and LLM unavailable, mark as LLM_AUGMENTED and continue.

    Returns:
        Tuple of (success_count, skipped_count, error_count).
    """
    # Check if LLM is available
    if not is_llm_available():
        if skip_if_unavailable:
            # Mark all separated tracks as LLM_AUGMENTED so pipeline can continue
            query = session.query(Track).filter(
                Track.state == TrackState.SEPARATED.value
            )
            if limit:
                query = query.limit(limit)

            tracks = query.all()
            for track in tracks:
                track.state = TrackState.LLM_AUGMENTED.value

            session.commit()
            logger.warning(f"LLM not available, skipped {len(tracks)} tracks")
            return 0, len(tracks), 0
        else:
            raise RuntimeError("LLM not available and skip_if_unavailable=False")

    # Process tracks
    query = session.query(Track).filter(
        Track.state == TrackState.SEPARATED.value
    )
    if limit:
        query = query.limit(limit)

    tracks = query.all()
    stage = LLMAugmentStage()

    success = 0
    skipped = 0
    errors = 0

    for track in tracks:
        try:
            if stage.augment_track(session, track):
                success += 1
            else:
                # Non-fatal error, track still moves forward
                skipped += 1
        except Exception as e:
            logger.error(f"Critical error augmenting {track.title}: {e}")
            errors += 1

    return success, skipped, errors


def reanalyze_with_llm(
    session: Session,
    track_ids: Optional[list[int]] = None,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """
    Re-run LLM analysis on already-processed tracks.

    Useful for updating tags with a newer model or after prompt changes.

    Args:
        session: Database session.
        track_ids: Specific track IDs to reanalyze. If None, reanalyze all COMPLETE tracks.
        limit: Maximum number of tracks to process.

    Returns:
        Tuple of (success_count, error_count).
    """
    if track_ids:
        query = session.query(Track).filter(Track.id.in_(track_ids))
    else:
        query = session.query(Track).filter(
            Track.state == TrackState.COMPLETE.value
        )

    if limit:
        query = query.limit(limit)

    tracks = query.all()
    stage = LLMAugmentStage()

    success = 0
    errors = 0

    for track in tracks:
        # Temporarily change state for processing
        original_state = track.state
        try:
            if stage.augment_track(session, track):
                success += 1
            else:
                errors += 1
        finally:
            # Restore original state (track stays COMPLETE)
            track.state = original_state
            session.commit()

    return success, errors
