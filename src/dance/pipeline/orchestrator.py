"""
Pipeline orchestrator - coordinates all processing stages.

Stages:
1. Ingest: Scan files, compute hashes, extract metadata
2. Analyze: BPM, key, energy, mood
3. Separate: Demucs stem separation (optional)
4. LLM Augment: Qwen2-Audio tagging and validation (optional)
5. Detect Cues: Phrase detection and cue point placement
6. Export: Write to Traktor collection.nml
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import Track, TrackState
from dance.pipeline.stages.ingest import IngestStage
from dance.pipeline.stages.analyze import AnalysisStage
from dance.pipeline.stages.separate import StemSeparationStage
from dance.pipeline.stages.detect_cues import CueDetectionStage

logger = logging.getLogger(__name__)


# Lazy import for LLM stage to avoid loading heavy deps
def _get_llm_stage():
    """Lazily import and return LLM augmentation stage."""
    try:
        from dance.pipeline.stages.llm_augment import LLMAugmentStage
        return LLMAugmentStage()
    except ImportError:
        logger.warning("LLM dependencies not available")
        return None


class PipelineOrchestrator:
    """
    Coordinates all pipeline stages.

    Tracks flow through stages:
    PENDING → ANALYZED → SEPARATED → LLM_AUGMENTED → COMPLETE
    (stages can be skipped via settings)
    """

    def __init__(self, settings: Settings, session: Session):
        self.settings = settings
        self.session = session

        # Initialize stages
        self.ingest_stage = IngestStage(settings.library_dir)
        self.analyze_stage = AnalysisStage()
        self.separate_stage = StemSeparationStage(settings) if not settings.skip_stems else None
        self.llm_stage = None if settings.skip_llm else _get_llm_stage()
        self.cue_stage = CueDetectionStage()

    def ingest_new_files(self) -> dict:
        """
        Scan library directory and ingest new files.

        Returns:
            Dict with counts: {new, updated, unchanged, errors}
        """
        logger.info("Scanning for new files...")
        result = self.ingest_stage.scan_and_ingest(self.session)

        return {
            "new": result.new,
            "updated": result.updated,
            "unchanged": result.unchanged,
            "errors": result.errors,
        }

    def process_pending(
        self,
        limit: Optional[int] = None,
        skip_stems: bool = False,
        skip_llm: bool = False,
    ) -> dict:
        """
        Process all pending tracks through the pipeline.

        Args:
            limit: Maximum number of tracks to process.
            skip_stems: Skip stem separation stage.
            skip_llm: Skip LLM augmentation stage.

        Returns:
            Dict with stage results.
        """
        results = {
            "analyzed": 0,
            "separated": 0,
            "llm_augmented": 0,
            "cues_detected": 0,
            "errors": 0,
        }

        # Get pending tracks
        query = self.session.query(Track).filter(
            Track.state == TrackState.PENDING.value
        )
        if limit:
            query = query.limit(limit)
        pending_tracks = query.all()

        logger.info(f"Processing {len(pending_tracks)} pending tracks")

        # Stage 1: Analyze
        for track in pending_tracks:
            if self.analyze_stage.analyze_track(self.session, track):
                results["analyzed"] += 1
            else:
                results["errors"] += 1

        # Stage 2: Stem separation (if enabled)
        if not skip_stems and self.separate_stage:
            analyzed_tracks = self.session.query(Track).filter(
                Track.state == TrackState.ANALYZED.value
            ).all()

            for track in analyzed_tracks:
                if self.separate_stage.separate_track(self.session, track):
                    results["separated"] += 1
                else:
                    # Continue to LLM stage even if stem separation fails
                    # Mark as SEPARATED so it can proceed
                    track.state = TrackState.SEPARATED.value
                    self.session.commit()
                    logger.warning(f"Stem separation failed for {track.title}, continuing")
        else:
            # If skipping stems, move analyzed tracks to SEPARATED state
            analyzed_tracks = self.session.query(Track).filter(
                Track.state == TrackState.ANALYZED.value
            ).all()
            for track in analyzed_tracks:
                track.state = TrackState.SEPARATED.value
            self.session.commit()

        # Stage 3: LLM augmentation (if enabled)
        if not skip_llm and self.llm_stage:
            separated_tracks = self.session.query(Track).filter(
                Track.state == TrackState.SEPARATED.value
            ).all()

            for track in separated_tracks:
                if self.llm_stage.augment_track(self.session, track):
                    results["llm_augmented"] += 1
                else:
                    # Non-fatal - mark as augmented and continue
                    logger.warning(f"LLM augmentation failed for {track.title}, continuing")
        else:
            # If skipping LLM, move separated tracks to LLM_AUGMENTED state
            separated_tracks = self.session.query(Track).filter(
                Track.state == TrackState.SEPARATED.value
            ).all()
            for track in separated_tracks:
                track.state = TrackState.LLM_AUGMENTED.value
            self.session.commit()

        # Stage 4: Cue detection
        # Process tracks that have been through LLM augmentation (or skipped it)
        ready_for_cues = self.session.query(Track).filter(
            Track.state.in_([
                TrackState.ANALYZED.value,  # Fallback if earlier stages skipped
                TrackState.SEPARATED.value,  # If LLM skipped
                TrackState.LLM_AUGMENTED.value,
            ])
        ).all()

        for track in ready_for_cues:
            if self.cue_stage.detect_cues(self.session, track):
                results["cues_detected"] += 1
            else:
                results["errors"] += 1

        return results

    def process_track(
        self,
        track: Track,
        skip_stems: bool = False,
        skip_llm: bool = False,
    ) -> bool:
        """
        Process a single track through all stages.

        Args:
            track: Track to process.
            skip_stems: Skip stem separation.
            skip_llm: Skip LLM augmentation.

        Returns:
            True if processing completed successfully.
        """
        # Analyze
        if not self.analyze_stage.analyze_track(self.session, track):
            return False

        # Stem separation (optional)
        if not skip_stems and self.separate_stage:
            # Don't fail pipeline if stems fail
            self.separate_stage.separate_track(self.session, track)
        else:
            # Move to next state
            track.state = TrackState.SEPARATED.value
            self.session.commit()

        # LLM augmentation (optional)
        if not skip_llm and self.llm_stage:
            # Don't fail pipeline if LLM fails
            self.llm_stage.augment_track(self.session, track)
        else:
            track.state = TrackState.LLM_AUGMENTED.value
            self.session.commit()

        # Cue detection
        return self.cue_stage.detect_cues(self.session, track)

    def get_status(self) -> dict:
        """
        Get pipeline status - counts of tracks in each state.

        Returns:
            Dict with state counts.
        """
        status = {}
        for state in TrackState:
            count = self.session.query(Track).filter(
                Track.state == state.value
            ).count()
            status[state.value] = count

        status["total"] = self.session.query(Track).count()
        return status


def run_full_pipeline(
    settings: Settings,
    session: Session,
    limit: Optional[int] = None,
    skip_stems: bool = False,
    skip_llm: bool = False,
    skip_export: bool = False,
) -> dict:
    """
    Run the complete pipeline: ingest → analyze → separate → llm → cues → export.

    Args:
        settings: Application settings.
        session: Database session.
        limit: Maximum tracks to process.
        skip_stems: Skip stem separation.
        skip_llm: Skip LLM augmentation.
        skip_export: Skip Traktor export.

    Returns:
        Dict with all stage results.
    """
    orchestrator = PipelineOrchestrator(settings, session)

    results = {
        "ingest": orchestrator.ingest_new_files(),
        "process": orchestrator.process_pending(limit, skip_stems, skip_llm),
        "export": None,
    }

    # Export to Traktor
    if not skip_export and settings.traktor_collection_path:
        from dance.export.traktor import TraktorExporter

        exporter = TraktorExporter(settings)
        results["export"] = exporter.export_all(session)

    return results
