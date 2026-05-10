"""
Stage dispatcher and registry.

The dispatcher is intentionally dumb: given a list of registered stages, it
finds tracks in each stage's input_state and runs the stage. It does NOT
encode the order — order emerges from the input_state/output_state chain that
the stages themselves declare.

To add a new stage:
    1. Implement the Stage protocol (see ``stage.py``)
    2. Register it: ``dispatcher.register(MyStage())``

The dispatcher walks the registered stages until no track changes state — so
a single ``run()`` call processes everything end-to-end.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import Track, TrackState
from dance.pipeline.events import EventBus, StageEvent
from dance.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class Dispatcher:
    """Discovers and runs registered stages until no track is ready."""

    def __init__(self, settings: Settings, session: Session) -> None:
        self.settings = settings
        self.session = session
        self.events = EventBus()
        self._stages: list[Stage] = []

        # Default progress logger.
        self.events.subscribe(_default_logger)

        # Auto-register the standard set; callers can clear/replace if they want.
        self._register_default_stages()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, stage: Stage) -> None:
        """Add a stage to the pipeline."""
        if not isinstance(stage, Stage):
            raise TypeError(f"{stage!r} does not satisfy the Stage protocol")
        if any(s.name == stage.name for s in self._stages):
            raise ValueError(f"Stage already registered: {stage.name}")
        self._stages.append(stage)
        logger.debug("Registered stage: %s (%s -> %s)", stage.name, stage.input_state, stage.output_state)

    def clear(self) -> None:
        self._stages.clear()

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(self._stages)

    def _register_default_stages(self) -> None:
        """Wire the standard pipeline.

        Stages are registered lazily so optional ones (which import heavy deps
        like demucs/transformers) don't blow up the dispatcher at import time
        if those packages aren't installed.
        """
        from dance.pipeline.stages.analyze import AnalysisStage
        from dance.pipeline.stages.ingest import IngestStage  # noqa: F401 (used in .ingest())

        self.register(AnalysisStage())

        try:
            from dance.pipeline.stages.separate import StemSeparationStage
            self.register(StemSeparationStage())
        except ImportError as e:
            logger.warning("Stem separation stage unavailable: %s", e)

        try:
            from dance.pipeline.stages.analyze_stems import StemAnalysisStage
            self.register(StemAnalysisStage())
        except ImportError as e:
            logger.warning("Stem analysis stage unavailable: %s", e)

        try:
            from dance.pipeline.stages.detect_regions import RegionDetectionStage
            self.register(RegionDetectionStage())
        except ImportError as e:
            logger.warning("Region detection stage unavailable: %s", e)

        try:
            from dance.pipeline.stages.embed import EmbeddingStage
            self.register(EmbeddingStage())
        except ImportError as e:
            logger.warning("Embedding stage unavailable: %s", e)

    # ------------------------------------------------------------------
    # Ingest (lives outside the state machine; finds NEW files)
    # ------------------------------------------------------------------

    def ingest(self) -> dict[str, int]:
        from dance.pipeline.stages.ingest import IngestStage

        ingest = IngestStage(self.settings.library_dir)
        result = ingest.scan_and_ingest(self.session)
        return {
            "new": result.new,
            "updated": result.updated,
            "unchanged": result.unchanged,
            "errors": result.errors,
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        limit: int | None = None,
        skip: Iterable[str] | None = None,
        track_id: int | None = None,
    ) -> dict[str, dict[str, int]]:
        """Run every registered stage until none can make progress.

        Args:
            limit: Max tracks per stage per pass.
            skip: Stage names to skip (auto-advances their state so the pipeline keeps flowing).
            track_id: Restrict to a single track.

        Returns:
            Dict of stage_name -> {processed, errors, skipped}.
        """
        skip_set = {s for s in (skip or []) if s}
        totals: dict[str, dict[str, int]] = {}

        # Outer loop: keep iterating until nothing changed in a full pass.
        # Bounded by 2 * number_of_stages to prevent runaway state oscillation.
        max_passes = max(2, 2 * len(self._stages))
        for _ in range(max_passes):
            changed = False
            for stage in self._stages:
                counts = totals.setdefault(stage.name, {"processed": 0, "errors": 0, "skipped": 0})

                if stage.name in skip_set:
                    n = self._auto_advance(stage, track_id=track_id, limit=limit)
                    counts["skipped"] += n
                    if n:
                        changed = True
                    continue

                processed, errors = self._run_stage(stage, track_id=track_id, limit=limit)
                counts["processed"] += processed
                counts["errors"] += errors
                if processed or errors:
                    changed = True

            if not changed:
                break

        return totals

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pending_query(self, stage: Stage, track_id: int | None, limit: int | None):
        q = self.session.query(Track).filter(Track.state == stage.input_state.value)
        if track_id is not None:
            q = q.filter(Track.id == track_id)
        if limit is not None:
            q = q.limit(limit)
        return q

    def _run_stage(self, stage: Stage, track_id: int | None, limit: int | None) -> tuple[int, int]:
        tracks = self._pending_query(stage, track_id, limit).all()
        processed = 0
        errors = 0

        for track in tracks:
            t0 = time.monotonic()
            self.events.emit("started", stage.name, track)
            try:
                ok = stage.process(self.session, track, self.settings)
            except Exception as exc:  # noqa: BLE001 — stages report via state, we still need to log
                logger.exception("Stage %s crashed on track %s", stage.name, track.id)
                # Roll back any partial writes from the crashed stage before
                # recording the error state — otherwise the commit may fail
                # for unrelated reasons.
                self.session.rollback()
                track = self.session.get(Track, track.id) or track
                track.state = stage.error_state.value
                track.error_message = f"{stage.name}: {exc}"[:500]
                self.session.commit()
                self.events.emit("failed", stage.name, track, error=str(exc))
                errors += 1
                continue

            dt_ms = int((time.monotonic() - t0) * 1000)
            if ok:
                processed += 1
                self.events.emit("completed", stage.name, track, duration_ms=dt_ms)
            else:
                errors += 1
                self.events.emit(
                    "failed",
                    stage.name,
                    track,
                    duration_ms=dt_ms,
                    error=track.error_message or "stage returned False",
                )

        return processed, errors

    def _auto_advance(self, stage: Stage, track_id: int | None, limit: int | None) -> int:
        """When a stage is skipped, fast-forward tracks past it."""
        tracks = self._pending_query(stage, track_id, limit).all()
        for track in tracks:
            track.state = stage.output_state.value
        if tracks:
            self.session.commit()
        return len(tracks)


# ---------------------------------------------------------------------------
# Default event logger
# ---------------------------------------------------------------------------


def _default_logger(event: StageEvent) -> None:
    if event.kind == "started":
        logger.info("▶ %s — track %s (%s)", event.stage_name, event.track_id, event.track_title or "")
    elif event.kind == "completed":
        logger.info("✓ %s — track %s (%dms)", event.stage_name, event.track_id, event.duration_ms or 0)
    elif event.kind == "failed":
        logger.warning("✗ %s — track %s: %s", event.stage_name, event.track_id, event.error)
    elif event.kind == "skipped":
        logger.debug("⤳ %s — track %s skipped", event.stage_name, event.track_id)
