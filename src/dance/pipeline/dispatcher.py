"""
Stage dispatcher and registry.

Two execution modes:

- **Sequential** (legacy): construct with ``Dispatcher(settings, session)``.
  Each stage processes its queue one track at a time, stages run one after
  another. This is what the existing tests use and what ``dance process``
  used through commit 5e917be. Idempotent + resumable but slow.

- **Parallel** (new): construct with
  ``Dispatcher(settings, session_factory=...)``. Each stage gets its own
  worker thread pool (size from ``stage.concurrency``). Stages run
  concurrently. GPU-bound stages (those declaring ``uses_gpu=True``) share
  a module-level semaphore so MPS isn't oversubscribed. Track claims are
  atomic via an optimistic UPDATE … WHERE state=input_state — two workers
  can never own the same row.

The default Stage protocol fields keep stages working in either mode
without code change. Stages that want parallel-mode acceleration just
declare ``concurrency``, ``uses_gpu``, and ``inflight_state``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Optional

from sqlalchemy.orm import Session, sessionmaker

from dance.config import Settings
from dance.core.database import Track, TrackState
from dance.pipeline._gpu import GPU_SEMAPHORE
from dance.pipeline.events import EventBus, StageEvent
from dance.pipeline.heal_ledger import HealLedger
from dance.pipeline.stage import Stage

logger = logging.getLogger(__name__)


# How many times a track may be silently healed from the SAME inflight state
# before we give up and flag it as ERROR instead of re-queuing. This breaks
# the infinite re-crash loop a hard native crash (SIGBUS / exit 138) would
# otherwise cause — that crash isn't catchable by the worker's
# ``except Exception``, so the track keeps coming back to the same *-ing state.
_MAX_HEALS_BEFORE_ERROR = 2


# Re-export under the legacy private name so existing call sites work.
# All GPU-bound code (dispatcher workers, text-search endpoint, etc.) should
# acquire this before touching MPS.
_GPU_SEMAPHORE = GPU_SEMAPHORE


# How long a worker keeps polling for new work after its stage queue
# goes empty. Long enough that upstream stages can feed it, short enough
# that ``dance process`` exits in reasonable time when truly done.
_IDLE_GRACE_S = 30.0
_POLL_INTERVAL_S = 0.5


class Dispatcher:
    """Discovers and runs registered stages until no track is ready."""

    def __init__(
        self,
        settings: Settings,
        session: Optional[Session] = None,
        session_factory: Optional[sessionmaker] = None,
    ) -> None:
        """Construct a dispatcher.

        Exactly one of ``session`` or ``session_factory`` must be provided.
        Sequential mode (``session``) is for tests and single-threaded
        callers; parallel mode (``session_factory``) is for production
        ``dance process`` runs.
        """
        if (session is None) == (session_factory is None):
            raise ValueError(
                "Pass exactly one of session= or session_factory=. "
                "(session for sequential / legacy, session_factory for parallel.)"
            )

        self.settings = settings
        self.session = session
        self.session_factory = session_factory
        self.events = EventBus()
        self._stages: list[Stage] = []
        # Serializes heal-ledger writes across worker threads (each write is a
        # load-modify-save, so concurrent successes could otherwise clobber).
        self._ledger_lock = threading.Lock()

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
        logger.debug(
            "Registered stage: %s (%s -> %s, concurrency=%d, gpu=%s)",
            stage.name,
            stage.input_state,
            stage.output_state,
            getattr(stage, "concurrency", 1),
            getattr(stage, "uses_gpu", False),
        )

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
        from dance.pipeline.stages.ingest import IngestStage  # noqa: F401

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
        """Scan library_dir and create Track rows for new files."""
        from dance.pipeline.stages.ingest import IngestStage

        ingest = IngestStage(self.settings.library_dir)
        session = self.session if self.session is not None else self.session_factory()
        try:
            result = ingest.scan_and_ingest(session)
        finally:
            if self.session is None:
                session.close()
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
        """Drain every registered stage's queue.

        Args:
            limit: Max tracks per stage per pass (sequential mode only).
            skip: Stage names to skip (auto-advances their state).
            track_id: Restrict to a single track.

        Returns:
            Dict of stage_name -> {processed, errors, skipped}.
        """
        if self.session_factory is not None:
            return self._run_parallel(skip=skip, track_id=track_id)
        return self._run_sequential(limit=limit, skip=skip, track_id=track_id)

    # ------------------------------------------------------------------
    # Sequential mode (legacy — unchanged behavior)
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        limit: int | None,
        skip: Iterable[str] | None,
        track_id: int | None,
    ) -> dict[str, dict[str, int]]:
        assert self.session is not None
        skip_set = {s for s in (skip or []) if s}
        totals: dict[str, dict[str, int]] = {}

        # Outer loop: keep iterating until nothing changed in a full pass.
        max_passes = max(2, 2 * len(self._stages))
        for _ in range(max_passes):
            changed = False
            for stage in self._stages:
                counts = totals.setdefault(
                    stage.name, {"processed": 0, "errors": 0, "skipped": 0}
                )

                if stage.name in skip_set:
                    n = self._auto_advance_seq(stage, track_id=track_id, limit=limit)
                    counts["skipped"] += n
                    if n:
                        changed = True
                    continue

                processed, errors = self._run_stage_seq(
                    stage, track_id=track_id, limit=limit
                )
                counts["processed"] += processed
                counts["errors"] += errors
                if processed or errors:
                    changed = True

            if not changed:
                break

        return totals

    def _pending_query_seq(
        self, stage: Stage, track_id: int | None, limit: int | None
    ):
        q = self.session.query(Track).filter(Track.state == stage.input_state.value)
        if track_id is not None:
            q = q.filter(Track.id == track_id)
        if limit is not None:
            q = q.limit(limit)
        return q

    def _run_stage_seq(
        self, stage: Stage, track_id: int | None, limit: int | None
    ) -> tuple[int, int]:
        tracks = self._pending_query_seq(stage, track_id, limit).all()
        processed = 0
        errors = 0

        for track in tracks:
            t0 = time.monotonic()
            self.events.emit("started", stage.name, track)
            try:
                ok = stage.process(self.session, track, self.settings)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Stage %s crashed on track %s", stage.name, track.id)
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

    def _auto_advance_seq(
        self, stage: Stage, track_id: int | None, limit: int | None
    ) -> int:
        tracks = self._pending_query_seq(stage, track_id, limit).all()
        for track in tracks:
            track.state = stage.output_state.value
        if tracks:
            self.session.commit()
        return len(tracks)

    # ------------------------------------------------------------------
    # Parallel mode
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        skip: Iterable[str] | None,
        track_id: int | None,
    ) -> dict[str, dict[str, int]]:
        assert self.session_factory is not None
        skip_set = {s for s in (skip or []) if s}

        # First: heal any tracks left stuck in *-ing states by a previous
        # crashed / killed run. No stage uses *-ing as its input_state, so
        # without this they'd sit there forever.
        heal_session = self.session_factory()
        try:
            healed = self._heal_inflight_orphans(heal_session)
        finally:
            heal_session.close()
        if healed:
            logger.info("Healed %d inflight orphans: %s", sum(healed.values()), healed)
            errored = {k: v for k, v in healed.items() if k.endswith(":errored")}
            if errored:
                logger.warning(
                    "Flagged %d track(s) ERROR after repeated heals (likely hard "
                    "native crashes): %s — see `dance status`",
                    sum(errored.values()),
                    errored,
                )

        # Auto-advance skipped stages once, up front, in a single session.
        if skip_set:
            session = self.session_factory()
            try:
                for stage in self._stages:
                    if stage.name in skip_set:
                        self._auto_advance_par(stage, session, track_id=track_id)
            finally:
                session.close()

        # Aggregator: per-stage counters updated under lock by workers.
        totals: dict[str, dict[str, int]] = {
            s.name: {"processed": 0, "errors": 0, "skipped": 0}
            for s in self._stages
        }
        totals_lock = threading.Lock()

        # Spin up a pool per stage. Each pool gets stage.concurrency workers.
        stop_event = threading.Event()
        futures = []
        pools: list[ThreadPoolExecutor] = []
        try:
            for stage in self._stages:
                if stage.name in skip_set:
                    continue
                if getattr(stage, "inflight_state", None) is None:
                    raise ValueError(
                        f"Stage {stage.name!r} declares no inflight_state — "
                        "required in parallel mode. Add inflight_state to the "
                        "stage class or use sequential mode."
                    )
                n = max(1, int(getattr(stage, "concurrency", 1)))
                pool = ThreadPoolExecutor(
                    max_workers=n, thread_name_prefix=f"dance-{stage.name}"
                )
                pools.append(pool)
                for _ in range(n):
                    futures.append(
                        pool.submit(
                            self._worker_loop,
                            stage,
                            totals,
                            totals_lock,
                            stop_event,
                            track_id,
                        )
                    )
            # Block until every worker exits (each exits after _IDLE_GRACE_S
            # of empty queue, so the dispatcher returns when the pipeline is
            # truly drained).
            wait(futures)
        finally:
            stop_event.set()
            for pool in pools:
                pool.shutdown(wait=False)

        return totals

    def _worker_loop(
        self,
        stage: Stage,
        totals: dict[str, dict[str, int]],
        totals_lock: threading.Lock,
        stop_event: threading.Event,
        track_id: int | None,
    ) -> None:
        """One worker. Claims tracks, processes them, repeats until idle."""
        assert self.session_factory is not None
        idle_streak = 0.0
        while not stop_event.is_set():
            session = self.session_factory()
            try:
                track = self._claim_track(stage, session, track_id=track_id)
            except Exception:  # noqa: BLE001
                logger.exception("Claim failed in worker %s", stage.name)
                session.close()
                time.sleep(_POLL_INTERVAL_S)
                continue

            if track is None:
                session.close()
                idle_streak += _POLL_INTERVAL_S
                if idle_streak >= _IDLE_GRACE_S:
                    return
                time.sleep(_POLL_INTERVAL_S)
                continue

            idle_streak = 0.0
            self.events.emit("started", stage.name, track)
            t0 = time.monotonic()
            try:
                if getattr(stage, "uses_gpu", False):
                    with _GPU_SEMAPHORE:
                        ok = stage.process(session, track, self.settings)
                else:
                    ok = stage.process(session, track, self.settings)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Stage %s crashed on track %s", stage.name, track.id
                )
                session.rollback()
                fresh = session.get(Track, track.id)
                if fresh is not None:
                    fresh.state = stage.error_state.value
                    fresh.error_message = f"{stage.name}: {exc}"[:500]
                    session.commit()
                self.events.emit(
                    "failed", stage.name, track, error=str(exc)
                )
                with totals_lock:
                    totals[stage.name]["errors"] += 1
                session.close()
                continue

            dt_ms = int((time.monotonic() - t0) * 1000)
            if ok:
                # Clean advance — forget any prior heal history for this track
                # so a fresh crash later starts its counter from zero.
                with self._ledger_lock:
                    HealLedger(self.settings.data_dir).clear(int(track.id))
                self.events.emit(
                    "completed", stage.name, track, duration_ms=dt_ms
                )
                with totals_lock:
                    totals[stage.name]["processed"] += 1
            else:
                self.events.emit(
                    "failed",
                    stage.name,
                    track,
                    duration_ms=dt_ms,
                    error=track.error_message or "stage returned False",
                )
                with totals_lock:
                    totals[stage.name]["errors"] += 1
            session.close()

    def _claim_track(
        self,
        stage: Stage,
        session: Session,
        track_id: int | None = None,
    ) -> Track | None:
        """Atomically pick a track in ``input_state`` and flip it to
        ``inflight_state``. Returns the (now-claimed) Track row, or None
        if no candidate was claimable.

        Implementation: SELECT a candidate id, then UPDATE ... WHERE id=?
        AND state=input_state. If rowcount==0, another worker won the
        race; we try again on the next loop iteration. SQLite serializes
        UPDATEs so two workers can't both increment rowcount on the same
        row.
        """
        q = session.query(Track.id).filter(
            Track.state == stage.input_state.value
        )
        if track_id is not None:
            q = q.filter(Track.id == track_id)
        candidate = q.order_by(Track.id).first()
        if candidate is None:
            return None

        cid = int(candidate[0])
        inflight = getattr(stage, "inflight_state", None)
        assert inflight is not None, (
            "_claim_track called on a stage without inflight_state — "
            "_run_parallel() should have caught this at startup"
        )
        rowcount = (
            session.query(Track)
            .filter(
                Track.id == cid, Track.state == stage.input_state.value
            )
            .update(
                {Track.state: inflight.value},
                synchronize_session=False,
            )
        )
        session.commit()

        if rowcount == 0:
            # Lost the race; the next iteration will find the next candidate.
            return None
        return session.get(Track, cid)

    def _heal_inflight_orphans(self, session: Session) -> dict[str, int]:
        """Reset tracks stuck in a stage's ``inflight_state`` back to the
        stage's ``input_state`` so a fresh worker can claim it — UNLESS the
        track has already been healed from that same state too many times, in
        which case it's promoted to ERROR.

        Tracks land in ``*-ing`` states when a worker claims them but then
        crashes / is killed / loses its process before flipping to
        ``output_state``. A *graceful* crash inside ``process()`` is caught by
        the worker's ``except Exception`` and marked ERROR directly — those
        never reach here. What reaches here is a *hard* crash: a native SIGBUS
        / segfault / exit 138 (common for Demucs on MPS) that takes the whole
        process down before any Python handler runs. On the next ``dance
        process`` startup we'd reset the track and null its error_message,
        and it would crash exactly the same way again — an infinite silent
        loop with nothing in ``dance status``.

        To break that loop we keep a per-track heal counter in a sidecar JSON
        (:class:`HealLedger`, no schema change). After
        ``_MAX_HEALS_BEFORE_ERROR`` heals from the *same* inflight state we
        flag the track ERROR with an actionable message instead of re-queuing.

        Stages without an ``inflight_state`` declaration are skipped (they
        can't have produced orphans in the parallel claim model). Idempotent:
        a clean DB returns an empty dict and writes nothing.

        Returns: dict of ``stage_name`` -> number of tracks reset back to
        input_state, plus a synthetic ``"<stage>:errored"`` key counting any
        promoted to ERROR. Empty dict if there was nothing to heal.
        """
        ledger = HealLedger(self.settings.data_dir)
        healed: dict[str, int] = {}
        dirty = False
        for stage in self._stages:
            inflight = getattr(stage, "inflight_state", None)
            if inflight is None:
                continue
            stuck_ids = [
                int(row[0])
                for row in session.query(Track.id)
                .filter(Track.state == inflight.value)
                .all()
            ]
            requeued = 0
            errored = 0
            for tid in stuck_ids:
                count = ledger.record_heal(tid, inflight.value)
                update = session.query(Track).filter(Track.id == tid)
                if count >= _MAX_HEALS_BEFORE_ERROR:
                    msg = (
                        f"repeated native crash (likely SIGBUS/exit 138) during "
                        f"{stage.name} on this machine — healed {count}× with no "
                        f"progress; see docs/troubleshooting.md "
                        f"(try DANCE_DEMUCS_DEVICE=cpu)"
                    )[:500]
                    update.update(
                        {
                            Track.state: stage.error_state.value,
                            Track.error_message: msg,
                        },
                        synchronize_session=False,
                    )
                    ledger.clear(tid)
                    errored += 1
                    logger.error(
                        "Track %s flagged ERROR: healed %d× from %s without "
                        "progress (likely hard native crash)",
                        tid,
                        count,
                        inflight.value,
                    )
                else:
                    update.update(
                        {
                            Track.state: stage.input_state.value,
                            Track.error_message: None,
                        },
                        synchronize_session=False,
                    )
                    requeued += 1
                dirty = True
            if requeued:
                healed[stage.name] = requeued
            if errored:
                healed[f"{stage.name}:errored"] = errored
        if dirty:
            session.commit()
        return healed

    def _auto_advance_par(
        self, stage: Stage, session: Session, track_id: int | None
    ) -> None:
        q = session.query(Track).filter(Track.state == stage.input_state.value)
        if track_id is not None:
            q = q.filter(Track.id == track_id)
        tracks = q.all()
        for t in tracks:
            t.state = stage.output_state.value
        if tracks:
            session.commit()


# ---------------------------------------------------------------------------
# Default event logger
# ---------------------------------------------------------------------------


def _default_logger(event: StageEvent) -> None:
    if event.kind == "started":
        logger.info(
            "▶ %s — track %s (%s)",
            event.stage_name,
            event.track_id,
            event.track_title or "",
        )
    elif event.kind == "completed":
        logger.info(
            "✓ %s — track %s (%dms)",
            event.stage_name,
            event.track_id,
            event.duration_ms or 0,
        )
    elif event.kind == "failed":
        logger.warning(
            "✗ %s — track %s: %s",
            event.stage_name,
            event.track_id,
            event.error,
        )
    elif event.kind == "skipped":
        logger.debug("⤳ %s — track %s skipped", event.stage_name, event.track_id)
