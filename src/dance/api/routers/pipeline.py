"""Pipeline-ops endpoints.

Read endpoints: ``GET /status`` and ``GET /recent`` for live progress.
Write endpoints:

- ``POST /ingest/preview`` and ``POST /ingest/commit`` for queueing
  yt-dlp downloads from an Exportify CSV.
- ``POST /process`` for kicking off the dispatcher (ingest + all stages)
  from the UI. Equivalent to running ``dance process`` at the CLI.
- ``GET /jobs`` and ``GET /jobs/{id}`` for download / pipeline-run job
  status.

Cross-process note: the dispatcher also runs as a CLI subprocess
(``dance process``). The API has no IPC into that shell. To prevent two
dispatchers thrashing the DB, ``POST /process`` rejects with 409 if a
``pipeline_run`` Job is already active in our in-process registry. It
does NOT detect an externally-launched ``dance process`` — if the user
has one running in another terminal, the second run will race. Same
DB-row semantics keep things consistent; it's just wasteful CPU.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from dance.api.dedup import find_duplicate, index_existing
from dance.api.deps import get_session, get_settings
from dance.api.jobs import get_job_registry
from dance.api.schemas import (
    IngestCommitRequest,
    IngestPreviewRequest,
    IngestPreviewResponse,
    IngestPreviewRow,
    JobOut,
    PipelineRecentTrackOut,
    PipelineStatusOut,
)
from dance.config import Settings
from dance.core.database import Track, TrackState, get_session_factory
from dance.pipeline.events import StageEvent
from dance.spotify.csv_importer import (
    CsvRow,
    download_track,
    expected_target,
    parse_csv,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_INGEST_WORKERS = 4

# Module-level lock so two API requests can't race to start ``dance process``.
# (External CLI runs are still possible; see module docstring.)
_PROCESS_LOCK = threading.Lock()

# Watch mode: a background thread polls every _WATCH_INTERVAL_S and, when
# enabled, auto-triggers a pipeline_run if there's pending work and no
# active job. State is persisted to <data_dir>/watch.json across restarts.
_WATCH_STATE = {"enabled": False}
_WATCH_INTERVAL_S = 60
_WATCH_STARTED = False
_WATCH_LOCK = threading.Lock()


def _watch_state_path(settings: Settings) -> Path:
    return settings.data_dir / "watch.json"


def _load_watch_state(settings: Settings) -> None:
    """Best-effort load of persisted watch state. Quietly tolerates a missing
    or corrupt file — defaults to disabled."""
    import json as _json

    path = _watch_state_path(settings)
    if not path.exists():
        return
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("enabled"), bool):
            _WATCH_STATE["enabled"] = data["enabled"]
    except (OSError, ValueError):
        pass


def _save_watch_state(settings: Settings) -> None:
    import json as _json

    path = _watch_state_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(_WATCH_STATE), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not persist watch state to %s: %s", path, e)


# These states mean "this track still has work to do." Used by both the
# /status endpoint's ``in_progress`` flag and the watch loop's
# pending-work check.
_TERMINAL_STATES: frozenset[str] = frozenset(
    {TrackState.COMPLETE.value, TrackState.ERROR.value}
)


def _pending_track_count(settings: Settings) -> int:
    """Count tracks not yet at a terminal state. Best-effort; opens its own
    session because the watch thread doesn't have one injected."""
    SessionLocal = get_session_factory(settings.db_url)
    session = SessionLocal()
    try:
        return int(
            session.query(func.count(Track.id))
            .filter(~Track.state.in_(_TERMINAL_STATES))
            .scalar()
            or 0
        )
    finally:
        session.close()


def _watch_loop(settings: Settings) -> None:
    """Daemon loop: every _WATCH_INTERVAL_S, if enabled + pending work + no
    active pipeline_run job, kick off a pipeline_run. Idempotent and safe
    to leave running forever — does nothing when ``enabled=False``."""
    registry = get_job_registry()
    while True:
        try:
            time.sleep(_WATCH_INTERVAL_S)
            if not _WATCH_STATE["enabled"]:
                continue
            if registry.active(kind="pipeline_run"):
                continue
            if _pending_track_count(settings) == 0:
                continue
            # Replicate the /process endpoint's locking behavior: skip
            # cleanly if some other path beat us to it.
            if not _PROCESS_LOCK.acquire(blocking=False):
                continue
            job = registry.create("pipeline_run", list(_DISPATCH_STAGES))
            t = threading.Thread(
                target=_run_dispatcher_job,
                args=(job.id, settings),
                daemon=True,
                name=f"watch-pipeline-{job.id}",
            )
            t.start()
        except Exception:  # noqa: BLE001
            logger.exception("watch loop iteration failed; continuing")


def start_watch_thread(settings: Settings) -> None:
    """Start the watch thread once per process. Called from ``create_app``."""
    global _WATCH_STARTED
    with _WATCH_LOCK:
        if _WATCH_STARTED:
            return
        _load_watch_state(settings)
        t = threading.Thread(
            target=_watch_loop, args=(settings,), daemon=True, name="watch-loop"
        )
        t.start()
        _WATCH_STARTED = True
        logger.info(
            "watch loop started (enabled=%s, interval=%ds)",
            _WATCH_STATE["enabled"],
            _WATCH_INTERVAL_S,
        )


@router.get("/status", response_model=PipelineStatusOut)
def get_pipeline_status(session: Session = Depends(get_session)) -> PipelineStatusOut:
    """Track counts per state, plus total and weighted progress.

    ``weighted_progress`` (0-100) sums per-track stage progress so the UI
    can show a meaningful percentage *during* a run. ``complete``-state
    tracks alone don't tell the story — a library with everything at
    "separated" is genuinely halfway done even though zero tracks have
    reached the terminal state.
    """
    rows = (
        session.query(Track.state, func.count(Track.id))
        .group_by(Track.state)
        .all()
    )
    by_state = {state: int(count) for state, count in rows}
    counts = {s.value: by_state.get(s.value, 0) for s in TrackState}
    total = sum(counts.values())

    in_progress_states = {
        TrackState.ANALYZING.value,
        TrackState.SEPARATING.value,
        TrackState.ANALYZING_STEMS.value,
        TrackState.DETECTING_REGIONS.value,
        TrackState.EMBEDDING.value,
    }
    in_progress = any(counts[s] > 0 for s in in_progress_states)

    # Stage-weighted progress. Order = the order tracks advance through
    # the pipeline. Each "completed" state advances progress by 1/6.
    # In-flight states count as the previous stage (partial credit
    # would require sub-stage observability we don't have).
    _STAGE_PROGRESS: dict[str, float] = {
        TrackState.PENDING.value: 0,
        TrackState.ANALYZING.value: 0,
        TrackState.ANALYZED.value: 1,
        TrackState.SEPARATING.value: 1,
        TrackState.SEPARATED.value: 2,
        TrackState.ANALYZING_STEMS.value: 2,
        TrackState.STEMS_ANALYZED.value: 3,
        TrackState.DETECTING_REGIONS.value: 3,
        TrackState.REGIONS_DETECTED.value: 4,
        TrackState.EMBEDDING.value: 4,
        TrackState.EMBEDDED.value: 5,
        TrackState.COMPLETE.value: 6,
        # Errored tracks don't contribute progress — they need fixing /
        # re-running before they advance.
        TrackState.ERROR.value: 0,
    }
    weighted_total = sum(counts[s] * _STAGE_PROGRESS[s] for s in counts)
    max_possible = total * 6
    weighted_progress = (
        round((weighted_total / max_possible) * 100, 1) if max_possible else 0.0
    )

    return PipelineStatusOut(
        counts=counts,
        total=total,
        in_progress=in_progress,
        errors=counts[TrackState.ERROR.value],
        complete=counts[TrackState.COMPLETE.value],
        weighted_progress=weighted_progress,
    )


@router.get("/recent", response_model=list[PipelineRecentTrackOut])
def get_pipeline_recent(
    limit: int = Query(20, ge=1, le=1000),
    state: str | None = Query(None, description="Filter by TrackState value"),
    session: Session = Depends(get_session),
) -> list[PipelineRecentTrackOut]:
    """Tracks ordered by ``updated_at`` desc.

    - Unfiltered: surfaces "the dispatcher is advancing X right now" — a
      glanceable live feed across all states.
    - With ``state=<value>``: returns every track currently in that state,
      newest-touched first. Used by the UI when the user clicks a
      state-count tile to drill in.

    ``limit`` caps at 1000 so the UI can offer "view all" without unbounded
    queries — beyond that, the pagination story is "use ``/tracks``".
    """
    q = session.query(Track)
    if state is not None:
        q = q.filter(Track.state == state)
    tracks = (
        q.order_by(desc(Track.updated_at), desc(Track.id))
        .limit(limit)
        .all()
    )
    return [
        PipelineRecentTrackOut(
            id=t.id,
            title=t.title,
            artist=t.artist,
            state=t.state,
            updated_at=t.updated_at,
            error_message=t.error_message,
        )
        for t in tracks
    ]


# ---------------------------------------------------------------------------
# CSV ingest (Exportify → yt-dlp)
# ---------------------------------------------------------------------------


def _classify_rows(
    rows: list[CsvRow], session: Session, library: Path
) -> tuple[list[IngestPreviewRow], list[IngestPreviewRow]]:
    """Split parsed CSV rows into (new, duplicates).

    A row is a "duplicate" if either:
    - The target file path already exists on disk, OR
    - A track in the DB has a fuzzy-matching (artist, title).
    """
    existing_rows = session.query(Track.id, Track.artist, Track.title).all()
    idx = index_existing(
        [(int(tid), a, t) for tid, a, t in existing_rows]
    )

    new: list[IngestPreviewRow] = []
    dupes: list[IngestPreviewRow] = []
    for row in rows:
        target = expected_target(library, row)
        on_disk = target.exists() and target.stat().st_size > 100 * 1024
        dup_id = find_duplicate(row.artist, row.title, idx)
        out_row = IngestPreviewRow(
            artist=row.artist,
            title=row.title,
            album=row.album,
            duration_s=row.duration_s,
            target_path=str(target),
            target_exists=on_disk,
            duplicate_of=dup_id,
        )
        if on_disk or dup_id is not None:
            dupes.append(out_row)
        else:
            new.append(out_row)
    return new, dupes


@router.post("/ingest/preview", response_model=IngestPreviewResponse)
def ingest_preview(
    body: IngestPreviewRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> IngestPreviewResponse:
    """Parse the CSV and classify each row as new vs duplicate.

    No side effects. The UI calls this on paste, shows the user what
    would happen, and only calls ``/ingest/commit`` after approval.
    """
    rows, parse_errors = parse_csv(body.csv_text)
    new, dupes = _classify_rows(rows, session, settings.library_dir)
    return IngestPreviewResponse(
        total_rows=len(rows),
        new_rows=new,
        duplicates=dupes,
        parse_errors=parse_errors,
    )


def _run_csv_job(
    job_id: str,
    csv_rows: list[CsvRow],
    library: Path,
    chrome_profile: str = "Profile 2",
) -> None:
    """Background worker: download each CSV row in a small thread pool."""
    registry = get_job_registry()
    registry.set_status(job_id, "running")
    try:
        library.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=_INGEST_WORKERS) as ex:
            futures = {
                ex.submit(download_track, row, library, chrome_profile): i
                for i, row in enumerate(csv_rows)
            }
            for fut in futures:
                pass  # ensure all submitted before any await — keeps order intact
            for fut, idx in list(futures.items()):
                try:
                    status, msg = fut.result()
                except Exception as e:  # noqa: BLE001
                    status, msg = "fail", str(e)[:200]
                registry.update_item(job_id, idx, status, msg)
        registry.set_status(job_id, "done")
    except Exception as e:  # noqa: BLE001
        registry.set_status(job_id, "error", error=str(e)[:300])


@router.post("/ingest/commit", response_model=JobOut)
def ingest_commit(
    body: IngestCommitRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> JobOut:
    """Kick off a background yt-dlp job for the new (and optionally duplicate) rows.

    Returns a ``Job`` immediately with all items in ``pending`` state. The UI
    subscribes to the ``/ws/pipeline`` WebSocket (or polls ``/jobs/{id}``)
    to watch the items flip to ``ok`` / ``skip`` / ``fail`` as downloads
    complete.

    After this job finishes, the **files exist on disk in
    ``library_dir``** — but ``dance process`` still needs to be run (CLI
    or future endpoint) for the dispatcher to ingest them as Track rows
    and run analysis / Demucs / etc.
    """
    rows, parse_errors = parse_csv(body.csv_text)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail={"error": "No valid rows in CSV", "parse_errors": parse_errors},
        )

    if body.selected_keys is not None:
        # Per-row toggle mode: download exactly the rows the UI chose.
        # ``|`` separator never appears in sanitized artist/title.
        wanted = set(body.selected_keys)
        rows = [r for r in rows if f"{r.artist}|{r.title}" in wanted]
    elif not body.include_duplicates:
        # Bulk mode, skip dupes: re-classify and keep only new rows.
        new, _ = _classify_rows(rows, session, settings.library_dir)
        keep_keys = {(r.artist, r.title) for r in new}
        rows = [r for r in rows if (r.artist, r.title) in keep_keys]
    # else: bulk mode + include_duplicates → keep everything

    if not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "No rows to download. Either every row was a duplicate "
                "(pass include_duplicates=true) or selected_keys matched none."
            ),
        )

    registry = get_job_registry()
    labels = [f"{r.artist} - {r.title}" for r in rows]
    job = registry.create("csv_ingest", labels)

    thread = threading.Thread(
        target=_run_csv_job,
        args=(job.id, rows, settings.library_dir),
        daemon=True,
        name=f"csv-ingest-{job.id}",
    )
    thread.start()

    return JobOut(**job.to_dict())


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(limit: int = Query(20, ge=1, le=100)) -> list[JobOut]:
    registry = get_job_registry()
    return [JobOut(**j.to_dict()) for j in registry.list(limit=limit)]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    registry = get_job_registry()
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(**job.to_dict())


# ---------------------------------------------------------------------------
# Pipeline-run trigger ("Process now" from the UI)
# ---------------------------------------------------------------------------

# Stage names a fresh Dispatcher registers. Used as the JobItem layout for
# pipeline-run jobs so the UI can render a stage-level progress bar before
# any events fire.
_DISPATCH_STAGES = (
    "ingest",
    "analyze",
    "separate",
    "analyze_stems",
    "detect_regions",
    "embed",
)


def _run_dispatcher_job(job_id: str, settings: Settings) -> None:
    """Background worker: build a Dispatcher in this thread, run it,
    pipe its events into the JobRegistry.

    Uses the parallel dispatcher (session_factory) so stages run
    concurrently and stages with ``concurrency>1`` get multiple
    workers. GPU contention is bounded by the dispatcher's module-level
    semaphore (``DANCE_GPU_CONCURRENCY``)."""
    registry = get_job_registry()
    registry.set_status(job_id, "running")

    SessionLocal = get_session_factory(settings.db_url)
    try:
        # Late import — keeps API startup snappy when the dispatcher's heavy
        # deps (Demucs / CLAP) aren't needed.
        from dance.pipeline.dispatcher import Dispatcher

        dispatcher = Dispatcher(settings, session_factory=SessionLocal)

        # Live counters per stage. Updated on every event; flushed to the
        # JobRegistry as a "X/Y done" message.
        stage_index = {name: i for i, name in enumerate(_DISPATCH_STAGES)}
        live: dict[str, dict[str, int]] = {
            name: {"done": 0, "fail": 0} for name in _DISPATCH_STAGES
        }

        def _on_event(event: StageEvent) -> None:
            idx = stage_index.get(event.stage_name)
            if idx is None:
                return
            counters = live[event.stage_name]
            if event.kind == "completed":
                counters["done"] += 1
            elif event.kind == "failed":
                counters["fail"] += 1
            msg = (
                f"{counters['done']} done"
                + (f" · {counters['fail']} failed" if counters["fail"] else "")
            )
            # Items stay "pending" until we finalize at the end; the message
            # is what the UI displays per stage.
            registry.update_item(job_id, idx, "pending", msg)

        dispatcher.events.subscribe(_on_event)

        # Ingest first — surface its counts on item 0.
        ingest_counts = dispatcher.ingest()
        registry.update_item(
            job_id,
            stage_index["ingest"],
            "ok",
            (
                f"new={ingest_counts.get('new', 0)} "
                f"updated={ingest_counts.get('updated', 0)} "
                f"unchanged={ingest_counts.get('unchanged', 0)}"
            ),
        )

        result = dispatcher.run()

        # Finalize: flip each stage item to ok / fail based on totals.
        for name, counts in result.items():
            idx = stage_index.get(name)
            if idx is None:
                continue
            processed = counts.get("processed", 0)
            errors = counts.get("errors", 0)
            skipped = counts.get("skipped", 0)
            status = "fail" if errors and processed == 0 else "ok"
            registry.update_item(
                job_id,
                idx,
                status,
                f"processed={processed} errors={errors} skipped={skipped}",
            )

        registry.set_status(job_id, "done")
    except Exception as e:  # noqa: BLE001
        logger.exception("pipeline_run job %s crashed", job_id)
        registry.set_status(job_id, "error", error=str(e)[:300])
    finally:
        # Drop the global lock so the next /process can fire.
        if _PROCESS_LOCK.locked():
            try:
                _PROCESS_LOCK.release()
            except RuntimeError:
                # Thread that acquired it isn't this one — shouldn't happen but
                # be defensive.
                pass


@router.post("/scan")
def trigger_scan(
    settings: Settings = Depends(get_settings),
) -> dict:
    """Run ingest only — scan ``library_dir`` for new audio files, hash
    them, create Track rows in ``pending``. **Does not** advance any
    stages. Use this to preview "what's new on disk" before deciding to
    commit a full Process run.

    Returns the per-bucket counts directly (no Job tracking — scan is
    synchronous and fast, typically < 1 s for hundreds of files).
    """
    # Import here so the heavy pipeline deps don't load on every API request.
    from dance.pipeline.dispatcher import Dispatcher

    SessionLocal = get_session_factory(settings.db_url)
    session = SessionLocal()
    try:
        dispatcher = Dispatcher(settings, session)
        return dispatcher.ingest()
    finally:
        session.close()


@router.get("/watch")
def get_watch() -> dict:
    """Current watch-mode state. ``interval_seconds`` is the poll cadence."""
    return {
        "enabled": _WATCH_STATE["enabled"],
        "interval_seconds": _WATCH_INTERVAL_S,
    }


@router.post("/watch")
def set_watch(
    body: dict,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Toggle watch mode. ``{enabled: bool}``. The background thread is
    always running; this flag controls whether it actually triggers
    pipeline runs."""
    if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
        raise HTTPException(
            status_code=400, detail="body must be {enabled: bool}"
        )
    _WATCH_STATE["enabled"] = body["enabled"]
    _save_watch_state(settings)
    return {
        "enabled": _WATCH_STATE["enabled"],
        "interval_seconds": _WATCH_INTERVAL_S,
    }


@router.post("/process", response_model=JobOut)
def trigger_process(
    settings: Settings = Depends(get_settings),
) -> JobOut:
    """Kick off ``dance process`` in a background thread.

    Equivalent to the CLI ``dance process``: ingests new files in
    ``library_dir``, then runs every registered stage (analyze, separate,
    analyze_stems, detect_regions, embed) until nothing advances.

    Returns 409 if a pipeline run kicked off through this endpoint is
    still active. (An externally-launched ``dance process`` in another
    shell is not detected; the run will start anyway and they will race.)
    """
    registry = get_job_registry()
    if registry.active(kind="pipeline_run"):
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already in flight. Wait for it to finish, or restart the API to clear it.",
        )

    if not _PROCESS_LOCK.acquire(blocking=False):
        # Defensive: lock without an active job means a previous run crashed
        # without releasing. Clear and re-acquire.
        logger.warning(
            "PROCESS_LOCK was held without an active job; force-acquiring."
        )
        try:
            _PROCESS_LOCK.release()
        except RuntimeError:
            pass
        _PROCESS_LOCK.acquire(blocking=False)

    # Pre-populate items so the UI can render a 6-row progress bar from t=0.
    job = registry.create("pipeline_run", list(_DISPATCH_STAGES))

    thread = threading.Thread(
        target=_run_dispatcher_job,
        args=(job.id, settings),
        daemon=True,
        name=f"pipeline-run-{job.id}",
    )
    thread.start()

    return JobOut(**job.to_dict())
