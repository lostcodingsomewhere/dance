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


@router.get("/status", response_model=PipelineStatusOut)
def get_pipeline_status(session: Session = Depends(get_session)) -> PipelineStatusOut:
    """Track counts per state, plus total and a flag for "anything moving"."""
    rows = (
        session.query(Track.state, func.count(Track.id))
        .group_by(Track.state)
        .all()
    )
    by_state = {state: int(count) for state, count in rows}
    # Ensure every enum value is present in the response, even if zero,
    # so the UI can render a stable grid.
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

    return PipelineStatusOut(
        counts=counts,
        total=total,
        in_progress=in_progress,
        errors=counts[TrackState.ERROR.value],
        complete=counts[TrackState.COMPLETE.value],
    )


@router.get("/recent", response_model=list[PipelineRecentTrackOut])
def get_pipeline_recent(
    limit: int = Query(20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> list[PipelineRecentTrackOut]:
    """Last N tracks ordered by ``updated_at`` desc.

    Useful as a low-rate activity feed during processing: every state
    transition bumps ``updated_at``, so this surfaces "the dispatcher is
    advancing X right now" without needing an event log.
    """
    tracks = (
        session.query(Track)
        .order_by(desc(Track.updated_at), desc(Track.id))
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
    pipe its events into the JobRegistry."""
    registry = get_job_registry()
    registry.set_status(job_id, "running")

    # Each worker thread gets its own SQLAlchemy session (sessions are not
    # threadsafe to share). The dispatcher uses this session to query and
    # commit transitions.
    SessionLocal = get_session_factory(settings.db_url)
    session = SessionLocal()
    try:
        # Late import — keeps API startup snappy when the dispatcher's heavy
        # deps (Demucs / CLAP) aren't needed.
        from dance.pipeline.dispatcher import Dispatcher

        dispatcher = Dispatcher(settings, session)

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
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        # Drop the global lock so the next /process can fire.
        if _PROCESS_LOCK.locked():
            try:
                _PROCESS_LOCK.release()
            except RuntimeError:
                # Thread that acquired it isn't this one — shouldn't happen but
                # be defensive.
                pass


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
