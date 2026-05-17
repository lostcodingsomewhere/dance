"""Pipeline-ops endpoints.

Read endpoints: ``GET /status`` and ``GET /recent`` for live progress.
Write endpoints: ``POST /ingest/preview`` and ``POST /ingest/commit`` for
queueing yt-dlp downloads from an Exportify CSV; ``GET /jobs`` and
``GET /jobs/{id}`` for download-job status.

Cross-process note: the dispatcher (a ``dance process`` invocation) runs
in a separate shell and the API can't see its EventBus directly. Status
endpoints derive everything from the SQLite DB (``tracks.updated_at``
moves on every state transition). Job endpoints are in-process — the API
itself runs the yt-dlp threads, so they emit events in real time.
"""

from __future__ import annotations

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
from dance.core.database import Track, TrackState
from dance.spotify.csv_importer import (
    CsvRow,
    download_track,
    expected_target,
    parse_csv,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_INGEST_WORKERS = 4


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

    if not body.include_duplicates:
        new, _ = _classify_rows(rows, session, settings.library_dir)
        # Re-derive CsvRow from preview output (keep field parity)
        keep_keys = {(r.artist, r.title) for r in new}
        rows = [r for r in rows if (r.artist, r.title) in keep_keys]

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="Every row was flagged as duplicate. Pass include_duplicates=true to download anyway.",
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
