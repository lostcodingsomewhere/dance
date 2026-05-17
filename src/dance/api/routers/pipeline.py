"""Pipeline-ops endpoints.

Read-only visibility into the in-flight pipeline. Mirrors `dance status`
at the CLI: track counts per state, plus a "recently changed" feed so the
UI can show what's actually moving without polling individual track rows.

Intentionally minimal:
- No write endpoints. (Restarting/cancelling the pipeline still goes
  through the CLI; the API can't see a running ``dance process`` from
  another shell.)
- No event log. The "recent" list is derived from ``tracks.updated_at``
  since the dispatcher's EventBus isn't currently persisted. Good enough
  for a glanceable feed during processing.
- No dedupe / URL-ingest. Those are tracked in
  `docs/proposals/pipeline-ops-ui.md` (future work).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from dance.api.deps import get_session
from dance.api.schemas import PipelineRecentTrackOut, PipelineStatusOut
from dance.core.database import Track, TrackState

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


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
