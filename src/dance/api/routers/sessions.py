"""DJ session endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dance.api.deps import fullmix_analysis, get_session
from dance.api.schemas import (
    SessionCreateRequest,
    SessionOut,
    SessionPlayCreateRequest,
)
from dance.core.database import DjSession, SessionPlay, Track, now_utc

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _session_to_out(session: Session, dj_session: DjSession) -> dict:
    plays_out: list[dict] = []
    plays = (
        session.query(SessionPlay)
        .filter(SessionPlay.session_id == dj_session.id)
        .order_by(SessionPlay.position_in_set)
        .all()
    )
    for p in plays:
        track = session.get(Track, p.track_id)
        plays_out.append(
            {
                "track_id": int(p.track_id),
                "played_at": p.played_at,
                "position_in_set": int(p.position_in_set),
                "energy_at_play": p.energy_at_play,
                "transition_type": p.transition_type,
                "title": track.title if track else None,
                "artist": track.artist if track else None,
            }
        )
    return {
        "id": int(dj_session.id),
        "name": dj_session.name,
        "notes": dj_session.notes,
        "started_at": dj_session.started_at,
        "ended_at": dj_session.ended_at,
        "plays": plays_out,
    }


@router.post("", response_model=SessionOut)
def create_session(
    body: SessionCreateRequest,
    session: Session = Depends(get_session),
) -> dict:
    dj = DjSession(name=body.name, notes=body.notes, started_at=now_utc())
    session.add(dj)
    session.commit()
    session.refresh(dj)
    return _session_to_out(session, dj)


@router.get("/current", response_model=SessionOut)
def get_current_session(session: Session = Depends(get_session)) -> dict:
    dj = (
        session.query(DjSession)
        .filter(DjSession.ended_at.is_(None))
        .order_by(DjSession.started_at.desc())
        .first()
    )
    if dj is None:
        raise HTTPException(status_code=404, detail="no active session")
    return _session_to_out(session, dj)


@router.get("/{session_id}", response_model=SessionOut)
def get_session_by_id(
    session_id: int,
    session: Session = Depends(get_session),
) -> dict:
    dj = session.get(DjSession, session_id)
    if dj is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _session_to_out(session, dj)


@router.post("/{session_id}/plays", response_model=SessionOut)
def add_play(
    session_id: int,
    body: SessionPlayCreateRequest,
    session: Session = Depends(get_session),
) -> dict:
    dj = session.get(DjSession, session_id)
    if dj is None:
        raise HTTPException(status_code=404, detail="session not found")
    if session.get(Track, body.track_id) is None:
        raise HTTPException(status_code=404, detail="track not found")

    max_pos = (
        session.query(SessionPlay.position_in_set)
        .filter(SessionPlay.session_id == session_id)
        .order_by(SessionPlay.position_in_set.desc())
        .first()
    )
    next_pos = (max_pos[0] + 1) if max_pos else 1

    analysis = fullmix_analysis(session, body.track_id)
    play = SessionPlay(
        session_id=session_id,
        track_id=body.track_id,
        played_at=now_utc(),
        position_in_set=next_pos,
        energy_at_play=analysis.floor_energy if analysis else None,
        transition_type=body.transition_type,
        duration_played_ms=body.duration_played_ms,
    )
    session.add(play)
    session.commit()
    session.refresh(dj)
    return _session_to_out(session, dj)


@router.post("/{session_id}/end", response_model=SessionOut)
def end_session(
    session_id: int,
    session: Session = Depends(get_session),
) -> dict:
    dj = session.get(DjSession, session_id)
    if dj is None:
        raise HTTPException(status_code=404, detail="session not found")
    dj.ended_at = now_utc()
    session.commit()
    session.refresh(dj)
    return _session_to_out(session, dj)
