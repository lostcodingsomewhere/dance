"""Set endpoints — persistent curated track plans + tail-rec scoring.

A ``Set`` is the *intent* (what the DJ planned for a gig); ``DjSession`` is the
*history* (what actually played). The two stay decoupled: a set is reused
across sessions, a session may or may not follow a set.

Tail-rec scoring lives in :mod:`dance.recommender.tail` and is surfaced at
``GET /sets/{id}/tail-recs`` — the Set Rail's "what comes next" section.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from dance.api.deps import get_session
from dance.api.schemas import (
    SetCreateRequest,
    SetOut,
    SetSummaryOut,
    SetTrackAddRequest,
    SetTrackUpdateRequest,
    SetUpdateRequest,
    TailRecsResponse,
)
from dance.core.database import (
    AudioAnalysis,
    Set,
    SetTrack,
    StemKind,
    Track,
    now_utc,
)
from dance.recommender.tail import DEFAULT_WINDOW, tail_recs_for_set

router = APIRouter(prefix="/sets", tags=["sets"])

_VALID_STEM_KINDS = {k.value for k in StemKind}


def _validate_stem_kinds(value: list[str] | None) -> str | None:
    """Normalize + validate a stem_kinds input. Returns the JSON-encoded
    string to store, or None for the all-stems default. Raises 400 for
    invalid kinds or an empty list (use null to clear)."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) == 0:
        raise HTTPException(
            status_code=400,
            detail="stem_kinds must be a non-empty list, or null to clear",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise HTTPException(
                status_code=400,
                detail=f"stem_kinds entries must be strings, got {type(raw).__name__}",
            )
        kind = raw.lower()
        if kind not in _VALID_STEM_KINDS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown stem kind {raw!r}; valid: {sorted(_VALID_STEM_KINDS)}",
            )
        if kind not in seen:
            seen.add(kind)
            normalized.append(kind)
    return json.dumps(normalized)


def _decode_stem_kinds(stored: str | None) -> list[str] | None:
    if stored is None:
        return None
    try:
        parsed = json.loads(stored)
        if isinstance(parsed, list) and parsed:
            return [str(x) for x in parsed]
    except (ValueError, TypeError):
        pass
    return None


def _set_to_out(session: Session, s: Set) -> dict:
    rows = session.query(SetTrack).filter(SetTrack.set_id == s.id).order_by(SetTrack.position).all()
    track_ids = [int(r.track_id) for r in rows]
    tracks_by_id: dict[int, Track] = (
        {int(t.id): t for t in session.query(Track).filter(Track.id.in_(track_ids))}
        if track_ids
        else {}
    )
    analyses_by_id: dict[int, AudioAnalysis] = (
        {
            int(a.track_id): a
            for a in session.query(AudioAnalysis).filter(
                AudioAnalysis.track_id.in_(track_ids),
                AudioAnalysis.stem_file_id.is_(None),
            )
        }
        if track_ids
        else {}
    )
    tracks_out: list[dict] = []
    for row in rows:
        track = tracks_by_id.get(int(row.track_id))
        a = analyses_by_id.get(int(row.track_id))
        tracks_out.append(
            {
                "track_id": int(row.track_id),
                "position": int(row.position),
                "note": row.note,
                "stem_kinds": _decode_stem_kinds(row.stem_kinds),
                "added_at": row.added_at,
                "title": track.title if track else None,
                "artist": track.artist if track else None,
                "bpm": float(a.bpm) if a and a.bpm is not None else None,
                "key_camelot": a.key_camelot if a else None,
                "floor_energy": int(a.floor_energy) if a and a.floor_energy is not None else None,
                "duration_seconds": (
                    float(track.duration_seconds)
                    if track and track.duration_seconds is not None
                    else None
                ),
                "track_state": str(track.state) if track else None,
                "track_error": track.error_message if track else None,
            }
        )
    return {
        "id": int(s.id),
        "name": s.name,
        "notes": s.notes,
        "is_active": bool(s.is_active),
        "created_at": s.created_at,
        "updated_at": s.updated_at,
        "tracks": tracks_out,
    }


def _require_set(session: Session, set_id: int) -> Set:
    s = session.get(Set, set_id)
    if s is None:
        raise HTTPException(status_code=404, detail="set not found")
    return s


def _deactivate_others(session: Session, keep_id: int | None) -> None:
    """Flip all other sets to inactive. Caller commits."""
    q = session.query(Set).filter(Set.is_active.is_(True))
    if keep_id is not None:
        q = q.filter(Set.id != keep_id)
    for other in q.all():
        other.is_active = False


@router.get("", response_model=list[SetSummaryOut])
def list_sets(session: Session = Depends(get_session)) -> list[dict]:
    sets = session.query(Set).order_by(Set.updated_at.desc()).all()
    counts = dict(session.query(SetTrack.set_id, _count_func()).group_by(SetTrack.set_id).all())
    return [
        {
            "id": int(s.id),
            "name": s.name,
            "notes": s.notes,
            "is_active": bool(s.is_active),
            "track_count": int(counts.get(s.id, 0)),
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in sets
    ]


@router.get("/active", response_model=SetOut)
def get_active_set(session: Session = Depends(get_session)) -> dict:
    s = session.query(Set).filter(Set.is_active.is_(True)).first()
    if s is None:
        raise HTTPException(status_code=404, detail="no active set")
    return _set_to_out(session, s)


@router.post("", response_model=SetOut)
def create_set(body: SetCreateRequest, session: Session = Depends(get_session)) -> dict:
    s = Set(name=body.name, notes=body.notes, is_active=False)
    session.add(s)
    session.commit()
    session.refresh(s)
    return _set_to_out(session, s)


@router.get("/{set_id}", response_model=SetOut)
def get_set(set_id: int, session: Session = Depends(get_session)) -> dict:
    s = _require_set(session, set_id)
    return _set_to_out(session, s)


@router.patch("/{set_id}", response_model=SetOut)
def update_set(
    set_id: int,
    body: SetUpdateRequest,
    session: Session = Depends(get_session),
) -> dict:
    s = _require_set(session, set_id)
    if body.name is not None:
        s.name = body.name
    if body.notes is not None:
        s.notes = body.notes
    s.updated_at = now_utc()
    session.commit()
    session.refresh(s)
    return _set_to_out(session, s)


@router.delete("/{set_id}")
def delete_set(set_id: int, session: Session = Depends(get_session)) -> Response:
    s = _require_set(session, set_id)
    session.delete(s)
    session.commit()
    return Response(status_code=204)


@router.post("/{set_id}/activate", response_model=SetOut)
def activate_set(set_id: int, session: Session = Depends(get_session)) -> dict:
    """Mark this set as the (only) active set.

    Two-step toggle: clear ``is_active`` on the prior active row first, then
    set it on the target. Done in one transaction so the partial unique
    index never sees two active rows.
    """
    s = _require_set(session, set_id)
    _deactivate_others(session, keep_id=s.id)
    s.is_active = True
    s.updated_at = now_utc()
    session.commit()
    session.refresh(s)
    return _set_to_out(session, s)


@router.post("/{set_id}/tracks", response_model=SetOut)
def add_track(
    set_id: int,
    body: SetTrackAddRequest,
    session: Session = Depends(get_session),
) -> dict:
    s = _require_set(session, set_id)
    if session.get(Track, body.track_id) is None:
        raise HTTPException(status_code=404, detail="track not found")

    existing = (
        session.query(SetTrack).filter(SetTrack.set_id == set_id).order_by(SetTrack.position).all()
    )
    max_pos = existing[-1].position if existing else -1
    target_pos = body.position if body.position is not None else max_pos + 1
    if target_pos < 0 or target_pos > max_pos + 1:
        raise HTTPException(
            status_code=400,
            detail=f"position must be in [0, {max_pos + 1}]",
        )

    # Insert at target_pos: shift everything at or after it up by one.
    # Do the shift in descending order to avoid hitting the unique constraint
    # mid-update.
    if target_pos <= max_pos:
        for row in reversed(existing):
            if row.position >= target_pos:
                row.position += 1
        session.flush()

    row = SetTrack(
        set_id=set_id,
        track_id=body.track_id,
        position=target_pos,
        note=body.note,
        stem_kinds=_validate_stem_kinds(body.stem_kinds),
    )
    session.add(row)
    s.updated_at = now_utc()
    session.commit()
    session.refresh(s)
    return _set_to_out(session, s)


@router.patch("/{set_id}/tracks/{track_id}", response_model=SetOut)
def update_track(
    set_id: int,
    track_id: int,
    body: SetTrackUpdateRequest,
    session: Session = Depends(get_session),
) -> dict:
    s = _require_set(session, set_id)
    row = (
        session.query(SetTrack)
        .filter(SetTrack.set_id == set_id, SetTrack.track_id == track_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="track not in set")

    if body.note is not None:
        row.note = body.note

    # stem_kinds uses model_fields_set so we distinguish "field omitted"
    # (leave alone) from "explicit null" (clear the filter).
    if "stem_kinds" in body.model_fields_set:
        if body.stem_kinds is None:
            row.stem_kinds = None
        else:
            row.stem_kinds = _validate_stem_kinds(body.stem_kinds)

    if body.position is not None and body.position != row.position:
        siblings = (
            session.query(SetTrack)
            .filter(SetTrack.set_id == set_id)
            .order_by(SetTrack.position)
            .all()
        )
        max_pos = len(siblings) - 1
        if body.position < 0 or body.position > max_pos:
            raise HTTPException(
                status_code=400,
                detail=f"position must be in [0, {max_pos}]",
            )
        _renumber_for_move(session, siblings, row, body.position)

    s.updated_at = now_utc()
    session.commit()
    session.refresh(s)
    return _set_to_out(session, s)


@router.delete("/{set_id}/tracks/{track_id}", response_model=SetOut)
def remove_track(
    set_id: int,
    track_id: int,
    session: Session = Depends(get_session),
) -> dict:
    s = _require_set(session, set_id)
    row = (
        session.query(SetTrack)
        .filter(SetTrack.set_id == set_id, SetTrack.track_id == track_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="track not in set")
    removed_pos = row.position
    session.delete(row)
    session.flush()
    # Compact positions above the gap so they stay 0..N-1 contiguous.
    above = (
        session.query(SetTrack)
        .filter(SetTrack.set_id == set_id, SetTrack.position > removed_pos)
        .order_by(SetTrack.position)
        .all()
    )
    for r in above:
        r.position -= 1
    s.updated_at = now_utc()
    session.commit()
    session.refresh(s)
    return _set_to_out(session, s)


@router.get("/{set_id}/tail-recs", response_model=TailRecsResponse)
def get_tail_recs(
    set_id: int,
    k: int = Query(10, ge=1, le=50),
    window: int = Query(DEFAULT_WINDOW, ge=1, le=20),
    exclude_session_plays: bool = Query(
        False,
        description="If true, also exclude tracks played in the current open "
        "DjSession (no double-recommending what's already been on tonight).",
    ),
    session: Session = Depends(get_session),
) -> dict:
    """Arc-fit candidates to append to this Set.

    Scoring uses the trailing ``window`` tracks of the set as the "current
    arc" (default 5) and ranks every other track in the library by
    embedding/key/BPM/energy fit. See :mod:`dance.recommender.tail` for the
    full scoring breakdown."""
    s = _require_set(session, set_id)

    exclude_ids: list[int] = []
    if exclude_session_plays:
        from dance.core.database import DjSession, SessionPlay

        current = (
            session.query(DjSession)
            .filter(DjSession.ended_at.is_(None))
            .order_by(DjSession.started_at.desc())
            .first()
        )
        if current is not None:
            exclude_ids = [
                int(p.track_id)
                for p in session.query(SessionPlay).filter(SessionPlay.session_id == current.id)
            ]

    results = tail_recs_for_set(
        session,
        set_id=int(s.id),
        k=k,
        window=window,
        exclude_track_ids=exclude_ids,
    )

    # Decorate with title/artist/key/BPM/energy for the rail card render.
    track_ids = [r.track_id for r in results]
    tracks: dict[int, Track] = (
        {int(t.id): t for t in session.query(Track).filter(Track.id.in_(track_ids))}
        if track_ids
        else {}
    )
    analyses: dict[int, AudioAnalysis] = (
        {
            int(a.track_id): a
            for a in session.query(AudioAnalysis).filter(
                AudioAnalysis.track_id.in_(track_ids),
                AudioAnalysis.stem_file_id.is_(None),
            )
        }
        if track_ids
        else {}
    )

    recs_out: list[dict] = []
    for r in results:
        t = tracks.get(r.track_id)
        a = analyses.get(r.track_id)
        recs_out.append(
            {
                "track_id": r.track_id,
                "track_title": t.title if t else None,
                "track_artist": t.artist if t else None,
                "bpm": float(a.bpm) if a and a.bpm is not None else None,
                "key_camelot": a.key_camelot if a else None,
                "floor_energy": int(a.floor_energy) if a and a.floor_energy is not None else None,
                "score": r.score,
                "score_breakdown": r.score_breakdown,
                "reasons": r.reasons,
            }
        )

    set_track_count = session.query(SetTrack).filter(SetTrack.set_id == set_id).count()
    return {
        "set_id": int(s.id),
        "set_track_count": int(set_track_count),
        "recs": recs_out,
    }


def _renumber_for_move(
    session: Session,
    siblings: list[SetTrack],
    moved: SetTrack,
    target: int,
) -> None:
    """Move ``moved`` to ``target`` within ``siblings`` and renumber 0..N-1.

    ``siblings`` is the current ordered list (including ``moved``). The
    unique ``(set_id, position)`` index can't tolerate two rows sharing a
    position mid-flush, so we first park every row at a negative sentinel,
    then assign the final positions. Bulletproof against any shift direction.
    """
    others = [r for r in siblings if r.id != moved.id]
    new_order = others[:target] + [moved] + others[target:]

    for i, r in enumerate(new_order):
        r.position = -(i + 1)
    session.flush()

    for i, r in enumerate(new_order):
        r.position = i
    session.flush()


def _count_func():
    """Imported lazily so the module stays importable without sqlalchemy.func
    in any odd-order import paths."""
    from sqlalchemy import func

    return func.count(SetTrack.id)
