"""Recommendation endpoints — thin wrapper around the Recommender."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from dance.api.deps import fullmix_analysis, get_session
from dance.api.schemas import RecommendationOut, RecommendRequest
from dance.core.database import EdgeKind, Track
from dance.recommender.recommender import Recommender

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _parse_kinds(values: list[str] | None) -> list[EdgeKind] | None:
    if values is None:
        return None
    try:
        return [EdgeKind(v) for v in values]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid edge kind: {exc}") from exc


def _parse_weights(values: dict[str, float] | None) -> dict[EdgeKind, float] | None:
    if values is None:
        return None
    try:
        return {EdgeKind(k): float(v) for k, v in values.items()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid edge kind: {exc}") from exc


def _run_recommend(
    session: Session,
    *,
    seeds: list[int],
    k: int,
    kinds: list[str] | None,
    weights: dict[str, float] | None,
    exclude: list[int] | None,
) -> list[dict]:
    rec = Recommender(session)
    results = rec.recommend(
        seeds=seeds,
        k=k,
        kinds=_parse_kinds(kinds),
        weights=_parse_weights(weights),
        exclude=exclude,
    )
    out: list[dict] = []
    for r in results:
        track = session.get(Track, r.track_id)
        analysis = fullmix_analysis(session, r.track_id)
        out.append(
            {
                "track_id": r.track_id,
                "score": r.score,
                "reasons": r.reasons,
                "title": track.title if track else None,
                "artist": track.artist if track else None,
                "bpm": analysis.bpm if analysis else None,
                "key_camelot": analysis.key_camelot if analysis else None,
                "floor_energy": analysis.floor_energy if analysis else None,
            }
        )
    return out


@router.post("", response_model=list[RecommendationOut])
def post_recommend(
    body: RecommendRequest,
    session: Session = Depends(get_session),
) -> list[dict]:
    return _run_recommend(
        session,
        seeds=body.seeds,
        k=body.k,
        kinds=body.kinds,
        weights=body.weights,
        exclude=body.exclude,
    )


@router.get("/by-seed/{track_id}", response_model=list[RecommendationOut])
def recommend_by_seed(
    track_id: int,
    session: Session = Depends(get_session),
    k: int = Query(10, ge=1, le=200),
) -> list[dict]:
    return _run_recommend(
        session,
        seeds=[track_id],
        k=k,
        kinds=None,
        weights=None,
        exclude=None,
    )
