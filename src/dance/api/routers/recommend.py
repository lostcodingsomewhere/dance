"""Recommendation endpoints — thin wrapper around the Recommender."""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from dance.api.deps import fullmix_analysis, get_session, get_settings
from dance.api.schemas import (
    RecommendationOut,
    RecommendRequest,
    TextRecommendRequest,
)
from dance.config import Settings
from dance.core.database import EdgeKind, Track
from dance.recommender.recommender import Recommender

logger = logging.getLogger(__name__)
_clap_lock = threading.Lock()

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
                "file_path": track.file_path if track else None,
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


def _get_text_encoder(request: Request, settings: Settings):
    """Lazy-load the EmbeddingStage on first text query; reuse it after."""
    stage = request.app.state.embedding_stage
    if stage is None:
        with _clap_lock:
            stage = request.app.state.embedding_stage
            if stage is None:
                from dance.pipeline.stages.embed import EmbeddingStage

                stage = EmbeddingStage()
                try:
                    stage._ensure_model(settings)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("CLAP load failed")
                    raise HTTPException(
                        status_code=503, detail=f"CLAP model unavailable: {exc}"
                    ) from exc
                request.app.state.embedding_stage = stage
    return stage.encode_text


@router.post("/text", response_model=list[RecommendationOut])
def recommend_by_text(
    body: TextRecommendRequest,
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    """Rank tracks by CLAP cosine similarity to a free-text query.

    Examples: "punchy techy with vocals", "deep rolling bassline",
    "afro-house drums", "ambient pad intro".
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    encoder = _get_text_encoder(request, settings)
    rec = Recommender(session)
    results = rec.recommend_by_text(
        query=body.query,
        text_encoder=encoder,
        k=body.k,
        model_name=settings.clap_model,
        exclude=body.exclude,
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
                "file_path": track.file_path if track else None,
                "bpm": analysis.bpm if analysis else None,
                "key_camelot": analysis.key_camelot if analysis else None,
                "floor_energy": analysis.floor_energy if analysis else None,
            }
        )
    return out
