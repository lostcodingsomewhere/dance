"""Recommendation endpoints — thin wrapper around the Recommender."""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from dance.api.deps import fullmix_analysis, get_session, get_settings
from dance.api.schemas import (
    ColumnRecOut,
    ColumnRecsRequest,
    ColumnRecsResponse,
    RecommendationOut,
    RecommendRequest,
    TextRecommendRequest,
)
from dance.config import Settings
from dance.core.database import Track
from dance.recommender.recommender import (
    VALID_COLUMNS,
    Recommender,
    recommend_by_column,
)

logger = logging.getLogger(__name__)
_clap_lock = threading.Lock()

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _run_recommend(
    session: Session,
    *,
    seeds: list[int],
    k: int,
    exclude: list[int] | None,
) -> list[dict]:
    rec = Recommender(session)
    results = rec.recommend(seeds=seeds, k=k, exclude=exclude)
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
        exclude=None,
    )


def _get_text_encoder(request: Request, settings: Settings):
    """Lazy-load the EmbeddingStage on first text query; reuse it after.

    Returns a callable ``encode_text(query) -> np.ndarray`` that acquires
    the process-wide GPU semaphore before running. Without this gate,
    text queries get starved when the dispatcher is also doing CLAP /
    Demucs work on MPS — the small fast text encode never wins a window
    in the queue.
    """
    from dance.pipeline._gpu import GPU_SEMAPHORE

    stage = request.app.state.embedding_stage
    if stage is None:
        with _clap_lock:
            stage = request.app.state.embedding_stage
            if stage is None:
                from dance.pipeline.stages.embed import EmbeddingStage

                stage = EmbeddingStage()
                try:
                    # Gate the model load too — first-call ~12-30s on MPS is
                    # painful but at least bounded with respect to dispatcher.
                    with GPU_SEMAPHORE:
                        stage._ensure_model(settings)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("CLAP load failed")
                    raise HTTPException(
                        status_code=503, detail=f"CLAP model unavailable: {exc}"
                    ) from exc
                request.app.state.embedding_stage = stage

    def _gated_encode(query: str):
        with GPU_SEMAPHORE:
            return stage.encode_text(query)

    return _gated_encode


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


# ---------------------------------------------------------------------------
# Per-column rec stream — the live-remixing rec backbone (Phase 3 redesign).
# ---------------------------------------------------------------------------


@router.post("/by-column", response_model=ColumnRecsResponse)
def recommend_by_column_route(
    body: ColumnRecsRequest,
    session: Session = Depends(get_session),
) -> ColumnRecsResponse:
    """Top-K candidate stems (or tracks for ``column='mix'``) for one column,
    re-scored against the currently-active combo of stems.

    POST body shape:
        column: str  — one of drums/bass/vocals/other/mix
        combo_stem_ids: list[int] — stem_file_ids currently active in Live
        master_bpm: float | None — Live's master tempo
        k: int — top-K to return (default 5)
        exclude_track_ids: list[int] — tracks already loaded (don't repeat)
    """
    if body.column not in VALID_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown column {body.column!r}; valid: {VALID_COLUMNS}",
        )

    results = recommend_by_column(
        session,
        column=body.column,
        combo_stem_ids=body.combo_stem_ids,
        master_bpm=body.master_bpm,
        k=body.k,
        exclude_track_ids=body.exclude_track_ids,
    )

    recs_out: list[ColumnRecOut] = []
    for r in results:
        track = session.get(Track, r.track_id)
        analysis = fullmix_analysis(session, r.track_id)
        recs_out.append(
            ColumnRecOut(
                track_id=r.track_id,
                stem_file_id=r.stem_file_id,
                track_title=getattr(track, "title", None),
                track_artist=getattr(track, "artist", None),
                bpm=getattr(analysis, "bpm", None),
                key_camelot=getattr(analysis, "key_camelot", None),
                floor_energy=getattr(analysis, "floor_energy", None),
                score=r.score,
                score_breakdown=r.score_breakdown,
                reasons=r.reasons,
            )
        )

    return ColumnRecsResponse(
        column=body.column,
        combo_size=len(body.combo_stem_ids),
        recs=recs_out,
    )
