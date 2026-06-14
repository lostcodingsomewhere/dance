"""Tests for the recommendation graph builder and query layer."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    EdgeKind,
    TrackEdge,
    TrackEmbedding,
    now_utc,
)
from dance.core.serialization import encode_embedding
from dance.recommender import (
    GraphBuilder,
    RecommendationResult,
    Recommender,
    recommend,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        recommender_top_k=5,
    )


def _add_analysis(
    session: Session,
    track_id: int,
    *,
    key: str | None = None,
    bpm: float | None = None,
) -> AudioAnalysis:
    row = AudioAnalysis(
        track_id=track_id,
        stem_file_id=None,
        bpm=bpm,
        key_camelot=key,
        analyzed_at=now_utc(),
    )
    session.add(row)
    session.flush()
    return row


def _add_embedding(
    session: Session,
    track_id: int,
    vec: np.ndarray,
    model: str = "laion/clap-htsat-unfused",
) -> TrackEmbedding:
    row = TrackEmbedding(
        track_id=track_id,
        stem_file_id=None,
        model=model,
        model_version=None,
        dim=int(vec.shape[0]),
        embedding=encode_embedding(vec.astype(np.float32)),
    )
    session.add(row)
    session.flush()
    return row


def _edges(session: Session, kind: EdgeKind) -> list[TrackEdge]:
    return (
        session.query(TrackEdge).filter(TrackEdge.kind == kind.value).all()
    )


# ===========================================================================
# GraphBuilder — empty DB
# ===========================================================================


def test_build_empty_db_returns_zero_counts(session, settings):
    gb = GraphBuilder(session, settings)
    out = gb.build()
    assert out == {EdgeKind.EMBEDDING_NEIGHBOR.value: 0}
    assert session.query(TrackEdge).count() == 0


# ===========================================================================
# Embedding neighbors
# ===========================================================================


def test_embedding_close_far(session, make_track, settings):
    """Track A close to B, far from C — A's top-K should include B but not C."""
    a = make_track()
    b = make_track()
    c = make_track()

    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    close = np.array([0.99, 0.05, 0.0, 0.0], dtype=np.float32)
    far = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    _add_embedding(session, a.id, base)
    _add_embedding(session, b.id, close)
    _add_embedding(session, c.id, far)
    session.commit()

    s = Settings(
        library_dir=settings.library_dir,
        stems_dir=settings.stems_dir,
        data_dir=settings.data_dir,
        recommender_top_k=1,
    )
    GraphBuilder(session, s).build(kinds=[EdgeKind.EMBEDDING_NEIGHBOR])

    edges = _edges(session, EdgeKind.EMBEDDING_NEIGHBOR)
    # A's top-1 = B (close). B's top-1 = A. C's top-1 = whichever is least bad,
    # but C is far from both — its top-1 still gets emitted.
    # Bidirectional materialization guarantees (A, B) and (B, A) exist.
    pairs = {(e.from_track_id, e.to_track_id): e for e in edges}
    assert (a.id, b.id) in pairs
    assert (b.id, a.id) in pairs

    # The cosine for A<->B should be very high (~1.0 after renorm).
    assert pairs[(a.id, b.id)].weight > 0.9

    # A's edge to C should NOT exist as a top-K (k=1 so only one neighbor).
    assert (a.id, c.id) not in pairs


def test_embedding_skips_non_matching_model(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_embedding(session, a.id, np.ones(4, dtype=np.float32), model="other-model")
    _add_embedding(session, b.id, np.ones(4, dtype=np.float32), model="other-model")
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.EMBEDDING_NEIGHBOR])
    assert _edges(session, EdgeKind.EMBEDDING_NEIGHBOR) == []


# ===========================================================================
# Idempotency + incremental
# ===========================================================================


def test_idempotent_double_build(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_embedding(session, a.id, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    _add_embedding(session, b.id, np.array([0.99, 0.05, 0.0, 0.0], dtype=np.float32))
    session.commit()

    gb = GraphBuilder(session, settings)
    counts1 = gb.build()
    edges_after_first = session.query(TrackEdge).count()
    counts2 = gb.build()
    edges_after_second = session.query(TrackEdge).count()
    assert edges_after_first == edges_after_second
    assert counts1 == counts2


def test_incremental_only_touches_listed_tracks(session, make_track, settings):
    a = make_track()
    b = make_track()
    c = make_track()
    # All three with embeddings that form distinct neighbours.
    _add_embedding(session, a.id, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    _add_embedding(session, b.id, np.array([0.9, 0.1, 0.0], dtype=np.float32))
    _add_embedding(session, c.id, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    session.commit()

    gb = GraphBuilder(session, settings)
    gb.build()

    # Manually set one edge's timestamp to a sentinel value.
    e_bc = (
        session.query(TrackEdge)
        .filter(
            TrackEdge.from_track_id == b.id,
            TrackEdge.to_track_id == c.id,
            TrackEdge.kind == EdgeKind.EMBEDDING_NEIGHBOR.value,
        )
        .first()
    )
    if e_bc is not None:
        sentinel = datetime(2000, 1, 1)
        e_bc.computed_at = sentinel
        session.commit()

        # Incremental build for [a.id] only: edges (b,c) and (c,b) must be left alone.
        gb.build(track_ids=[a.id])

        e_bc_after = (
            session.query(TrackEdge)
            .filter(
                TrackEdge.from_track_id == b.id,
                TrackEdge.to_track_id == c.id,
                TrackEdge.kind == EdgeKind.EMBEDDING_NEIGHBOR.value,
            )
            .first()
        )
        assert e_bc_after is not None
        assert e_bc_after.computed_at == sentinel


# ===========================================================================
# Recommender — live scoring
# ===========================================================================


def _setup_recommend_tracks(session, make_track, settings):
    """Create seed + two candidates with embeddings, analysis, and edges.

    Returns (seed_id, close_id, far_id).
    close is near the seed in embedding space + compatible key/BPM.
    far is distant in embedding space + incompatible key/BPM.
    """
    seed = make_track()
    close = make_track()
    far = make_track()

    # Embeddings
    seed_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    close_vec = np.array([0.99, 0.1, 0.0, 0.0], dtype=np.float32)
    far_vec = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _add_embedding(session, seed.id, seed_vec)
    _add_embedding(session, close.id, close_vec)
    _add_embedding(session, far.id, far_vec)

    # Analysis — same key + close BPM for close; different key + far BPM for far
    _add_analysis(session, seed.id, key="8A", bpm=128.0)
    _add_analysis(session, close.id, key="8A", bpm=129.0)
    _add_analysis(session, far.id, key="2B", bpm=100.0)

    session.commit()

    # Build the embedding-neighbor graph so the candidate pool is populated.
    s = Settings(
        library_dir=settings.library_dir,
        stems_dir=settings.stems_dir,
        data_dir=settings.data_dir,
        recommender_top_k=5,
    )
    GraphBuilder(session, s).build()

    return seed.id, close.id, far.id


def test_recommend_empty_seeds_returns_empty(session, settings):
    assert Recommender(session).recommend([]) == []


def test_recommend_close_outranks_far(session, make_track, settings):
    seed_id, close_id, far_id = _setup_recommend_tracks(session, make_track, settings)
    results = Recommender(session).recommend([seed_id], k=10)
    track_ids = [r.track_id for r in results]
    assert close_id in track_ids
    assert far_id in track_ids
    close_score = next(r.score for r in results if r.track_id == close_id)
    far_score = next(r.score for r in results if r.track_id == far_id)
    assert close_score > far_score


def test_recommend_seeds_excluded_from_results(session, make_track, settings):
    seed_id, close_id, far_id = _setup_recommend_tracks(session, make_track, settings)
    results = Recommender(session).recommend([seed_id], k=10)
    result_ids = {r.track_id for r in results}
    assert seed_id not in result_ids


def test_recommend_exclude_param_respected(session, make_track, settings):
    seed_id, close_id, far_id = _setup_recommend_tracks(session, make_track, settings)
    results = Recommender(session).recommend([seed_id], exclude=[close_id])
    result_ids = {r.track_id for r in results}
    assert close_id not in result_ids


def test_recommend_top_k_limit(session, make_track, settings):
    seed_id, close_id, far_id = _setup_recommend_tracks(session, make_track, settings)
    results = Recommender(session).recommend([seed_id], k=1)
    assert len(results) <= 1


def test_recommend_reasons_populated(session, make_track, settings):
    seed_id, close_id, far_id = _setup_recommend_tracks(session, make_track, settings)
    results = Recommender(session).recommend([seed_id], k=10)
    assert len(results) >= 1
    for r in results:
        assert isinstance(r.reasons, list)
        # Each reason is a dict with kind + value
        for reason in r.reasons:
            assert "kind" in reason
            assert "value" in reason


def test_recommend_module_level_convenience(session, make_track, settings):
    seed_id, close_id, far_id = _setup_recommend_tracks(session, make_track, settings)
    out = recommend(session, [seed_id])
    assert all(isinstance(r, RecommendationResult) for r in out)
    assert any(r.track_id == close_id for r in out)


def test_recommend_no_edges_returns_empty(session, make_track, settings):
    """If no embedding-neighbor edges exist, result is empty (no candidates)."""
    seed = make_track()
    _add_embedding(session, seed.id, np.array([1.0, 0.0], dtype=np.float32))
    _add_analysis(session, seed.id, key="8A", bpm=128.0)
    session.commit()
    # Don't build the graph — so no edges exist.
    results = Recommender(session).recommend([seed.id])
    assert results == []


# ---------------------------------------------------------------------------
# recommend_by_text — CLAP text↔audio joint embedding
# ---------------------------------------------------------------------------


def _add_full_mix_embedding(session, track_id: int, vector: np.ndarray, model: str = "test-clap") -> None:
    session.add(
        TrackEmbedding(
            track_id=track_id,
            stem_file_id=None,
            model=model,
            model_version=None,
            dim=int(vector.shape[0]),
            embedding=encode_embedding(vector.astype(np.float32)),
            created_at=now_utc(),
        )
    )


def test_recommend_by_text_orders_by_cosine(session, make_track, settings):
    """The track whose embedding is closest to the text query ranks first."""
    near = make_track()
    far = make_track()
    other = make_track()

    # Crafted vectors — `near` aligned with [1,0,0], `far` perpendicular,
    # `other` somewhere in between.
    _add_full_mix_embedding(session, near.id, np.array([1.0, 0.0, 0.0]))
    _add_full_mix_embedding(session, far.id, np.array([0.0, 1.0, 0.0]))
    _add_full_mix_embedding(session, other.id, np.array([0.7, 0.7, 0.0]))
    session.commit()

    def fake_encoder(q: str) -> np.ndarray:
        assert q == "punchy techy with vocals"
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = Recommender(session).recommend_by_text(
        "punchy techy with vocals", text_encoder=fake_encoder, k=3
    )

    assert [r.track_id for r in results] == [near.id, other.id, far.id]
    assert results[0].score > results[1].score > results[2].score
    # `near` is exact match → cosine = 1.0
    assert pytest.approx(results[0].score, abs=1e-5) == 1.0
    # Reasons populated
    assert results[0].reasons[0]["kind"] == "text_query"
    assert results[0].reasons[0]["query"] == "punchy techy with vocals"


def test_recommend_by_text_excludes_listed(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_full_mix_embedding(session, a.id, np.array([1.0, 0.0]))
    _add_full_mix_embedding(session, b.id, np.array([1.0, 0.0]))
    session.commit()

    out = Recommender(session).recommend_by_text(
        "q", text_encoder=lambda _q: np.array([1.0, 0.0], dtype=np.float32),
        exclude=[a.id],
    )
    assert [r.track_id for r in out] == [b.id]


def test_recommend_by_text_filters_by_model(session, make_track, settings):
    """Only embeddings matching the specified model are considered."""
    a = make_track()
    b = make_track()
    _add_full_mix_embedding(session, a.id, np.array([1.0, 0.0]), model="model-A")
    _add_full_mix_embedding(session, b.id, np.array([1.0, 0.0]), model="model-B")
    session.commit()

    out_a = Recommender(session).recommend_by_text(
        "q", text_encoder=lambda _q: np.array([1.0, 0.0], dtype=np.float32),
        model_name="model-A",
    )
    assert [r.track_id for r in out_a] == [a.id]


def test_recommend_by_text_empty_query_returns_empty(session, make_track):
    out = Recommender(session).recommend_by_text(
        "  ", text_encoder=lambda _q: np.zeros(2, dtype=np.float32)
    )
    assert out == []


def test_recommend_by_text_no_embeddings_returns_empty(session, make_track):
    make_track()  # has no embedding rows
    out = Recommender(session).recommend_by_text(
        "q", text_encoder=lambda _q: np.array([1.0, 0.0], dtype=np.float32)
    )
    assert out == []
