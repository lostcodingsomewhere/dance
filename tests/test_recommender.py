"""Tests for the recommendation graph builder and query layer."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    now_utc,
    AudioAnalysis,
    EdgeKind,
    Tag,
    TagKind,
    TagSource,
    TrackEdge,
    TrackEmbedding,
    TrackTag,
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


def _add_tag(session: Session, value: str) -> Tag:
    tag = Tag(kind=TagKind.SUBGENRE.value, value=value, normalized_value=value.lower())
    session.add(tag)
    session.flush()
    return tag


def _link_tag(session: Session, track_id: int, tag_id: int) -> None:
    session.add(
        TrackTag(
            track_id=track_id,
            tag_id=tag_id,
            source=TagSource.MANUAL.value,
            confidence=1.0,
        )
    )
    session.flush()


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
    assert out == {
        EdgeKind.HARMONIC_COMPAT.value: 0,
        EdgeKind.TEMPO_COMPAT.value: 0,
        EdgeKind.EMBEDDING_NEIGHBOR.value: 0,
        EdgeKind.TAG_OVERLAP.value: 0,
    }
    assert session.query(TrackEdge).count() == 0


# ===========================================================================
# Harmonic
# ===========================================================================


def test_harmonic_same_key_weight_one(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, key="8A")
    _add_analysis(session, b.id, key="8A")
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.HARMONIC_COMPAT])
    edges = _edges(session, EdgeKind.HARMONIC_COMPAT)
    assert len(edges) == 2  # both directions
    for e in edges:
        assert e.weight == 1.0
        assert json.loads(e.meta)["distance"] == "exact"
    assert {(e.from_track_id, e.to_track_id) for e in edges} == {
        (a.id, b.id),
        (b.id, a.id),
    }


def test_harmonic_adjacent_keys_weight_zero_seven(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, key="8A")
    _add_analysis(session, b.id, key="9A")
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.HARMONIC_COMPAT])
    edges = _edges(session, EdgeKind.HARMONIC_COMPAT)
    assert len(edges) == 2
    for e in edges:
        assert e.weight == pytest.approx(0.7)
        assert json.loads(e.meta)["distance"] == "adjacent"


def test_harmonic_relative_keys_weight_zero_eight(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, key="8A")
    _add_analysis(session, b.id, key="8B")
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.HARMONIC_COMPAT])
    edges = _edges(session, EdgeKind.HARMONIC_COMPAT)
    assert len(edges) == 2
    for e in edges:
        assert e.weight == pytest.approx(0.8)
        assert json.loads(e.meta)["distance"] == "relative"


def test_harmonic_incompatible_no_edges(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, key="8A")
    _add_analysis(session, b.id, key="5A")  # not adjacent (diff = 3)
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.HARMONIC_COMPAT])
    assert _edges(session, EdgeKind.HARMONIC_COMPAT) == []


def test_harmonic_wraps_around_twelve_to_one(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, key="12A")
    _add_analysis(session, b.id, key="1A")
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.HARMONIC_COMPAT])
    edges = _edges(session, EdgeKind.HARMONIC_COMPAT)
    assert len(edges) == 2
    for e in edges:
        assert e.weight == pytest.approx(0.7)


# ===========================================================================
# Tempo
# ===========================================================================


@pytest.mark.parametrize(
    "a_bpm,b_bpm,expected_weight",
    [
        (128.0, 130.0, 1.0),
        (128.0, 134.0, 0.7),
        (128.0, 64.0, 0.5),  # halftime
    ],
)
def test_tempo_buckets(session, make_track, settings, a_bpm, b_bpm, expected_weight):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, bpm=a_bpm)
    _add_analysis(session, b.id, bpm=b_bpm)
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.TEMPO_COMPAT])
    edges = _edges(session, EdgeKind.TEMPO_COMPAT)
    assert len(edges) == 2
    for e in edges:
        assert e.weight == pytest.approx(expected_weight)


def test_tempo_no_edge_when_far_apart(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, bpm=128.0)
    _add_analysis(session, b.id, bpm=90.0)
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.TEMPO_COMPAT])
    assert _edges(session, EdgeKind.TEMPO_COMPAT) == []


def test_tempo_halftime_meta_flag(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, bpm=128.0)
    _add_analysis(session, b.id, bpm=64.0)
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.TEMPO_COMPAT])
    for e in _edges(session, EdgeKind.TEMPO_COMPAT):
        meta = json.loads(e.meta)
        assert meta["halftime"] is True
        assert meta["bpm_delta"] == pytest.approx(64.0)


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
# Tag overlap
# ===========================================================================


def test_tag_overlap_jaccard(session, make_track, settings):
    a = make_track()
    b = make_track()
    # A: {t1, t2, t3}, B: {t2, t3, t4}. shared=2, union=4, jaccard=0.5
    t1 = _add_tag(session, "house")
    t2 = _add_tag(session, "tech-house")
    t3 = _add_tag(session, "minimal")
    t4 = _add_tag(session, "deep")
    for tid in (t1.id, t2.id, t3.id):
        _link_tag(session, a.id, tid)
    for tid in (t2.id, t3.id, t4.id):
        _link_tag(session, b.id, tid)
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.TAG_OVERLAP])
    edges = _edges(session, EdgeKind.TAG_OVERLAP)
    assert len(edges) == 2  # both directions
    for e in edges:
        assert e.weight == pytest.approx(0.5)
        meta = json.loads(e.meta)
        assert meta["shared_count"] == 2
        assert meta["total_count"] == 4


def test_tag_overlap_no_shared_tags(session, make_track, settings):
    a = make_track()
    b = make_track()
    t1 = _add_tag(session, "house")
    t2 = _add_tag(session, "techno")
    _link_tag(session, a.id, t1.id)
    _link_tag(session, b.id, t2.id)
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.TAG_OVERLAP])
    assert _edges(session, EdgeKind.TAG_OVERLAP) == []


def test_tag_overlap_below_threshold(session, make_track, settings):
    """A: 10 tags, B: shares 1 -> jaccard ~ 1/19 < 0.1, no edge."""
    a = make_track()
    b = make_track()
    shared = _add_tag(session, "house")
    _link_tag(session, a.id, shared.id)
    _link_tag(session, b.id, shared.id)
    for i in range(9):
        ta = _add_tag(session, f"a-only-{i}")
        tb = _add_tag(session, f"b-only-{i}")
        _link_tag(session, a.id, ta.id)
        _link_tag(session, b.id, tb.id)
    session.commit()

    GraphBuilder(session, settings).build(kinds=[EdgeKind.TAG_OVERLAP])
    assert _edges(session, EdgeKind.TAG_OVERLAP) == []


# ===========================================================================
# Bidirectional + idempotency + incremental
# ===========================================================================


def test_every_edge_appears_in_both_directions(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, key="8A", bpm=128.0)
    _add_analysis(session, b.id, key="8A", bpm=129.0)
    session.commit()

    GraphBuilder(session, settings).build()
    for kind in (EdgeKind.HARMONIC_COMPAT, EdgeKind.TEMPO_COMPAT):
        pairs = {(e.from_track_id, e.to_track_id) for e in _edges(session, kind)}
        assert (a.id, b.id) in pairs
        assert (b.id, a.id) in pairs


def test_idempotent_double_build(session, make_track, settings):
    a = make_track()
    b = make_track()
    _add_analysis(session, a.id, key="8A", bpm=128.0)
    _add_analysis(session, b.id, key="8A", bpm=129.0)
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
    _add_analysis(session, a.id, key="8A")
    _add_analysis(session, b.id, key="8A")
    _add_analysis(session, c.id, key="8A")
    session.commit()

    gb = GraphBuilder(session, settings)
    gb.build()
    initial = {
        (e.from_track_id, e.to_track_id): e.computed_at
        for e in _edges(session, EdgeKind.HARMONIC_COMPAT)
    }
    # Sanity: 3 tracks all mutually compatible => 6 directed edges
    assert len(initial) == 6

    # Manually mark one untouched edge's timestamp old so we can detect rewrite.
    e_bc = (
        session.query(TrackEdge)
        .filter(
            TrackEdge.from_track_id == b.id,
            TrackEdge.to_track_id == c.id,
            TrackEdge.kind == EdgeKind.HARMONIC_COMPAT.value,
        )
        .one()
    )
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
            TrackEdge.kind == EdgeKind.HARMONIC_COMPAT.value,
        )
        .one()
    )
    assert e_bc_after.computed_at == sentinel

    # And edges touching A should still exist (rewritten).
    pairs = {
        (e.from_track_id, e.to_track_id)
        for e in _edges(session, EdgeKind.HARMONIC_COMPAT)
    }
    assert (a.id, b.id) in pairs
    assert (a.id, c.id) in pairs


# ===========================================================================
# Recommender
# ===========================================================================


def test_recommend_returns_top_k_by_aggregate_score(session, make_track, settings):
    seed = make_track()
    near = make_track()
    far = make_track()
    # Big-weight edge to near, small-weight to far
    session.add(
        TrackEdge(
            from_track_id=seed.id,
            to_track_id=near.id,
            kind=EdgeKind.HARMONIC_COMPAT.value,
            weight=1.0,
            computed_at=now_utc(),
        )
    )
    session.add(
        TrackEdge(
            from_track_id=seed.id,
            to_track_id=far.id,
            kind=EdgeKind.HARMONIC_COMPAT.value,
            weight=0.7,
            computed_at=now_utc(),
        )
    )
    session.commit()

    results = Recommender(session).recommend([seed.id], k=10)
    assert [r.track_id for r in results] == [near.id, far.id]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.7)


def test_recommend_excludes_listed_tracks(session, make_track, settings):
    seed = make_track()
    a = make_track()
    b = make_track()
    for t, w in ((a, 1.0), (b, 0.5)):
        session.add(
            TrackEdge(
                from_track_id=seed.id,
                to_track_id=t.id,
                kind=EdgeKind.HARMONIC_COMPAT.value,
                weight=w,
                computed_at=now_utc(),
            )
        )
    session.commit()

    results = Recommender(session).recommend([seed.id], exclude=[a.id])
    assert [r.track_id for r in results] == [b.id]


def test_recommend_restricts_by_kinds(session, make_track, settings):
    seed = make_track()
    target = make_track()
    session.add(
        TrackEdge(
            from_track_id=seed.id,
            to_track_id=target.id,
            kind=EdgeKind.TAG_OVERLAP.value,
            weight=1.0,
            computed_at=now_utc(),
        )
    )
    session.commit()

    # Only ask for harmonic edges — should get nothing.
    results = Recommender(session).recommend(
        [seed.id], kinds=[EdgeKind.HARMONIC_COMPAT]
    )
    assert results == []


def test_recommend_per_kind_weights(session, make_track, settings):
    seed = make_track()
    target = make_track()
    session.add(
        TrackEdge(
            from_track_id=seed.id,
            to_track_id=target.id,
            kind=EdgeKind.HARMONIC_COMPAT.value,
            weight=1.0,
            computed_at=now_utc(),
        )
    )
    session.commit()

    # Silencing harmonic should hide it entirely.
    results = Recommender(session).recommend(
        [seed.id], weights={EdgeKind.HARMONIC_COMPAT: 0.0}
    )
    assert results == []


def test_recommend_multiple_seeds_sum_scores(session, make_track, settings):
    s1 = make_track()
    s2 = make_track()
    both = make_track()
    only_s1 = make_track()
    for src, tgt, w in (
        (s1, both, 0.8),
        (s2, both, 0.6),
        (s1, only_s1, 0.9),
    ):
        session.add(
            TrackEdge(
                from_track_id=src.id,
                to_track_id=tgt.id,
                kind=EdgeKind.HARMONIC_COMPAT.value,
                weight=w,
                computed_at=now_utc(),
            )
        )
    session.commit()

    results = Recommender(session).recommend([s1.id, s2.id])
    assert [r.track_id for r in results] == [both.id, only_s1.id]
    assert results[0].score == pytest.approx(1.4)
    assert results[1].score == pytest.approx(0.9)


def test_recommend_seeds_never_in_results(session, make_track, settings):
    s1 = make_track()
    s2 = make_track()
    # Edge from s1 -> s2 should not surface s2 (it's a seed).
    session.add(
        TrackEdge(
            from_track_id=s1.id,
            to_track_id=s2.id,
            kind=EdgeKind.HARMONIC_COMPAT.value,
            weight=1.0,
            computed_at=now_utc(),
        )
    )
    session.commit()

    results = Recommender(session).recommend([s1.id, s2.id])
    assert results == []


def test_recommend_reasons_populated(session, make_track, settings):
    seed = make_track()
    target = make_track()
    session.add(
        TrackEdge(
            from_track_id=seed.id,
            to_track_id=target.id,
            kind=EdgeKind.HARMONIC_COMPAT.value,
            weight=0.7,
            computed_at=now_utc(),
        )
    )
    session.add(
        TrackEdge(
            from_track_id=seed.id,
            to_track_id=target.id,
            kind=EdgeKind.TEMPO_COMPAT.value,
            weight=1.0,
            computed_at=now_utc(),
        )
    )
    session.commit()

    results = Recommender(session).recommend([seed.id])
    assert len(results) == 1
    res = results[0]
    assert res.track_id == target.id
    assert len(res.reasons) == 2
    kinds = {r["kind"] for r in res.reasons}
    assert kinds == {EdgeKind.HARMONIC_COMPAT.value, EdgeKind.TEMPO_COMPAT.value}
    for r in res.reasons:
        assert r["from_seed"] == seed.id
        assert "weight" in r


def test_recommend_module_level_convenience(session, make_track, settings):
    seed = make_track()
    target = make_track()
    session.add(
        TrackEdge(
            from_track_id=seed.id,
            to_track_id=target.id,
            kind=EdgeKind.HARMONIC_COMPAT.value,
            weight=1.0,
            computed_at=now_utc(),
        )
    )
    session.commit()

    out = recommend(session, [seed.id])
    assert isinstance(out[0], RecommendationResult)
    assert out[0].track_id == target.id


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
