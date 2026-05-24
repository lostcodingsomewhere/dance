"""Tests for :mod:`dance.recommender.tail` and the
``GET /sets/{id}/tail-recs`` endpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dance.core.database import (
    AudioAnalysis,
    Set,
    SetTrack,
    TrackEmbedding,
    now_utc,
)
from dance.core.serialization import encode_embedding
from dance.recommender.tail import (
    _energy_score,
    _project_energy,
    tail_recs_for_set,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_project_energy_flat_returns_flat():
    assert _project_energy([5, 5, 5, 5]) == pytest.approx(5.0)


def test_project_energy_upward_slope_extrapolates():
    # 3, 4, 5, 6 → slope=1, next = 7
    assert _project_energy([3, 4, 5, 6]) == pytest.approx(7.0)


def test_project_energy_clamps_to_range():
    # 8, 9, 10, 10 → projection could go above 10; clamp.
    assert _project_energy([8, 9, 10, 10]) <= 10.0


def test_project_energy_handles_empty_and_single():
    assert _project_energy([]) is None
    assert _project_energy([7]) == pytest.approx(7.0)


def test_energy_score_at_target_is_one():
    assert _energy_score(6, 6.0) == pytest.approx(1.0)


def test_energy_score_falls_off_at_three_floors():
    assert _energy_score(3, 6.0) == pytest.approx(0.0)
    assert _energy_score(6, 3.0) == pytest.approx(0.0)


def test_energy_score_neutral_when_missing():
    assert _energy_score(None, 5.0) == 0.5
    assert _energy_score(5, None) == 0.5


# ---------------------------------------------------------------------------
# tail_recs_for_set — scoring against a real (fixture) DB
# ---------------------------------------------------------------------------


def _add_analysis(
    session,
    track_id: int,
    *,
    bpm: float = 124.0,
    key: str = "8A",
    energy: int = 6,
) -> AudioAnalysis:
    a = AudioAnalysis(
        track_id=track_id,
        stem_file_id=None,
        bpm=bpm,
        key_camelot=key,
        floor_energy=energy,
        analyzed_at=now_utc(),
    )
    session.add(a)
    session.flush()
    return a


def _add_embedding(session, track_id: int, vec: np.ndarray) -> TrackEmbedding:
    row = TrackEmbedding(
        track_id=track_id,
        stem_file_id=None,
        model="test-clap",
        model_version=None,
        dim=int(vec.shape[0]),
        embedding=encode_embedding(vec.astype(np.float32)),
        created_at=now_utc(),
    )
    session.add(row)
    session.flush()
    return row


def _make_set(session, tracks: list[Any], name: str = "S") -> Set:
    s = Set(name=name)
    session.add(s)
    session.flush()
    for i, t in enumerate(tracks):
        session.add(SetTrack(set_id=s.id, track_id=t.id, position=i))
    session.flush()
    return s


def test_tail_recs_prefers_close_bpm_and_same_key(session, make_track):
    """With a set at 124 BPM in 8A, a candidate at 124 BPM in 8A should
    outrank one at 140 BPM in 5A."""
    set_t1 = make_track(title="set1")
    set_t2 = make_track(title="set2")
    _add_analysis(session, set_t1.id, bpm=124, key="8A", energy=6)
    _add_analysis(session, set_t2.id, bpm=124, key="8A", energy=6)

    close = make_track(title="close")
    far = make_track(title="far")
    _add_analysis(session, close.id, bpm=124, key="8A", energy=6)
    _add_analysis(session, far.id, bpm=140, key="5A", energy=9)

    s = _make_set(session, [set_t1, set_t2])
    session.commit()

    results = tail_recs_for_set(session, set_id=s.id, k=10)
    assert len(results) == 2
    assert results[0].track_id == close.id
    assert results[0].score > results[1].score


def test_tail_recs_excludes_tracks_already_in_set(session, make_track):
    a = make_track(title="a")
    b = make_track(title="b")
    _add_analysis(session, a.id, bpm=124, key="8A")
    _add_analysis(session, b.id, bpm=124, key="8A")
    s = _make_set(session, [a, b])
    session.commit()

    results = tail_recs_for_set(session, set_id=s.id, k=10)
    assert all(r.track_id not in {a.id, b.id} for r in results)


def test_tail_recs_embedding_signal_dominates_when_bpm_keys_tied(session, make_track):
    """All candidates have identical BPM/key/energy; only embedding differs.
    The candidate aligned with the trailing-window embedding should win."""
    rng = np.random.default_rng(0)
    target_vec = rng.standard_normal(16).astype(np.float32)
    target_vec /= np.linalg.norm(target_vec)
    orthogonal = rng.standard_normal(16).astype(np.float32)
    orthogonal -= orthogonal.dot(target_vec) * target_vec
    orthogonal /= np.linalg.norm(orthogonal)

    set_t = make_track(title="trail")
    _add_analysis(session, set_t.id, bpm=124, key="8A", energy=6)
    _add_embedding(session, set_t.id, target_vec)

    aligned = make_track(title="aligned")
    misaligned = make_track(title="misaligned")
    _add_analysis(session, aligned.id, bpm=124, key="8A", energy=6)
    _add_analysis(session, misaligned.id, bpm=124, key="8A", energy=6)
    _add_embedding(session, aligned.id, target_vec)
    _add_embedding(session, misaligned.id, orthogonal)

    s = _make_set(session, [set_t])
    session.commit()

    results = tail_recs_for_set(session, set_id=s.id, k=10)
    assert results[0].track_id == aligned.id
    breakdown = results[0].score_breakdown
    # Aligned should hit near-perfect embedding similarity.
    assert breakdown["embedding"] > 0.9


def test_tail_recs_energy_arc_extrapolation(session, make_track):
    """Set energies rise 3, 4, 5 → projection ≈ 6. Candidate at energy 6
    should outrank an identical candidate at energy 9 (overshoots arc)."""
    rng = np.random.default_rng(1)
    fixed_vec = rng.standard_normal(8).astype(np.float32)

    a = make_track(title="a")
    b = make_track(title="b")
    c = make_track(title="c")
    _add_analysis(session, a.id, bpm=124, key="8A", energy=3)
    _add_analysis(session, b.id, bpm=124, key="8A", energy=4)
    _add_analysis(session, c.id, bpm=124, key="8A", energy=5)
    _add_embedding(session, a.id, fixed_vec)
    _add_embedding(session, b.id, fixed_vec)
    _add_embedding(session, c.id, fixed_vec)

    on_arc = make_track(title="on-arc")
    overshoot = make_track(title="overshoot")
    _add_analysis(session, on_arc.id, bpm=124, key="8A", energy=6)
    _add_analysis(session, overshoot.id, bpm=124, key="8A", energy=9)
    _add_embedding(session, on_arc.id, fixed_vec)
    _add_embedding(session, overshoot.id, fixed_vec)

    s = _make_set(session, [a, b, c])
    session.commit()

    results = {r.track_id: r for r in tail_recs_for_set(session, set_id=s.id)}
    assert results[on_arc.id].score > results[overshoot.id].score
    assert (
        results[on_arc.id].score_breakdown["energy"]
        > results[overshoot.id].score_breakdown["energy"]
    )


def test_tail_recs_empty_set_returns_zero_scores_not_crash(session, make_track):
    candidate = make_track(title="c")
    _add_analysis(session, candidate.id, bpm=124, key="8A", energy=6)
    s = _make_set(session, [])
    session.commit()

    results = tail_recs_for_set(session, set_id=s.id, k=5)
    # Returns *something* (single candidate) but all scores are zero — the
    # rail will show "set is empty, add tracks first" rather than render.
    assert len(results) == 1
    assert results[0].score == 0.0


def test_tail_recs_respects_window_size(session, make_track):
    """A very old high-BPM opener shouldn't dominate when window=2 captures
    only the recent calmer run."""
    old = make_track(title="old")
    recent_a = make_track(title="recent-a")
    recent_b = make_track(title="recent-b")
    _add_analysis(session, old.id, bpm=145, key="8A", energy=9)
    _add_analysis(session, recent_a.id, bpm=120, key="8A", energy=5)
    _add_analysis(session, recent_b.id, bpm=120, key="8A", energy=5)

    cand_old_bpm = make_track(title="cand-old-bpm")
    cand_recent_bpm = make_track(title="cand-recent-bpm")
    _add_analysis(session, cand_old_bpm.id, bpm=145, key="8A", energy=9)
    _add_analysis(session, cand_recent_bpm.id, bpm=120, key="8A", energy=5)

    s = _make_set(session, [old, recent_a, recent_b])
    session.commit()

    results = tail_recs_for_set(session, set_id=s.id, k=10, window=2)
    by_id = {r.track_id: r for r in results}
    assert by_id[cand_recent_bpm.id].score > by_id[cand_old_bpm.id].score


# Endpoint integration tests for /sets/{id}/tail-recs live in tests/test_api.py
# alongside the other Sets HTTP tests so they share the ``client`` fixture.
