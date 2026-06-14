"""Unit tests for :mod:`dance.recommender.journey` — trajectory helpers and
the JourneyState builders."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from dance.core.database import AudioAnalysis, TrackEmbedding, now_utc
from dance.core.serialization import encode_embedding
from dance.recommender import journey as jn

# --- pure helpers ---------------------------------------------------------


def test_recency_weighted_vibe_empty():
    assert jn.recency_weighted_vibe([]) is None


def test_recency_weighted_vibe_is_unit():
    vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    out = jn.recency_weighted_vibe(vecs)
    assert out is not None
    assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-5)


def test_recency_weighting_favors_recent():
    # Newest vector weighted more → result leans toward [0, 1].
    vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    out = jn.recency_weighted_vibe(vecs)
    assert out[1] > out[0]


def test_vibe_trend_direction():
    # Moving from x-axis toward y-axis → trend points toward +y.
    vecs = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    trend = jn.vibe_trend(vecs)
    assert trend is not None
    assert trend[1] > 0.0


def test_vibe_trend_too_short():
    assert jn.vibe_trend([np.array([1.0, 0.0])]) is None


def test_vibe_trend_no_movement():
    vecs = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    assert jn.vibe_trend(vecs) is None


# --- JourneyState.vibe_target / vibe_score --------------------------------


def test_vibe_target_falls_back_without_trend():
    target = np.array([1.0, 0.0], dtype=np.float32)
    st = jn.JourneyState(target_vibe_vec=target, vibe_trend_vec=None)
    np.testing.assert_allclose(st.vibe_target(), jn._unit(target), atol=1e-5)


def test_vibe_target_projects_along_trend():
    target = np.array([1.0, 0.0], dtype=np.float32)
    trend = np.array([0.0, 1.0], dtype=np.float32)
    st = jn.JourneyState(target_vibe_vec=target, vibe_trend_vec=trend)
    projected = st.vibe_target(beta=1.0)
    # Pushed off the x-axis toward +y.
    assert projected[1] > 0.0


def test_vibe_score_none_target():
    st = jn.JourneyState()
    assert st.vibe_score(np.array([1.0, 0.0])) is None


# --- DB builders ----------------------------------------------------------


def _seed_track(session: Session, track_factory, tid_vec, key, bpm, energy):
    track = track_factory()
    vec = np.array(tid_vec, dtype=np.float32)
    session.add(
        TrackEmbedding(
            track_id=track.id,
            stem_file_id=None,
            model="laion/clap-htsat-unfused",
            dim=int(vec.shape[0]),
            embedding=encode_embedding(vec),
        )
    )
    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=None,
            bpm=bpm,
            key_camelot=key,
            floor_energy=energy,
            analyzed_at=now_utc(),
        )
    )
    session.flush()
    return track.id


def test_journey_from_tracks_builds_targets(session, make_track):
    a = _seed_track(session, make_track, [1.0, 0.0], "8A", 124.0, 3)
    b = _seed_track(session, make_track, [0.9, 0.1], "8A", 126.0, 4)
    c = _seed_track(session, make_track, [0.0, 1.0], "9A", 128.0, 5)

    st = jn.journey_from_tracks(session, [a, b, c], window=5)
    assert st.target_bpm == pytest.approx(126.0)  # median of 124,126,128
    assert st.projected_energy == pytest.approx(6.0)  # 3,4,5 → next 6
    assert set(st.used_source_ids) == {a, b, c}
    assert st.target_vibe_vec is not None
    assert st.vibe_trend_vec is not None  # moved from x toward y


def test_journey_from_tracks_empty():
    # Empty list short-circuits before any query, so session is never touched.
    out = jn.journey_from_tracks(session=None, ordered_track_ids=[], window=5)  # type: ignore[arg-type]
    assert out.target_vibe_vec is None
    assert out.target_bpm is None


def test_journey_from_combo(session, make_track):
    from dance.core.database import StemFile

    track = make_track()
    stem = StemFile(track_id=track.id, kind="drums", path="/tmp/d.wav")
    session.add(stem)
    session.flush()
    vec = np.array([1.0, 0.0], dtype=np.float32)
    session.add(
        TrackEmbedding(
            track_id=track.id,
            stem_file_id=stem.id,
            model="laion/clap-htsat-unfused",
            dim=2,
            embedding=encode_embedding(vec),
        )
    )
    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=stem.id,
            bpm=128.0,
            dominant_pitch_camelot="8A",
            analyzed_at=now_utc(),
        )
    )
    session.flush()

    st = jn.journey_from_combo(session, [stem.id], master_bpm=130.0)
    assert st.target_bpm == pytest.approx(130.0)  # master wins
    assert st.target_vibe_vec is not None
    assert st.current_combo_stem_ids == [stem.id]
