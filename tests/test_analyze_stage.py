"""Integration test: AnalysisStage on synthetic audio.

Hinge test for the foundation — if this works, the rest of the pipeline
stages have a solid base to build on.
"""

from __future__ import annotations

import pytest

from dance.config import Settings
from dance.core.database import AudioAnalysis, TrackState
from dance.pipeline.stages.analyze import AnalysisStage
from tests.audio_fixtures import TrackSpec, write_track


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
    )


def test_analyze_real_synthetic_track(tmp_path, session, make_track, settings):
    audio_path = write_track(tmp_path / "house128.wav", TrackSpec(bpm=128.0, bars=4))
    track = make_track(file_path=str(audio_path), file_name=audio_path.name)
    session.commit()

    stage = AnalysisStage()
    ok = stage.process(session, track, settings)
    assert ok is True

    session.refresh(track)
    assert track.state == TrackState.ANALYZED.value
    assert track.duration_seconds is not None
    assert track.duration_seconds > 5  # 4 bars at 128 BPM ≈ 7.5s

    analysis = (
        session.query(AudioAnalysis)
        .filter_by(track_id=track.id, stem_file_id=None)
        .one()
    )
    # BPM may be half/double — allow either.
    candidates = [analysis.bpm, analysis.bpm * 2, analysis.bpm / 2]
    assert min(abs(c - 128.0) for c in candidates) < 4.0
    assert analysis.key_camelot is not None
    assert 1 <= analysis.floor_energy <= 10
    assert 0.0 <= analysis.energy_overall <= 1.0
    assert analysis.analyzed_at is not None


def test_analyze_missing_file_marks_error(session, make_track, settings):
    track = make_track(file_path="/does/not/exist.wav")
    session.commit()

    stage = AnalysisStage()
    ok = stage.process(session, track, settings)
    assert ok is False

    session.refresh(track)
    assert track.state == TrackState.ERROR.value
    assert track.error_message is not None
