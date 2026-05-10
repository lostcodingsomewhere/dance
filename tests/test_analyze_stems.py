"""Integration tests for StemAnalysisStage.

Uses synthetic audio (same fixture as the full-mix analyzer) written to four
stem paths — fine for exercising the analysis math even though the four
files are identical in content.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from dance.config import Settings
from dance.core.database import (
    now_utc,
    AudioAnalysis,
    StemFile,
    StemKind,
    TrackState,
)
from dance.core.serialization import decode_curve
from dance.pipeline.stage import Stage
from dance.pipeline.stages.analyze_stems import StemAnalysisStage
from tests.audio_fixtures import TrackSpec, write_track


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def separated_track(tmp_path, session, make_track):
    """A SEPARATED track with 4 StemFile rows pointing at synthetic audio.

    All four stems point at the same file content (fine for the analyzer —
    each stem still gets its own AudioAnalysis row).
    """
    spec = TrackSpec(bpm=128.0, bars=4)
    audio_path = write_track(tmp_path / "mix.wav", spec)
    track = make_track(
        file_path=str(audio_path),
        file_name=audio_path.name,
        state=TrackState.SEPARATED.value,
    )
    session.flush()

    for kind in (StemKind.DRUMS, StemKind.BASS, StemKind.VOCALS, StemKind.OTHER):
        stem_path = tmp_path / f"{kind.value}.wav"
        write_track(stem_path, spec)
        session.add(
            StemFile(
                track_id=track.id,
                kind=kind.value,
                path=str(stem_path),
                model_used="test",
            )
        )

    # Full-mix analysis row to mirror the real pipeline order.
    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=None,
            bpm=128.0,
            analyzed_at=now_utc(),
        )
    )
    session.commit()
    return track


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_protocol_compliance():
    assert isinstance(StemAnalysisStage(), Stage)


def test_stage_attributes():
    stage = StemAnalysisStage()
    assert stage.name == "analyze_stems"
    assert stage.input_state == TrackState.SEPARATED
    assert stage.output_state == TrackState.STEMS_ANALYZED
    assert stage.error_state == TrackState.ERROR


def test_writes_one_analysis_per_stem(session, separated_track, settings):
    track = separated_track
    stage = StemAnalysisStage()

    ok = stage.process(session, track, settings)
    assert ok is True

    stem_rows = (
        session.query(AudioAnalysis)
        .filter(
            AudioAnalysis.track_id == track.id,
            AudioAnalysis.stem_file_id.isnot(None),
        )
        .all()
    )
    assert len(stem_rows) == 4

    # Full-mix row from the fixture remains untouched.
    full_mix = (
        session.query(AudioAnalysis)
        .filter_by(track_id=track.id, stem_file_id=None)
        .one()
    )
    assert full_mix.bpm == 128.0


def test_rms_curve_roundtrip(session, separated_track, settings):
    stage = StemAnalysisStage()
    stage.process(session, separated_track, settings)

    row = (
        session.query(AudioAnalysis)
        .filter(AudioAnalysis.stem_file_id.isnot(None))
        .first()
    )
    assert row is not None
    assert row.rms_curve is not None
    assert row.rms_curve_hop_ms == 100
    assert row.rms_curve_length > 0

    decoded = decode_curve(row.rms_curve, expected_length=row.rms_curve_length)
    assert decoded.shape[0] == row.rms_curve_length
    assert float(decoded.min()) >= 0.0
    assert float(decoded.max()) <= 1.0


def test_kind_specific_metrics(session, separated_track, settings):
    stage = StemAnalysisStage()
    stage.process(session, separated_track, settings)

    def _row(kind: StemKind) -> AudioAnalysis:
        stem = (
            session.query(StemFile)
            .filter_by(track_id=separated_track.id, kind=kind.value)
            .one()
        )
        return (
            session.query(AudioAnalysis)
            .filter_by(track_id=separated_track.id, stem_file_id=stem.id)
            .one()
        )

    drums = _row(StemKind.DRUMS)
    bass = _row(StemKind.BASS)
    vocals = _row(StemKind.VOCALS)
    other = _row(StemKind.OTHER)

    # Drums: kick_density is a real number, others' kick_density is None.
    assert drums.kick_density is not None
    assert drums.kick_density >= 0.0
    assert bass.kick_density is None

    # Bass + other: Camelot is populated.
    assert bass.dominant_pitch_camelot is not None
    assert bass.dominant_pitch_camelot.endswith("A")  # forced minor
    assert other.dominant_pitch_camelot is not None
    assert drums.dominant_pitch_camelot is None

    # Vocals: bool set.
    assert vocals.vocal_present is not None
    assert isinstance(vocals.vocal_present, bool)
    assert drums.vocal_present is None

    # Presence intervals are a JSON list of [start, end] pairs.
    intervals = json.loads(drums.presence_intervals)
    assert isinstance(intervals, list)
    for pair in intervals:
        assert len(pair) == 2
        assert pair[0] < pair[1]


def test_upsert_is_idempotent(session, separated_track, settings):
    stage = StemAnalysisStage()
    assert stage.process(session, separated_track, settings) is True

    # Second invocation: track is no longer SEPARATED, but calling the stage
    # directly should still upsert — and not double-insert.
    # Move it back to SEPARATED to simulate a re-run.
    separated_track.state = TrackState.SEPARATED.value
    session.commit()

    assert stage.process(session, separated_track, settings) is True

    stem_rows = (
        session.query(AudioAnalysis)
        .filter(
            AudioAnalysis.track_id == separated_track.id,
            AudioAnalysis.stem_file_id.isnot(None),
        )
        .all()
    )
    assert len(stem_rows) == 4


def test_missing_stem_file_marks_error(session, separated_track, settings, tmp_path):
    # Delete one of the on-disk stem files.
    stem = (
        session.query(StemFile)
        .filter_by(track_id=separated_track.id, kind=StemKind.BASS.value)
        .one()
    )
    Path(stem.path).unlink()

    stage = StemAnalysisStage()
    ok = stage.process(session, separated_track, settings)

    assert ok is False
    session.refresh(separated_track)
    assert separated_track.state == TrackState.ERROR.value
    assert separated_track.error_message is not None
    assert "analyze_stems" in separated_track.error_message


def test_state_transitions_on_success(session, separated_track, settings):
    assert separated_track.state == TrackState.SEPARATED.value

    stage = StemAnalysisStage()
    stage.process(session, separated_track, settings)

    session.refresh(separated_track)
    assert separated_track.state == TrackState.STEMS_ANALYZED.value
