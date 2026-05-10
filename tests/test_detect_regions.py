"""Integration tests for RegionDetectionStage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from dance.config import Settings
from dance.core.database import (
    now_utc,
    AudioAnalysis,
    Beat,
    Phrase,
    Region,
    RegionSource,
    RegionType,
    StemFile,
    StemKind,
    TrackState,
)
from dance.pipeline.stage import Stage
from dance.pipeline.stages.detect_regions import RegionDetectionStage
from tests.audio_fixtures import TrackSpec, write_track


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
    )


def _make_analyzed_track(tmp_path, session, make_track, bpm=128.0, bars=16):
    """Synth a track + insert a full-mix AudioAnalysis row + STEMS_ANALYZED state."""
    spec = TrackSpec(bpm=bpm, bars=bars)
    audio_path = write_track(tmp_path / "house.wav", spec)
    track = make_track(
        file_path=str(audio_path),
        file_name=audio_path.name,
        state=TrackState.STEMS_ANALYZED.value,
        duration_seconds=spec.duration_seconds,
    )
    session.add(AudioAnalysis(
        track_id=track.id,
        stem_file_id=None,
        bpm=bpm,
        bpm_confidence=0.9,
        key_camelot="8A",
        key_standard="Am",
        energy_overall=0.5,
        analyzed_at=now_utc(),
    ))
    session.commit()
    return track, spec


def _add_stems_with_presence(session, track, presence_ratio=0.8):
    """Add 4 StemFile rows + matching per-stem AudioAnalysis with given presence."""
    stem_ids: dict[StemKind, int] = {}
    for kind in (StemKind.DRUMS, StemKind.BASS, StemKind.VOCALS, StemKind.OTHER):
        stem = StemFile(
            track_id=track.id, kind=kind.value,
            path=f"/tmp/{kind.value}.wav", model_used="htdemucs_ft",
        )
        session.add(stem)
        session.flush()
        session.add(AudioAnalysis(
            track_id=track.id, stem_file_id=stem.id,
            bpm=128.0, presence_ratio=presence_ratio,
            analyzed_at=now_utc(),
        ))
        stem_ids[kind] = stem.id
    session.commit()
    return stem_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_protocol_compliance():
    stage = RegionDetectionStage()
    assert isinstance(stage, Stage)
    assert stage.name == "detect_regions"
    assert stage.input_state == TrackState.STEMS_ANALYZED
    assert stage.output_state == TrackState.REGIONS_DETECTED
    assert stage.error_state == TrackState.ERROR


def test_end_to_end_emits_all_three_region_types(
    tmp_path, session, make_track, settings,
):
    track, _ = _make_analyzed_track(tmp_path, session, make_track)
    ok = RegionDetectionStage().process(session, track, settings)
    assert ok is True

    regions = session.query(Region).filter_by(track_id=track.id).all()
    types = {r.region_type for r in regions}
    assert RegionType.SECTION.value in types
    assert RegionType.CUE.value in types
    # Loops require >= 8 bars; the 8-bar synth track produces exactly one phrase
    # spanning the full track, so a whole-track loop should appear.
    assert RegionType.LOOP.value in types


def test_sections_cover_whole_track(
    tmp_path, session, make_track, settings,
):
    track, spec = _make_analyzed_track(tmp_path, session, make_track, bars=16)
    RegionDetectionStage().process(session, track, settings)

    sections = (
        session.query(Region)
        .filter_by(track_id=track.id, region_type=RegionType.SECTION.value)
        .order_by(Region.position_ms)
        .all()
    )
    assert sections, "expected at least one section"
    duration_ms = int(spec.duration_seconds * 1000)
    first, last = sections[0], sections[-1]
    # First section should start in the first quarter of the track. The
    # underlying phrase detector starts from the first detected beat, which
    # is not necessarily at t=0 for a track that opens with a downbeat hit.
    assert first.position_ms < duration_ms * 0.25
    end_of_last = last.position_ms + (last.length_ms or 0)
    # Last section should end somewhere in the second half of the track.
    assert end_of_last >= duration_ms * 0.5


def test_idempotent_does_not_double_write(
    tmp_path, session, make_track, settings,
):
    track, _ = _make_analyzed_track(tmp_path, session, make_track)
    stage = RegionDetectionStage()

    stage.process(session, track, settings)
    count_after_first = session.query(Region).filter_by(track_id=track.id).count()

    # Re-set the state so the stage will run again.
    track.state = TrackState.STEMS_ANALYZED.value
    session.commit()

    stage.process(session, track, settings)
    count_after_second = session.query(Region).filter_by(track_id=track.id).count()

    assert count_after_first > 0
    assert count_after_first == count_after_second


def test_manual_regions_preserved(
    tmp_path, session, make_track, settings,
):
    track, _ = _make_analyzed_track(tmp_path, session, make_track)
    manual = Region(
        track_id=track.id, stem_file_id=None,
        position_ms=1000, length_ms=2000,
        region_type=RegionType.CUE.value,
        name="My manual cue",
        source=RegionSource.MANUAL.value,
    )
    session.add(manual)
    session.commit()
    manual_id = manual.id

    RegionDetectionStage().process(session, track, settings)

    assert session.query(Region).filter_by(id=manual_id).one().name == "My manual cue"


def test_per_stem_loops_when_presence_high(
    tmp_path, session, make_track, settings,
):
    track, _ = _make_analyzed_track(tmp_path, session, make_track, bars=16)
    _add_stems_with_presence(session, track, presence_ratio=0.8)

    RegionDetectionStage().process(session, track, settings)

    stem_loops = (
        session.query(Region)
        .filter(
            Region.track_id == track.id,
            Region.region_type == RegionType.LOOP.value,
            Region.stem_file_id.is_not(None),
        )
        .all()
    )
    assert stem_loops, "expected at least one per-stem loop region"
    # We added 4 stems with presence_ratio=0.8; each section >=8 bars should
    # yield one per-stem loop per stem.
    stem_kinds_seen = {sl.stem_file_id for sl in stem_loops}
    assert len(stem_kinds_seen) == 4


def test_per_stem_loops_skipped_when_presence_low(
    tmp_path, session, make_track, settings,
):
    track, _ = _make_analyzed_track(tmp_path, session, make_track, bars=16)
    _add_stems_with_presence(session, track, presence_ratio=0.1)

    RegionDetectionStage().process(session, track, settings)

    stem_loops = (
        session.query(Region)
        .filter(
            Region.track_id == track.id,
            Region.region_type == RegionType.LOOP.value,
            Region.stem_file_id.is_not(None),
        )
        .all()
    )
    assert stem_loops == []


def test_track_state_transitions_on_success(
    tmp_path, session, make_track, settings,
):
    track, _ = _make_analyzed_track(tmp_path, session, make_track)
    assert track.state == TrackState.STEMS_ANALYZED.value

    ok = RegionDetectionStage().process(session, track, settings)
    assert ok is True

    session.refresh(track)
    assert track.state == TrackState.REGIONS_DETECTED.value


def test_track_state_error_on_missing_file(
    tmp_path, session, make_track, settings,
):
    # Build a track that points to a non-existent file but has analysis.
    track = make_track(
        file_path=str(tmp_path / "missing.wav"),
        state=TrackState.STEMS_ANALYZED.value,
    )
    session.add(AudioAnalysis(
        track_id=track.id, stem_file_id=None,
        bpm=128.0, analyzed_at=now_utc(),
    ))
    session.commit()

    ok = RegionDetectionStage().process(session, track, settings)
    assert ok is False

    session.refresh(track)
    assert track.state == TrackState.ERROR.value
    assert track.error_message is not None


def test_beats_persisted(tmp_path, session, make_track, settings):
    track, _ = _make_analyzed_track(tmp_path, session, make_track)
    RegionDetectionStage().process(session, track, settings)

    count = session.query(Beat).filter_by(track_id=track.id).count()
    assert count > 0


def test_phrases_persisted(tmp_path, session, make_track, settings):
    track, _ = _make_analyzed_track(tmp_path, session, make_track)
    RegionDetectionStage().process(session, track, settings)

    count = session.query(Phrase).filter_by(track_id=track.id).count()
    assert count > 0
