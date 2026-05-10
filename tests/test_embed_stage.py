"""Tests for EmbeddingStage.

CLAP weights are never loaded in pytest — both ``_ensure_model`` and ``_encode``
are monkey-patched. The synthetic audio fixtures only need to exist on disk;
the (mocked) encoder returns a deterministic zero vector regardless of input.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dance.config import Settings
from dance.core.database import (
    StemFile,
    StemKind,
    TrackEmbedding,
    TrackState,
)
from dance.core.serialization import decode_embedding
from dance.pipeline.stage import Stage
from dance.pipeline.stages.embed import EmbeddingStage
from tests.audio_fixtures import TrackSpec, write_track


EMBED_DIM = 512
EMBED_MODULE = "dance.pipeline.stages.embed"


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
def regions_track(tmp_path, session, make_track):
    """A REGIONS_DETECTED track with 4 StemFile rows pointing at real audio."""
    spec = TrackSpec(bpm=128.0, bars=4)
    audio_path = write_track(tmp_path / "mix.wav", spec)
    track = make_track(
        file_path=str(audio_path),
        file_name=audio_path.name,
        state=TrackState.REGIONS_DETECTED.value,
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
    session.commit()
    return track


@pytest.fixture
def stage_with_mock(monkeypatch):
    """An EmbeddingStage with model loading bypassed and ``_encode`` mocked.

    The fake encoder returns a deterministic 512-d float32 vector regardless of
    the input audio — fine because we only care about plumbing here.
    """
    stage = EmbeddingStage()

    def _fake_ensure(settings):
        stage._model = object()  # any truthy sentinel
        stage._processor = object()
        stage._device = "cpu"
        stage._model_name = settings.clap_model
        stage._model_version = "test-version"

    def _fake_encode(audio: np.ndarray) -> np.ndarray:
        # Embedding values seeded from the audio length so different inputs
        # produce different vectors (useful for at least one upsert check).
        rng = np.random.default_rng(int(audio.size) % 2**31)
        return rng.standard_normal(EMBED_DIM).astype(np.float32)

    monkeypatch.setattr(stage, "_ensure_model", _fake_ensure)
    monkeypatch.setattr(stage, "_encode", _fake_encode)
    return stage


# ---------------------------------------------------------------------------
# Protocol / attributes
# ---------------------------------------------------------------------------


def test_stage_protocol_compliance():
    assert isinstance(EmbeddingStage(), Stage)


def test_stage_attributes():
    stage = EmbeddingStage()
    assert stage.name == "embed"
    assert stage.input_state == TrackState.REGIONS_DETECTED
    assert stage.output_state == TrackState.COMPLETE
    assert stage.error_state == TrackState.ERROR


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_writes_full_mix_plus_stem_embeddings(
    session, regions_track, settings, stage_with_mock
):
    ok = stage_with_mock.process(session, regions_track, settings)
    assert ok is True

    rows = (
        session.query(TrackEmbedding)
        .filter_by(track_id=regions_track.id)
        .all()
    )
    # 1 full mix (stem_file_id IS NULL) + 4 stems = 5
    assert len(rows) == 5
    assert sum(1 for r in rows if r.stem_file_id is None) == 1
    assert sum(1 for r in rows if r.stem_file_id is not None) == 4

    for r in rows:
        assert r.dim == EMBED_DIM
        assert r.model == settings.clap_model
        assert r.embedding is not None and len(r.embedding) == EMBED_DIM * 4


def test_state_transitions_to_complete(
    session, regions_track, settings, stage_with_mock
):
    assert regions_track.state == TrackState.REGIONS_DETECTED.value
    stage_with_mock.process(session, regions_track, settings)

    session.refresh(regions_track)
    assert regions_track.state == TrackState.COMPLETE.value


def test_embedding_roundtrip(
    session, regions_track, settings, stage_with_mock
):
    stage_with_mock.process(session, regions_track, settings)

    row = (
        session.query(TrackEmbedding)
        .filter_by(track_id=regions_track.id, stem_file_id=None)
        .one()
    )
    decoded = decode_embedding(row.embedding, dim=row.dim)
    assert decoded.shape == (EMBED_DIM,)
    assert decoded.dtype == np.float32


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_upsert_does_not_double_insert(
    session, regions_track, settings, stage_with_mock
):
    assert stage_with_mock.process(session, regions_track, settings) is True

    # Re-run; same model + version => UPDATE, not INSERT.
    regions_track.state = TrackState.REGIONS_DETECTED.value
    session.commit()

    assert stage_with_mock.process(session, regions_track, settings) is True

    rows = (
        session.query(TrackEmbedding)
        .filter_by(track_id=regions_track.id)
        .all()
    )
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_track_file_marks_error(
    session, regions_track, settings, stage_with_mock
):
    Path(regions_track.file_path).unlink()

    ok = stage_with_mock.process(session, regions_track, settings)
    assert ok is False

    session.refresh(regions_track)
    assert regions_track.state == TrackState.ERROR.value
    assert regions_track.error_message is not None
    assert "embed" in regions_track.error_message


def test_missing_stem_file_marks_error(
    session, regions_track, settings, stage_with_mock
):
    stem = (
        session.query(StemFile)
        .filter_by(track_id=regions_track.id, kind=StemKind.BASS.value)
        .one()
    )
    Path(stem.path).unlink()

    ok = stage_with_mock.process(session, regions_track, settings)
    assert ok is False

    session.refresh(regions_track)
    assert regions_track.state == TrackState.ERROR.value
    assert "embed" in regions_track.error_message


def test_no_stems_still_writes_full_mix(
    tmp_path, session, make_track, settings, stage_with_mock
):
    """A track in REGIONS_DETECTED with no StemFile rows still gets a mix embedding."""
    spec = TrackSpec(bpm=128.0, bars=2)
    audio_path = write_track(tmp_path / "mix.wav", spec)
    track = make_track(
        file_path=str(audio_path),
        file_name=audio_path.name,
        state=TrackState.REGIONS_DETECTED.value,
    )
    session.commit()

    ok = stage_with_mock.process(session, track, settings)
    assert ok is True

    rows = session.query(TrackEmbedding).filter_by(track_id=track.id).all()
    assert len(rows) == 1
    assert rows[0].stem_file_id is None


# ---------------------------------------------------------------------------
# Lazy-import guard
# ---------------------------------------------------------------------------


def test_process_errors_when_clap_unavailable(
    session, regions_track, settings
):
    """If transformers/torch aren't importable, process must mark the track ERROR."""
    stage = EmbeddingStage()
    with patch(f"{EMBED_MODULE}._CLAP_OK", False):
        ok = stage.process(session, regions_track, settings)

    assert ok is False
    session.refresh(regions_track)
    assert regions_track.state == TrackState.ERROR.value
