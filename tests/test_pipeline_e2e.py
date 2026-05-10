"""End-to-end pipeline integration test.

Drives every stage through the Dispatcher on a synthetic audio track. Demucs
and CLAP are mocked (the test would take minutes otherwise + downloads model
weights), but every OTHER stage runs for real:

    ingest (real file scan)
      → analyze (real librosa BPM/key/energy)
      → separate (MOCKED: writes a fake StemFile per kind)
      → analyze_stems (real per-stem analysis on the synthetic audio)
      → detect_regions (real region detection)
      → embed (MOCKED: writes fake embeddings)
      → graph builder (real)

The audit found the previous pipeline "never really ran." This test guards
against a repeat.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Beat,
    Phrase,
    Region,
    RegionType,
    StemFile,
    StemKind,
    Track,
    TrackEmbedding,
    TrackState,
    init_db,
    now_utc,
)
from dance.pipeline.dispatcher import Dispatcher
from dance.recommender import GraphBuilder
from tests.audio_fixtures import TrackSpec, write_track


# ---------------------------------------------------------------------------
# Fakes (Demucs + CLAP are the only stages we mock)
# ---------------------------------------------------------------------------


def _fake_separate_process(self, session, track, settings) -> bool:
    """Stand-in for Demucs that copies the input as each of 4 stems."""
    # Skip if already separated.
    if session.query(StemFile).filter_by(track_id=track.id).count() == 4:
        track.state = self.output_state.value
        session.commit()
        return True

    out_dir = settings.stems_dir / track.file_hash[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in StemKind:
        path = out_dir / f"{kind.value}.wav"
        shutil.copy(track.file_path, path)
        session.add(
            StemFile(
                track_id=track.id,
                kind=kind.value,
                path=str(path),
                model_used="fake-demucs",
            )
        )
    track.state = self.output_state.value
    session.commit()
    return True


def _fake_embed_encode(self, audio: np.ndarray) -> np.ndarray:
    """Stand-in for CLAP that returns a deterministic 512-d vector."""
    # Use first 512 samples (or pad) to produce a stable embedding per audio.
    flat = audio.flatten()
    if flat.size >= 512:
        v = flat[:512].astype(np.float32)
    else:
        v = np.pad(flat, (0, 512 - flat.size)).astype(np.float32)
    # Normalize so cosine sim behaves predictably.
    norm = np.linalg.norm(v) or 1.0
    return v / norm


def _fake_ensure_model(self, settings) -> None:  # noqa: ARG001
    self._model = object()  # sentinel; never read
    self._processor = object()
    self._model_name = "fake-clap"
    self._model_version = "test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_settings(tmp_path) -> Settings:
    return Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'e2e.db'}",
    )


@pytest.fixture
def e2e_session(e2e_settings):
    init_db(e2e_settings.db_url)
    from dance.core.database import get_session as _gs
    from dance.core.database import _reset_engine_for_tests

    s = _gs(e2e_settings.db_url)
    yield s
    s.close()
    _reset_engine_for_tests()


@pytest.fixture
def two_synth_tracks(e2e_settings) -> list[Path]:
    """Drop two short synthetic tracks at different BPMs into the library."""
    e2e_settings.library_dir.mkdir(parents=True, exist_ok=True)
    t128 = write_track(e2e_settings.library_dir / "house_128.wav", TrackSpec(bpm=128.0, bars=4, seed=1))
    t130 = write_track(e2e_settings.library_dir / "house_130.wav", TrackSpec(bpm=130.0, bars=4, seed=2))
    return [t128, t130]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_full_pipeline_on_synthetic_audio(e2e_settings, e2e_session, two_synth_tracks):
    """Run every stage end-to-end on two synthetic tracks; verify each table is populated."""
    dispatcher = Dispatcher(e2e_settings, e2e_session)

    # Mock the heavy ML stages — same protocol contract, fake guts.
    with patch(
        "dance.pipeline.stages.separate.StemSeparationStage.process",
        new=_fake_separate_process,
    ), patch(
        "dance.pipeline.stages.embed.EmbeddingStage._ensure_model",
        new=_fake_ensure_model,
    ), patch(
        "dance.pipeline.stages.embed.EmbeddingStage._encode",
        new=_fake_embed_encode,
    ):
        # Step 1: ingest the WAVs from disk
        ingest_result = dispatcher.ingest()
        assert ingest_result["new"] == 2

        # Step 2: run all stages
        result = dispatcher.run()

    # Sanity: both tracks reached COMPLETE
    tracks = e2e_session.query(Track).all()
    assert len(tracks) == 2
    for t in tracks:
        assert t.state == TrackState.COMPLETE.value, f"{t.title} stuck at {t.state}: {t.error_message}"

    # Stage tallies — each stage processed both tracks
    for stage_name in ("analyze", "analyze_stems", "detect_regions", "embed"):
        assert result[stage_name]["processed"] == 2, f"{stage_name} {result[stage_name]}"
        assert result[stage_name]["errors"] == 0, f"{stage_name} {result[stage_name]}"

    # Full-mix analysis row per track
    full_mix_count = (
        e2e_session.query(AudioAnalysis).filter(AudioAnalysis.stem_file_id.is_(None)).count()
    )
    assert full_mix_count == 2

    # 4 stems × 2 tracks = 8 stem rows + 8 stem analysis rows
    assert e2e_session.query(StemFile).count() == 8
    stem_analysis_count = (
        e2e_session.query(AudioAnalysis).filter(AudioAnalysis.stem_file_id.isnot(None)).count()
    )
    assert stem_analysis_count == 8

    # Regions: each track has at least one section, one cue, plus loops
    region_count = e2e_session.query(Region).count()
    assert region_count > 0
    section_count = e2e_session.query(Region).filter_by(region_type=RegionType.SECTION.value).count()
    assert section_count >= 2

    # Beats + phrases persisted
    assert e2e_session.query(Beat).count() > 0
    assert e2e_session.query(Phrase).count() > 0

    # Embeddings: 1 full mix + 4 stem = 5 per track = 10 total
    embedding_count = e2e_session.query(TrackEmbedding).count()
    assert embedding_count == 10

    # Recommendation graph
    builder = GraphBuilder(e2e_session, e2e_settings)
    edge_counts = builder.build()
    # At minimum we expect tempo_compat (128 and 130 are within 3 BPM).
    assert edge_counts["tempo_compat"] >= 2, edge_counts  # bidirectional

    # Idempotency: running the pipeline again does not produce errors
    # (every stage's input_state will be empty so this is essentially a no-op).
    with patch(
        "dance.pipeline.stages.separate.StemSeparationStage.process",
        new=_fake_separate_process,
    ), patch(
        "dance.pipeline.stages.embed.EmbeddingStage._ensure_model",
        new=_fake_ensure_model,
    ), patch(
        "dance.pipeline.stages.embed.EmbeddingStage._encode",
        new=_fake_embed_encode,
    ):
        second_run = dispatcher.run()
    # Nothing left to process — both tracks are COMPLETE
    for stage_name, counts in second_run.items():
        assert counts["processed"] == 0, f"{stage_name} unexpectedly processed tracks on second run"
        assert counts["errors"] == 0
