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
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from dance import cli
from dance.api import create_app
from dance.api.app import API_PREFIX
from dance.config import Settings, get_settings
from dance.core.database import (
    AudioAnalysis,
    Beat,
    EdgeKind,
    Phrase,
    Region,
    RegionType,
    StemFile,
    StemKind,
    Track,
    TrackEmbedding,
    TrackState,
    _reset_engine_for_tests,
    get_session,
    get_session_factory,
    init_db,
    now_utc,
)
from dance.pipeline import dispatcher as dispatcher_mod
from dance.pipeline import preflight
from dance.pipeline.dispatcher import Dispatcher
from dance.pipeline.heal_ledger import HealLedger
from dance.recommender import GraphBuilder
from dance.spotify import csv_importer, download_pool
from tests.audio_fixtures import TrackSpec, write_track
from tests.test_acquisition import _FAKE_YT_DLP, FakeYtDlp, _encode, _entry, needs_ffmpeg
from tests.test_api import FakeAbletonBridge
from tests.test_dispatcher_lock import _HOLDER, _Child, _child_env, _release


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
    # Recommendation graph — post-collapse, only embedding-neighbor edges are
    # materialized (key/BPM/energy/structure are now scored live at query time).
    builder = GraphBuilder(e2e_session, e2e_settings)
    edge_counts = builder.build()
    # The builder completes and returns only the embedding-neighbor key. (Fake
    # embeddings use model "fake-clap"; settings.clap_model is the real name, so
    # the cosine matrix finds no matching rows — 0 edges is the correct outcome.)
    assert EdgeKind.EMBEDDING_NEIGHBOR.value in edge_counts, edge_counts
    assert set(edge_counts.keys()) == {EdgeKind.EMBEDDING_NEIGHBOR.value}

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


MOST_PASSES_ONE_SHORT_SONG_NEEDS = 6
OUT_OF_MEMORY = "MPS backend out of memory"
NOTHING_TO_RETRY = "This track has not failed, so there is nothing to retry."
UPLOAD = "RIGHTupload"
SONG_ON_SPOTIFY = {
    "spotify_id": "shallow-water",
    "title": "Shallow Water",
    "artist": "Elderbrook",
    "duration_ms": 60_000,
}


def _separate_fails_without_a_message(self, session, track, settings) -> bool:
    raise RuntimeError


def _separate_runs_out_of_memory(self, session, track, settings) -> bool:
    raise RuntimeError(OUT_OF_MEMORY)


def _heavy_models(separate):
    return (
        patch("dance.pipeline.stages.separate.StemSeparationStage.process", new=separate),
        patch("dance.pipeline.stages.embed.EmbeddingStage._ensure_model", new=_fake_ensure_model),
        patch("dance.pipeline.stages.embed.EmbeddingStage._encode", new=_fake_embed_encode),
    )


class PipelineView:
    def __init__(self, client: TestClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def rows(self, state: str | None = None) -> list[dict]:
        listed = self.client.get(
            f"{API_PREFIX}/pipeline/recent", params={"state": state} if state else {}
        )
        assert listed.status_code == 200, listed.text
        return listed.json()

    def retry(self, track_id: int):
        return self.client.post(f"{API_PREFIX}/pipeline/tracks/{track_id}/retry")

    def settled(self) -> bool:
        status = self.client.get(f"{API_PREFIX}/pipeline/status").json()
        return status["total"] == status["complete"] + status["errors"]

    def dance_process(self, separate) -> None:
        stand_in, ensure_model, encode = _heavy_models(separate)
        with stand_in, ensure_model, encode:
            for _ in range(MOST_PASSES_ONE_SHORT_SONG_NEEDS):
                ran = CliRunner().invoke(cli.main, ["process"])
                assert ran.exit_code == 0, ran.output
                if self.settled():
                    return
        raise AssertionError(f"the pipeline never settled: {self.rows()}")

    def one_track_at_a_time(self, separate) -> None:
        session = get_session(self.settings.db_url)
        stand_in, ensure_model, encode = _heavy_models(separate)
        with stand_in, ensure_model, encode:
            dispatcher = Dispatcher(self.settings, session)
            dispatcher.ingest()
            dispatcher.run()
        session.close()

    def left_mid_stage_by_a_hard_crash(self, track_id: int, stage_state: TrackState) -> None:
        with get_session_factory(self.settings.db_url)() as session:
            session.get(Track, track_id).state = stage_state.value
            session.commit()

    def newest_job_finished(self) -> dict:
        for _ in range(400):
            job = self.client.get(f"{API_PREFIX}/pipeline/jobs").json()[0]
            if job["status"] in ("done", "error"):
                return job
            time.sleep(0.05)
        raise AssertionError("the newest download job never finished")

    def stored(self, track_id: int) -> tuple[str, str]:
        with get_session_factory(self.settings.db_url)() as session:
            track = session.get(Track, track_id)
            return str(track.file_hash), str(track.file_path)


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(dispatcher_mod, "_IDLE_GRACE_S", 0.3)
    monkeypatch.setattr(dispatcher_mod, "_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    settings = get_settings()
    settings.ensure_directories()
    init_db(settings.db_url)
    download_pool._reset_download_pool_for_tests()
    app = create_app(
        settings=settings,
        bridge=FakeAbletonBridge(),
        session_factory=get_session_factory(settings.db_url),
    )
    with TestClient(app) as client:
        yield PipelineView(client, settings)
    download_pool._reset_download_pool_for_tests()
    _reset_engine_for_tests()


@pytest.fixture
def a_song_in_the_library(pipeline) -> Path:
    return write_track(
        Path(pipeline.settings.library_dir) / "night_drive.wav",
        TrackSpec(bpm=128.0, bars=4, seed=1),
    )


@pytest.fixture
def another_run(tmp_path):
    started: list[_Child] = []

    def hold_the_lock(data_dir: Path) -> _Child:
        holder = _Child(_HOLDER, [str(data_dir)], _child_env(tmp_path))
        started.append(holder)
        holder.expect("LOCKED")
        return holder

    yield hold_the_lock
    for holder in started:
        holder.kill()


@pytest.mark.parametrize("run", ["dance_process", "one_track_at_a_time"])
def test_a_stage_that_fails_without_a_message_still_leaves_a_reason(
    pipeline, a_song_in_the_library, run
):
    getattr(pipeline, run)(separate=_separate_fails_without_a_message)

    assert [row["error_message"] for row in pipeline.rows("error")] == ["separate: RuntimeError"]


def test_a_failed_track_goes_back_in_line_and_the_next_run_finishes_it(
    pipeline, a_song_in_the_library
):
    pipeline.dance_process(separate=_separate_runs_out_of_memory)
    (failed,) = pipeline.rows("error")
    assert failed["error_message"] == f"separate: {OUT_OF_MEMORY}"

    retried = pipeline.retry(failed["id"])

    assert retried.status_code == 200, retried.text
    assert (retried.json()["id"], retried.json()["state"], retried.json()["error_message"]) == (
        failed["id"],
        "pending",
        None,
    )
    assert pipeline.rows("error") == []
    assert [row["state"] for row in pipeline.rows()] == ["pending"]

    pipeline.dance_process(separate=_fake_separate_process)

    assert [(row["id"], row["state"]) for row in pipeline.rows()] == [(failed["id"], "complete")]


def test_retry_waits_for_a_run_that_is_in_progress(pipeline, a_song_in_the_library, another_run):
    pipeline.dance_process(separate=_separate_runs_out_of_memory)
    (failed,) = pipeline.rows("error")

    running = another_run(pipeline.settings.data_dir)
    refused = pipeline.retry(failed["id"])

    assert refused.status_code == 409, refused.text
    assert "already in progress" in refused.json()["detail"]
    assert f"pid {running.pid}" in refused.json()["detail"]
    assert pipeline.rows() == [failed]

    _release(running)
    assert pipeline.retry(failed["id"]).json()["state"] == "pending"


def test_retry_says_why_it_leaves_a_track_that_has_not_failed(pipeline, a_song_in_the_library):
    pipeline.dance_process(separate=_fake_separate_process)
    (finished,) = pipeline.rows()

    refused = pipeline.retry(finished["id"])

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == NOTHING_TO_RETRY
    assert pipeline.rows() == [finished]
    assert pipeline.retry(999_999).status_code == 404


def test_a_retried_track_starts_its_count_of_hard_crashes_again(pipeline, a_song_in_the_library):
    pipeline.dance_process(separate=_separate_runs_out_of_memory)
    (failed,) = pipeline.rows("error")
    HealLedger(pipeline.settings.data_dir).record_heal(failed["id"], TrackState.SEPARATING.value)

    assert pipeline.retry(failed["id"]).status_code == 200
    pipeline.left_mid_stage_by_a_hard_crash(failed["id"], TrackState.SEPARATING)
    pipeline.dance_process(separate=_fake_separate_process)

    assert [(row["state"], row["error_message"]) for row in pipeline.rows()] == [("complete", None)]


@needs_ffmpeg
def test_retry_downloads_a_track_again_when_its_download_never_finished(
    pipeline, another_run, monkeypatch, tmp_path
):
    upload = _entry(UPLOAD, "Elderbrook - Shallow Water", "Elderbrook", 60)
    youtube = FakeYtDlp({"Elderbrook Shallow Water": [upload]}, media={UPLOAD: None})
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    monkeypatch.setattr(subprocess, "run", youtube)
    asked = pipeline.client.post(f"{API_PREFIX}/pipeline/ingest/track", json=SONG_ON_SPOTIFY)
    track_id = asked.json()["track_id"]
    assert pipeline.newest_job_finished()["items"][0]["status"] == "fail"
    (failed,) = pipeline.rows("error")
    assert failed["id"] == track_id

    youtube.media[UPLOAD] = _encode(tmp_path / "upload.mp3", 60, "libmp3lame")
    another_run(pipeline.settings.data_dir)
    retried = pipeline.retry(track_id)

    assert retried.status_code == 200, retried.text
    assert (retried.json()["state"], retried.json()["error_message"]) == ("pending", None)
    assert pipeline.newest_job_finished()["items"][0]["status"] == "ok"
    assert youtube.downloads == [UPLOAD, UPLOAD]
    file_hash, file_path = pipeline.stored(track_id)
    assert not file_hash.startswith("pending:")
    assert Path(file_path).exists()
    assert [row["state"] for row in pipeline.rows()] == ["pending"]
