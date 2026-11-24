from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from fastapi.testclient import TestClient

from dance import cli
from dance.api import create_app
from dance.api.app import API_PREFIX
from dance.config import get_settings
from dance.core.database import (
    StemFile,
    StemKind,
    Track,
    TrackState,
    _reset_engine_for_tests,
    get_session,
    get_session_factory,
    init_db,
)
from dance.pipeline import dispatcher as dispatcher_mod
from dance.pipeline import preflight
from dance.pipeline.dispatcher import Dispatcher
from tests.audio_fixtures import TrackSpec, bass_layer, hat_layer, kick_layer
from tests.test_api import FakeAbletonBridge
from tests.test_pipeline_e2e import _fake_embed_encode, _fake_ensure_model

BPM = 128.0
BARS_PER_BLOCK = 8
BLOCK_LEVELS = [0.9, 0.7, 1.0, 0.3, 0.95, 0.6]
BLOCK_MS = int(BARS_PER_BLOCK * 4 * 60_000 / BPM)
EXPECTED_PAD_STARTS_MS = [0, BLOCK_MS, 2 * BLOCK_MS, 3 * BLOCK_MS, 4 * BLOCK_MS]
EXPECTED_LABELS = ["intro", "buildup", "drop", "breakdown", "drop"]
BAR_MS = int(4 * 60_000 / BPM)
ROLES = ["drums", "bass", "vocals", "other", "mix"]
API = API_PREFIX


def _song_with_six_sections(bpm: float = BPM, seed: int = 7) -> tuple[np.ndarray, TrackSpec]:
    spec = TrackSpec(bpm=bpm, bars=BARS_PER_BLOCK * len(BLOCK_LEVELS), seed=seed)
    seconds = np.arange(spec.total_samples) / spec.sample_rate
    sustained_tone = (0.2 * np.sin(2 * np.pi * 330.0 * seconds)).astype(np.float32)
    layers = kick_layer(spec) + bass_layer(spec) + hat_layer(spec) + sustained_tone
    block_samples = int(BARS_PER_BLOCK * spec.bar_seconds * spec.sample_rate)
    gain = np.ones(spec.total_samples, dtype=np.float32)
    for block, level in enumerate(BLOCK_LEVELS):
        gain[block * block_samples : (block + 1) * block_samples] = level
    mix = layers * gain
    return (mix / np.max(np.abs(mix)) * 0.9).astype(np.float32), spec


def _separate_with_silent_vocal_intro(self, session, track, settings) -> bool:
    if session.query(StemFile).filter_by(track_id=track.id).count() == 4:
        track.state = self.output_state.value
        session.commit()
        return True
    audio, sample_rate = sf.read(track.file_path, dtype="float32")
    intro_samples = int(BLOCK_MS / 1000 * sample_rate)
    out_dir = settings.stems_dir / track.file_hash[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in StemKind:
        stem_audio = audio.copy()
        if kind is StemKind.VOCALS:
            stem_audio[:intro_samples] = 0.0
        path = out_dir / f"{kind.value}.wav"
        sf.write(str(path), stem_audio, sample_rate, subtype="PCM_16")
        session.add(
            StemFile(track_id=track.id, kind=kind.value, path=str(path), model_used="fake-demucs")
        )
    track.state = self.output_state.value
    session.commit()
    return True


def _heavy_models_faked():
    return (
        patch(
            "dance.pipeline.stages.separate.StemSeparationStage.process",
            new=_separate_with_silent_vocal_intro,
        ),
        patch("dance.pipeline.stages.embed.EmbeddingStage._ensure_model", new=_fake_ensure_model),
        patch("dance.pipeline.stages.embed.EmbeddingStage._encode", new=_fake_embed_encode),
    )


@pytest.fixture
def library_with_one_song(monkeypatch):
    monkeypatch.setattr(dispatcher_mod, "_IDLE_GRACE_S", 0.3)
    monkeypatch.setattr(dispatcher_mod, "_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    settings = get_settings()
    settings.ensure_directories()
    init_db(settings.db_url)
    audio, spec = _song_with_six_sections()
    sf.write(
        str(Path(settings.library_dir) / "six_sections.wav"),
        audio,
        spec.sample_rate,
        subtype="PCM_16",
    )

    session = get_session(settings.db_url)
    separate, ensure_model, encode = _heavy_models_faked()
    with separate, ensure_model, encode:
        dispatcher = Dispatcher(settings, session)
        assert dispatcher.ingest()["new"] == 1
        dispatcher.run()
    track = session.query(Track).one()
    assert track.state == TrackState.COMPLETE.value, track.error_message
    track_id = track.id
    session.close()

    app = create_app(
        settings=settings,
        bridge=FakeAbletonBridge(),
        session_factory=get_session_factory(settings.db_url),
    )
    with TestClient(app) as client:
        yield client, track_id
    _reset_engine_for_tests()


def _pads_by_role(client: TestClient, track_id: int) -> dict[str, list[dict]]:
    response = client.get(f"{API}/tracks/{track_id}/pads")
    assert response.status_code == 200, response.text
    by_role: dict[str, list[dict]] = {role: [] for role in ROLES}
    for pad in response.json():
        by_role[pad["role"]].append(pad)
    return {role: sorted(pads, key=lambda p: p["slot"]) for role, pads in by_role.items()}


def _section_starts_by_label(client: TestClient, track_id: int) -> dict[str, set[int]]:
    response = client.get(f"{API}/tracks/{track_id}/regions", params={"region_type": "section"})
    assert response.status_code == 200, response.text
    starts: dict[str, set[int]] = {}
    for section in response.json():
        starts.setdefault(section["section_label"], set()).add(section["position_ms"])
    return starts


def _rerun_region_detection() -> None:
    separate, ensure_model, encode = _heavy_models_faked()
    with separate, ensure_model, encode:
        result = CliRunner().invoke(
            cli.main, ["reprocess", "--from", "detect_regions", "--all", "--skip-embeddings"]
        )
    assert result.exit_code == 0, result.output


def test_every_role_gets_the_same_five_pads_in_song_order(library_with_one_song):
    client, track_id = library_with_one_song
    pads = _pads_by_role(client, track_id)
    section_starts = _section_starts_by_label(client, track_id)

    for role in ROLES:
        assert [p["slot"] for p in pads[role]] == [1, 2, 3, 4, 5], role
        assert [p["section_label"] for p in pads[role]] == EXPECTED_LABELS, role
        assert [p["length_bars"] for p in pads[role]] == [8, 8, 8, 8, 8], role
        assert all(p["source"] == "auto" for p in pads[role]), role
        for pad, expected_ms in zip(pads[role], EXPECTED_PAD_STARTS_MS):
            assert abs(pad["position_ms"] - expected_ms) <= BAR_MS, (role, pad)
            assert pad["position_ms"] in section_starts[pad["section_label"]], (role, pad)

    assert [p["position_ms"] for p in pads["vocals"]] == [p["position_ms"] for p in pads["mix"]]


def test_a_pad_whose_stem_is_silent_there_is_flagged(library_with_one_song):
    client, track_id = library_with_one_song
    pads = _pads_by_role(client, track_id)

    assert [p["silent"] for p in pads["vocals"]] == [True, False, False, False, False]
    assert [p["silent"] for p in pads["drums"]] == [False] * 5
    assert [p["silent"] for p in pads["mix"]] == [False] * 5


def test_a_pad_moved_by_hand_survives_a_rerun_and_reset_restores_the_pick(library_with_one_song):
    client, track_id = library_with_one_song
    before = _pads_by_role(client, track_id)
    outro_ms = 5 * BLOCK_MS

    moved = client.put(
        f"{API}/tracks/{track_id}/pads/vocals/1", json={"position_ms": outro_ms + 40}
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["source"] == "manual"
    assert moved.json()["section_label"] == "outro"
    assert abs(moved.json()["position_ms"] - outro_ms) <= BAR_MS

    _rerun_region_detection()

    after = _pads_by_role(client, track_id)
    vocal_pad_one = after["vocals"][0]
    assert vocal_pad_one["source"] == "manual"
    assert vocal_pad_one["position_ms"] == moved.json()["position_ms"]
    assert vocal_pad_one["silent"] is False
    assert [p["slot"] for p in after["vocals"]] == [1, 2, 3, 4, 5]
    assert [(p["slot"], p["position_ms"], p["source"]) for p in after["drums"]] == [
        (p["slot"], p["position_ms"], p["source"]) for p in before["drums"]
    ]

    reset = client.delete(f"{API}/tracks/{track_id}/pads/vocals/1")
    assert reset.status_code == 200, reset.text
    restored = _pads_by_role(client, track_id)["vocals"][0]
    assert restored["source"] == "auto"
    assert restored["position_ms"] == before["vocals"][0]["position_ms"]
    assert restored["silent"] is True


def test_moving_a_pad_for_the_whole_song_moves_every_role(library_with_one_song):
    client, track_id = library_with_one_song
    outro_ms = 5 * BLOCK_MS

    moved = client.put(f"{API}/tracks/{track_id}/pads/all/5", json={"position_ms": outro_ms})
    assert moved.status_code == 200, moved.text

    pads = _pads_by_role(client, track_id)
    for role in ROLES:
        assert pads[role][4]["source"] == "manual", role
        assert pads[role][4]["section_label"] == "outro", role
        assert abs(pads[role][4]["position_ms"] - outro_ms) <= BAR_MS, role


def test_pad_routes_reject_what_does_not_exist(library_with_one_song):
    client, track_id = library_with_one_song

    assert client.get(f"{API}/tracks/999999/pads").status_code == 404
    assert (
        client.put(f"{API}/tracks/{track_id}/pads/kazoo/1", json={"position_ms": 0}).status_code
        == 404
    )
    assert (
        client.put(f"{API}/tracks/{track_id}/pads/drums/6", json={"position_ms": 0}).status_code
        == 422
    )
    assert (
        client.put(
            f"{API}/tracks/{track_id}/pads/drums/1", json={"position_ms": 10_000_000}
        ).status_code
        == 422
    )


def test_exported_live_set_gets_no_locators_for_pads(library_with_one_song):
    client, track_id = library_with_one_song

    exported = client.post(f"{API}/tracks/{track_id}/als", json={})
    assert exported.status_code == 200, exported.text

    live_set_xml = gzip.decompress(Path(exported.json()["out_path"]).read_bytes()).decode()
    assert "<Locator " in live_set_xml
    assert 'Value="Pad ' not in live_set_xml
