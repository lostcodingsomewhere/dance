"""Tests for the FastAPI backend.

These tests use a fake bridge so no UDP sockets are opened and AbletonOSC
doesn't need to be running.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from dance.api import create_app
from dance.config import get_settings
from dance.core import database as db
from dance.core.database import (
    AudioAnalysis,
    DjSession,
    EdgeKind,
    Region,
    RegionSource,
    RegionType,
    SessionPlay,
    StemFile,
    Tag,
    TagKind,
    TagSource,
    Track,
    TrackEdge,
    TrackTag,
    now_utc,
)
from dance.osc.bridge import AbletonState


# ---------------------------------------------------------------------------
# Fake bridge — records calls, exposes a swappable state
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def play(self) -> None:
        self.calls.append(("play", ()))

    def stop(self) -> None:
        self.calls.append(("stop", ()))

    def set_tempo(self, bpm: float) -> None:
        self.calls.append(("set_tempo", (bpm,)))

    def fire_clip(self, track: int, scene: int) -> None:
        self.calls.append(("fire_clip", (track, scene)))

    def set_track_volume(self, track: int, volume: float) -> None:
        self.calls.append(("set_track_volume", (track, volume)))


class FakeAbletonBridge:
    """Stand-in for AbletonBridge — no sockets, no threads."""

    def __init__(self) -> None:
        self.client = _FakeClient()
        self.state = AbletonState(tempo=120.0, is_playing=False, beat=0.0)
        self._subscribers: list[Callable[[AbletonState], None]] = []
        self.started = False
        self.stopped = False
        # Recorded push_track_to_live invocations.
        self.push_calls: list[dict[str, Any]] = []
        # Override-able return; default mimics a happy 5-stem push.
        self.push_return: dict[str, Any] | None = None
        self.push_raises: Exception | None = None
        # Recorded preview invocations.
        self.preview_calls: list[dict[str, Any]] = []
        self.preview_raises: Exception | None = None
        self.stop_preview_calls: int = 0
        # Cell-level deck state — overridable by individual tests.
        self.deck_state_return: dict[str, Any] = {
            "columns": None,
            "cells": [],
        }

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def subscribe(self, listener: Callable[[AbletonState], None]) -> None:
        self._subscribers.append(listener)

    def emit_state(self, **changes: Any) -> None:
        for k, v in changes.items():
            setattr(self.state, k, v)
        for sub in list(self._subscribers):
            sub(self.state)

    def push_track_to_live(
        self, track, stems, *, include_stems: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        self.push_calls.append(
            {
                "track_id": int(track.id),
                "stem_count": len(stems),
                "include_stems": include_stems,
                "kinds": kwargs.get("kinds"),
                "scene_index": kwargs.get("scene_index"),
            }
        )
        if self.push_raises is not None:
            raise self.push_raises
        if self.push_return is not None:
            return self.push_return
        # Default: mix + (drums/bass/vocals/other when include_stems).
        indices: dict[str, int] = {"mix": 0}
        if include_stems:
            for i, kind in enumerate(("drums", "bass", "vocals", "other"), start=1):
                # Only include stems we were actually handed (let callers control this).
                if any(str(s.kind).lower() == kind for s in stems):
                    indices[kind] = i
        return {"scene_index": 0, "track_indices": indices, "warnings": []}

    def preview_audio(
        self, audio_path: str, *, label: str | None = None
    ) -> dict[str, Any]:
        self.preview_calls.append({"audio_path": audio_path, "label": label})
        if self.preview_raises is not None:
            raise self.preview_raises
        return {
            "ok": True,
            "cue_track_idx": 5,
            "slot": 0,
            "audio_path": audio_path,
            "label": label,
            "warnings": [],
        }

    def stop_preview(self) -> dict[str, Any]:
        self.stop_preview_calls += 1
        return {"ok": True, "cleared": True}

    def get_deck_state(self) -> dict[str, Any]:
        return self.deck_state_return


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_bridge() -> FakeAbletonBridge:
    return FakeAbletonBridge()


@pytest.fixture
def app(
    session_factory: sessionmaker,
    fake_bridge: FakeAbletonBridge,
    tmp_path,
):
    """Build a fresh app with test-isolated paths.

    ``data_dir`` is rooted at ``tmp_path`` so the JobRegistry's
    ``jobs.json``, watch-mode flag, etc. don't write to the real
    ``~/.dance/`` directory and pollute production. Library, stems and
    als output dirs are also tmp so endpoints that touch the filesystem
    leave nothing behind."""
    from dance.config import Settings

    test_settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return create_app(
        settings=test_settings,
        bridge=fake_bridge,
        session_factory=session_factory,
    )


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------


def _add_fullmix_analysis(
    session,
    track: Track,
    *,
    bpm: float = 124.0,
    key: str = "8A",
    floor_energy: int = 6,
    energy_overall: float = 0.5,
) -> AudioAnalysis:
    a = AudioAnalysis(
        track_id=track.id,
        stem_file_id=None,
        bpm=bpm,
        key_camelot=key,
        floor_energy=floor_energy,
        energy_overall=energy_overall,
        analyzed_at=now_utc(),
    )
    session.add(a)
    session.flush()
    return a


def _add_tag(session, track: Track, value: str, kind: TagKind = TagKind.MOOD) -> Tag:
    tag = Tag(kind=kind.value, value=value, normalized_value=value.lower())
    session.add(tag)
    session.flush()
    session.add(TrackTag(track_id=track.id, tag_id=tag.id, source=TagSource.LLM.value))
    session.flush()
    return tag


# ---------------------------------------------------------------------------
# /tracks
# ---------------------------------------------------------------------------


def test_list_tracks_empty(client: TestClient) -> None:
    r = client.get("/api/v1/tracks")
    assert r.status_code == 200
    assert r.json() == []


def test_list_tracks_with_analysis(
    client: TestClient, session, make_track
) -> None:
    t = make_track(title="Hello", artist="World")
    _add_fullmix_analysis(session, t, bpm=128.0, key="9A", floor_energy=7)
    _add_tag(session, t, "uplifting")
    session.commit()

    r = client.get("/api/v1/tracks")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    row = body[0]
    assert row["title"] == "Hello"
    assert row["analysis"]["bpm"] == 128.0
    assert row["analysis"]["key_camelot"] == "9A"
    assert row["tags"] == ["uplifting"]


def test_list_tracks_filters(client: TestClient, session, make_track) -> None:
    fast = make_track(title="fast", state="complete")
    slow = make_track(title="slow", state="complete")
    other = make_track(title="other", state="pending")
    _add_fullmix_analysis(session, fast, bpm=130.0, key="8A", floor_energy=8)
    _add_fullmix_analysis(session, slow, bpm=90.0, key="3B", floor_energy=3)
    _add_fullmix_analysis(session, other, bpm=120.0, key="8A", floor_energy=5)
    session.commit()

    r = client.get("/api/v1/tracks", params={"bpm_min": 100, "bpm_max": 140})
    titles = {row["title"] for row in r.json()}
    assert titles == {"fast", "other"}

    r = client.get("/api/v1/tracks", params={"key": "8A"})
    assert {row["title"] for row in r.json()} == {"fast", "other"}

    r = client.get("/api/v1/tracks", params={"energy": 3})
    assert {row["title"] for row in r.json()} == {"slow"}

    r = client.get("/api/v1/tracks", params={"state": "pending"})
    assert {row["title"] for row in r.json()} == {"other"}


def test_get_track_404(client: TestClient) -> None:
    r = client.get("/api/v1/tracks/999")
    assert r.status_code == 404


def test_get_track_regions_filtered(
    client: TestClient, session, make_track
) -> None:
    t1 = make_track(title="t1")
    t2 = make_track(title="t2")
    session.add_all(
        [
            Region(
                track_id=t1.id,
                position_ms=0,
                region_type=RegionType.CUE.value,
                source=RegionSource.AUTO.value,
            ),
            Region(
                track_id=t1.id,
                position_ms=5000,
                length_ms=2000,
                region_type=RegionType.LOOP.value,
                source=RegionSource.AUTO.value,
            ),
            Region(
                track_id=t2.id,
                position_ms=1000,
                region_type=RegionType.CUE.value,
                source=RegionSource.AUTO.value,
            ),
        ]
    )
    session.commit()

    r = client.get(f"/api/v1/tracks/{t1.id}/regions")
    rows = r.json()
    assert len(rows) == 2
    assert all(row["region_type"] in ("cue", "loop") for row in rows)
    assert rows[0]["position_ms"] <= rows[1]["position_ms"]

    r = client.get(f"/api/v1/tracks/{t1.id}/regions", params={"region_type": "loop"})
    assert len(r.json()) == 1


def test_get_track_stems(client: TestClient, session, make_track) -> None:
    t = make_track(title="t")
    stem = StemFile(track_id=t.id, kind="drums", path="/tmp/drums.wav")
    session.add(stem)
    session.flush()
    session.add(
        AudioAnalysis(
            track_id=t.id,
            stem_file_id=stem.id,
            bpm=128.0,
            energy_overall=0.6,
            floor_energy=7,
            presence_ratio=0.8,
            vocal_present=False,
            kick_density=0.9,
            analyzed_at=now_utc(),
        )
    )
    session.commit()

    r = client.get(f"/api/v1/tracks/{t.id}/stems")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    row = body[0]
    assert row["kind"] == "drums"
    assert row["analysis"]["bpm"] == 128.0
    assert row["analysis"]["kick_density"] == 0.9


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------


def test_recommend_single_seed(client: TestClient, session, make_track) -> None:
    seed = make_track(title="seed")
    near = make_track(title="near")
    far = make_track(title="far")
    session.add_all(
        [
            TrackEdge(
                from_track_id=seed.id,
                to_track_id=near.id,
                kind=EdgeKind.HARMONIC_COMPAT.value,
                weight=0.9,
                computed_at=now_utc(),
            ),
            TrackEdge(
                from_track_id=seed.id,
                to_track_id=far.id,
                kind=EdgeKind.HARMONIC_COMPAT.value,
                weight=0.3,
                computed_at=now_utc(),
            ),
        ]
    )
    session.commit()

    r = client.post("/api/v1/recommend", json={"seeds": [seed.id], "k": 5})
    assert r.status_code == 200
    rows = r.json()
    assert [row["track_id"] for row in rows] == [near.id, far.id]
    assert rows[0]["score"] > rows[1]["score"]


def test_recommend_excludes_seeds_and_exclude(
    client: TestClient, session, make_track
) -> None:
    seed = make_track(title="seed")
    other = make_track(title="other")
    skip = make_track(title="skip")
    session.add_all(
        [
            # seed -> other
            TrackEdge(
                from_track_id=seed.id,
                to_track_id=other.id,
                kind=EdgeKind.TAG_OVERLAP.value,
                weight=0.5,
                computed_at=now_utc(),
            ),
            # seed -> skip
            TrackEdge(
                from_track_id=seed.id,
                to_track_id=skip.id,
                kind=EdgeKind.TAG_OVERLAP.value,
                weight=0.5,
                computed_at=now_utc(),
            ),
            # seed -> seed-like self should never appear (CHECK constraint;
            # we simulate by using a separate "fake-self" edge from seed to seed via different edge — not allowed.)
        ]
    )
    session.commit()

    r = client.post(
        "/api/v1/recommend",
        json={"seeds": [seed.id], "k": 10, "exclude": [skip.id]},
    )
    rows = r.json()
    ids = {row["track_id"] for row in rows}
    assert seed.id not in ids
    assert skip.id not in ids
    assert ids == {other.id}


# ---------------------------------------------------------------------------
# /sessions
# ---------------------------------------------------------------------------


def test_create_and_get_current_session(client: TestClient) -> None:
    r = client.post("/api/v1/sessions", json={"name": "Practice 1", "notes": "n"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Practice 1"
    assert body["ended_at"] is None
    assert body["plays"] == []
    sid = body["id"]

    r = client.get("/api/v1/sessions/current")
    assert r.status_code == 200
    assert r.json()["id"] == sid


def test_session_plays_auto_position(
    client: TestClient, session, make_track
) -> None:
    t1 = make_track(title="t1")
    t2 = make_track(title="t2")
    _add_fullmix_analysis(session, t1, floor_energy=4)
    session.commit()

    sid = client.post("/api/v1/sessions", json={}).json()["id"]

    r1 = client.post(
        f"/api/v1/sessions/{sid}/plays",
        json={"track_id": t1.id, "transition_type": "blend"},
    )
    assert r1.status_code == 200
    plays = r1.json()["plays"]
    assert len(plays) == 1
    assert plays[0]["position_in_set"] == 1
    assert plays[0]["energy_at_play"] == 4
    assert plays[0]["title"] == "t1"

    r2 = client.post(f"/api/v1/sessions/{sid}/plays", json={"track_id": t2.id})
    plays = r2.json()["plays"]
    assert [p["position_in_set"] for p in plays] == [1, 2]


def test_end_session_closes_current(client: TestClient) -> None:
    sid = client.post("/api/v1/sessions", json={}).json()["id"]
    assert client.get("/api/v1/sessions/current").status_code == 200

    r = client.post(f"/api/v1/sessions/{sid}/end")
    assert r.status_code == 200
    assert r.json()["ended_at"] is not None

    assert client.get("/api/v1/sessions/current").status_code == 404


# ---------------------------------------------------------------------------
# /ableton
# ---------------------------------------------------------------------------


def test_ableton_endpoints_call_client(
    client: TestClient, fake_bridge: FakeAbletonBridge
) -> None:
    assert client.post("/api/v1/ableton/play").json() == {"ok": True}
    assert client.post("/api/v1/ableton/stop").json() == {"ok": True}
    assert client.post("/api/v1/ableton/tempo", json={"bpm": 124.0}).json() == {"ok": True}
    assert client.post("/api/v1/ableton/fire", json={"track": 2, "scene": 1}).json() == {"ok": True}
    assert client.post(
        "/api/v1/ableton/volume", json={"track": 3, "volume": 0.8}
    ).json() == {"ok": True}

    names = [call[0] for call in fake_bridge.client.calls]
    assert names == ["play", "stop", "set_tempo", "fire_clip", "set_track_volume"]
    assert fake_bridge.client.calls[2] == ("set_tempo", (124.0,))
    assert fake_bridge.client.calls[3] == ("fire_clip", (2, 1))
    assert fake_bridge.client.calls[4] == ("set_track_volume", (3, 0.8))


def test_load_track_404_when_missing(client: TestClient) -> None:
    r = client.post(
        "/api/v1/ableton/load-track",
        json={"track_id": 999_999, "include_stems": True},
    )
    assert r.status_code == 404


def test_load_track_with_stems_creates_five_tracks(
    client: TestClient, fake_bridge: FakeAbletonBridge, session, make_track
) -> None:
    t = make_track(title="Anthem")
    # Attach one stem per kind so the bridge's default return creates 5 indices.
    for kind in ("drums", "bass", "vocals", "other"):
        session.add(StemFile(track_id=t.id, kind=kind, path=f"/tmp/{kind}.wav"))
    session.commit()

    r = client.post(
        "/api/v1/ableton/load-track",
        json={"track_id": t.id, "include_stems": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["scene_index"] == 0
    # Five logical tracks: mix + 4 stems.
    assert set(body["track_indices"].keys()) == {
        "mix",
        "drums",
        "bass",
        "vocals",
        "other",
    }
    # Bridge actually invoked with the right track.
    assert fake_bridge.push_calls[-1] == {
        "track_id": t.id,
        "stem_count": 4,
        "include_stems": True,
        "kinds": None,
        "scene_index": None,
    }


def test_load_track_without_stems_creates_one_track(
    client: TestClient, fake_bridge: FakeAbletonBridge, session, make_track
) -> None:
    t = make_track(title="Solo")
    # Add a stem on disk — but we ask for include_stems=False so it's ignored.
    session.add(StemFile(track_id=t.id, kind="drums", path="/tmp/d.wav"))
    session.commit()

    r = client.post(
        "/api/v1/ableton/load-track",
        json={"track_id": t.id, "include_stems": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["track_indices"] == {"mix": 0}
    # The bridge was passed an empty stems list.
    assert fake_bridge.push_calls[-1]["stem_count"] == 0
    assert fake_bridge.push_calls[-1]["include_stems"] is False


def test_load_track_returns_warnings_for_missing_files(
    client: TestClient, fake_bridge: FakeAbletonBridge, session, make_track
) -> None:
    t = make_track(title="Broken")
    session.commit()
    fake_bridge.push_return = {
        "scene_index": 0,
        "track_indices": {"mix": 0},
        "warnings": ["Full-mix file missing on disk: '/tmp/missing.wav'"],
    }

    r = client.post(
        "/api/v1/ableton/load-track",
        json={"track_id": t.id, "include_stems": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["warnings"] and "missing" in body["warnings"][0]


def test_load_track_503_when_osc_unreachable(
    client: TestClient, fake_bridge: FakeAbletonBridge, session, make_track
) -> None:
    t = make_track(title="Offline")
    session.commit()
    fake_bridge.push_raises = OSError("connection refused")

    r = client.post(
        "/api/v1/ableton/load-track",
        json={"track_id": t.id, "include_stems": False},
    )
    assert r.status_code == 503


def test_ableton_state(client: TestClient, fake_bridge: FakeAbletonBridge) -> None:
    fake_bridge.state.tempo = 126.0
    fake_bridge.state.is_playing = True
    fake_bridge.state.playing_clips[1] = 4

    r = client.get("/api/v1/ableton/state")
    body = r.json()
    assert body["tempo"] == 126.0
    assert body["is_playing"] is True
    # JSON object keys come back as strings.
    assert body["playing_clips"]["1"] == 4


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def test_websocket_initial_and_broadcast(
    client: TestClient, fake_bridge: FakeAbletonBridge
) -> None:
    fake_bridge.state.tempo = 120.0

    with client.websocket_connect("/ws") as ws:
        initial = ws.receive_json()
        assert initial["tempo"] == 120.0

        # Trigger a state change from a non-asyncio thread (simulates the
        # OSC listener thread).
        def fire() -> None:
            time.sleep(0.05)
            fake_bridge.emit_state(tempo=130.0, is_playing=True)

        threading.Thread(target=fire, daemon=True).start()

        update = ws.receive_json()
        assert update["tempo"] == 130.0
        assert update["is_playing"] is True


# ---------------------------------------------------------------------------
# Text recommendation endpoint (CLAP audio↔text)
# ---------------------------------------------------------------------------


def _stub_text_encoder_app(app, embedding_vec):
    """Replace the CLAP-loading hook with a stub returning a fixed vector."""
    import numpy as np

    class _Stub:
        def encode_text(self, _query: str):
            return np.asarray(embedding_vec, dtype=np.float32)

    app.state.embedding_stage = _Stub()


def test_recommend_by_text_endpoint(client: TestClient, app, fake_bridge):
    """POST /recommend/text returns tracks ordered by CLAP cosine similarity."""
    import numpy as np

    from dance.core.serialization import encode_embedding

    # Use the same settings the app does so the model_name filter matches.
    settings = get_settings()
    model = settings.clap_model

    session = app.state.session_factory()
    try:
        near = Track(
            file_hash="1" * 64, file_path="/a", file_name="a.wav",
            file_size_bytes=1, title="near", state="complete",
            created_at=now_utc(), updated_at=now_utc(),
        )
        far = Track(
            file_hash="2" * 64, file_path="/b", file_name="b.wav",
            file_size_bytes=1, title="far", state="complete",
            created_at=now_utc(), updated_at=now_utc(),
        )
        session.add_all([near, far])
        session.flush()

        # `near` aligned to [1,0,0], `far` aligned to [0,1,0].
        from dance.core.database import TrackEmbedding

        session.add(
            TrackEmbedding(
                track_id=near.id, stem_file_id=None,
                model=model, model_version=None, dim=3,
                embedding=encode_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                created_at=now_utc(),
            )
        )
        session.add(
            TrackEmbedding(
                track_id=far.id, stem_file_id=None,
                model=model, model_version=None, dim=3,
                embedding=encode_embedding(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                created_at=now_utc(),
            )
        )
        session.commit()
    finally:
        session.close()

    _stub_text_encoder_app(app, [1.0, 0.0, 0.0])

    r = client.post(
        "/api/v1/recommend/text",
        json={"query": "punchy techy with vocals", "k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert [item["title"] for item in body] == ["near", "far"]
    assert body[0]["score"] > body[1]["score"]
    assert body[0]["reasons"][0]["kind"] == "text_query"


def test_recommend_by_text_empty_query_rejected(client, app, fake_bridge):
    _stub_text_encoder_app(app, [1.0, 0.0])
    r = client.post("/api/v1/recommend/text", json={"query": "   ", "k": 5})
    assert r.status_code == 400


def test_recommend_by_text_exclude(client, app, fake_bridge):
    import numpy as np

    from dance.core.database import TrackEmbedding
    from dance.core.serialization import encode_embedding

    settings = get_settings()
    session = app.state.session_factory()
    try:
        a = Track(
            file_hash="3" * 64, file_path="/x", file_name="x.wav",
            file_size_bytes=1, title="a", state="complete",
            created_at=now_utc(), updated_at=now_utc(),
        )
        b = Track(
            file_hash="4" * 64, file_path="/y", file_name="y.wav",
            file_size_bytes=1, title="b", state="complete",
            created_at=now_utc(), updated_at=now_utc(),
        )
        session.add_all([a, b])
        session.flush()
        for tid in (a.id, b.id):
            session.add(
                TrackEmbedding(
                    track_id=tid, stem_file_id=None,
                    model=settings.clap_model, model_version=None, dim=2,
                    embedding=encode_embedding(np.array([1.0, 0.0], dtype=np.float32)),
                    created_at=now_utc(),
                )
            )
        session.commit()
        excluded_id = a.id
    finally:
        session.close()

    _stub_text_encoder_app(app, [1.0, 0.0])
    r = client.post(
        "/api/v1/recommend/text",
        json={"query": "anything", "k": 5, "exclude": [excluded_id]},
    )
    assert r.status_code == 200
    body = r.json()
    assert excluded_id not in [item["track_id"] for item in body]


# ---------------------------------------------------------------------------
# Reveal-in-Finder endpoint
# ---------------------------------------------------------------------------


def test_reveal_requires_path(client):
    r = client.post("/api/v1/files/reveal", json={})
    assert r.status_code == 400


def test_reveal_404_when_missing(client, tmp_path, app):
    app.state.settings.library_dir = tmp_path / "lib"
    app.state.settings.stems_dir = tmp_path / "stems"
    (tmp_path / "lib").mkdir()
    bogus = tmp_path / "lib" / "does_not_exist.wav"
    r = client.post("/api/v1/files/reveal", json={"path": str(bogus)})
    assert r.status_code == 404


def test_reveal_403_when_outside_allowed_dirs(client, tmp_path, app):
    app.state.settings.library_dir = tmp_path / "lib"
    app.state.settings.stems_dir = tmp_path / "stems"
    (tmp_path / "lib").mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"data")
    r = client.post("/api/v1/files/reveal", json={"path": str(outside)})
    assert r.status_code == 403


def test_reveal_success_invokes_command(client, tmp_path, app, monkeypatch):
    app.state.settings.library_dir = tmp_path / "lib"
    app.state.settings.stems_dir = tmp_path / "stems"
    (tmp_path / "lib").mkdir()
    target = tmp_path / "lib" / "track.wav"
    target.write_bytes(b"data")

    invocations: list[list[str]] = []

    class _FakePopen:
        def __init__(self, cmd, **_kwargs):
            invocations.append(list(cmd))

    import dance.api.routers.files as files_mod
    monkeypatch.setattr(files_mod.subprocess, "Popen", _FakePopen)

    r = client.post("/api/v1/files/reveal", json={"path": str(target)})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert len(invocations) == 1
    assert str(target) in " ".join(invocations[0])


# ---------------------------------------------------------------------------
# Tag endpoint
# ---------------------------------------------------------------------------


def test_tag_endpoint_zeroshot(client, app, fake_bridge):
    """POST /tracks/{id}/tag (default mode) runs CLAP zero-shot."""
    import numpy as np

    from dance.core.database import TrackEmbedding
    from dance.core.serialization import encode_embedding

    settings = get_settings()
    session = app.state.session_factory()
    try:
        t = Track(
            file_hash="9" * 64, file_path="/z", file_name="z.wav",
            file_size_bytes=1, title="z", state="complete",
            created_at=now_utc(), updated_at=now_utc(),
        )
        session.add(t)
        session.flush()
        session.add(
            TrackEmbedding(
                track_id=t.id, stem_file_id=None,
                model=settings.clap_model, model_version=None, dim=2,
                embedding=encode_embedding(np.array([1.0, 0.0], dtype=np.float32)),
                created_at=now_utc(),
            )
        )
        session.commit()
        track_id = t.id
    finally:
        session.close()

    # Stub the CLAP tagger's text encoder so we don't load real CLAP.
    import dance.llm.tagger as tagger_mod

    original = tagger_mod.ClapZeroShotTagger._ensure_encoder

    def stub_ensure(self):
        self._text_encoder = lambda _l: np.array([1.0, 0.0], dtype=np.float32)
        return self._text_encoder

    tagger_mod.ClapZeroShotTagger._ensure_encoder = stub_ensure
    try:
        r = client.post(f"/api/v1/tracks/{track_id}/tag")
    finally:
        tagger_mod.ClapZeroShotTagger._ensure_encoder = original

    assert r.status_code == 200
    body = r.json()
    assert body["id"] == track_id
    # Tags should be populated.
    assert isinstance(body["tags"], list)
    assert len(body["tags"]) > 0


def test_tag_endpoint_404(client):
    r = client.post("/api/v1/tracks/99999/tag")
    assert r.status_code == 404


def test_tag_endpoint_deep_disabled_by_default(client, app, make_track):
    """Deep mode is opt-in; requesting it when disabled returns 503."""
    session = app.state.session_factory()
    try:
        t = Track(
            file_hash="8" * 64, file_path="/d", file_name="d.wav",
            file_size_bytes=1, title="d", state="complete",
            created_at=now_utc(), updated_at=now_utc(),
        )
        session.add(t)
        session.commit()
        track_id = t.id
    finally:
        session.close()

    r = client.post(f"/api/v1/tracks/{track_id}/tag?deep=true")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# .als export endpoint
# ---------------------------------------------------------------------------


def _setup_exportable_track(session, make_track, tmp_path):
    """Create a COMPLETE track with stems on disk under tmp_path."""
    from tests.audio_fixtures import TrackSpec, write_track

    spec = TrackSpec(bpm=128.0, bars=2)
    lib = tmp_path / "library"
    lib.mkdir(parents=True, exist_ok=True)
    full = lib / "exp.wav"
    write_track(full, spec)
    stem_dir = tmp_path / "stems" / "exp"
    stem_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("drums", "bass", "vocals", "other"):
        p = stem_dir / f"{kind}.wav"
        write_track(p, spec)

    t = make_track(
        title="ExportMe",
        artist="Tester",
        file_path=str(full),
        duration_seconds=spec.duration_seconds,
        state="complete",
    )
    session.add(
        AudioAnalysis(
            track_id=t.id,
            stem_file_id=None,
            bpm=spec.bpm,
            key_camelot="8A",
            floor_energy=6,
            analyzed_at=now_utc(),
        )
    )
    for kind in ("drums", "bass", "vocals", "other"):
        session.add(
            StemFile(track_id=t.id, kind=kind, path=str(stem_dir / f"{kind}.wav"))
        )
    session.commit()
    return t


def _set_als_output_dir(app, tmp_path):
    """Point the app's Settings at a tmp als_output_dir."""
    app.state.settings.als_output_dir = tmp_path / "sets"
    (tmp_path / "sets").mkdir(parents=True, exist_ok=True)


def test_export_als_happy_path(client, app, session, make_track, tmp_path):
    """POST /tracks/{id}/als generates a .als file and returns its path."""
    _set_als_output_dir(app, tmp_path)
    t = _setup_exportable_track(session, make_track, tmp_path)

    r = client.post(f"/api/v1/tracks/{t.id}/als", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    out_path = body["out_path"]
    assert out_path.endswith(".als")
    assert body["size_bytes"] > 100
    # 4 stems + mix = 5 audio tracks.
    assert body["track_count"] == 5
    # No regions added → 0 locators.
    assert body["locator_count"] == 0

    # File actually exists on disk and is gzipped XML.
    import gzip
    from pathlib import Path as _P

    assert _P(out_path).exists()
    raw = _P(out_path).read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    xml = gzip.decompress(raw)
    assert b"<Ableton" in xml
    assert b"<LiveSet>" in xml


def test_export_als_404_when_track_missing(client, app, tmp_path):
    _set_als_output_dir(app, tmp_path)
    r = client.post("/api/v1/tracks/9999/als", json={})
    assert r.status_code == 404


def test_export_als_400_when_not_complete(client, app, session, make_track, tmp_path):
    _set_als_output_dir(app, tmp_path)
    t = make_track(state="pending")
    session.commit()
    r = client.post(f"/api/v1/tracks/{t.id}/als", json={})
    assert r.status_code == 400
    assert "complete" in r.json()["detail"].lower()


def test_export_als_403_when_out_path_outside_dir(
    client, app, session, make_track, tmp_path
):
    _set_als_output_dir(app, tmp_path)
    t = _setup_exportable_track(session, make_track, tmp_path)

    outside = tmp_path / "escape.als"
    r = client.post(
        f"/api/v1/tracks/{t.id}/als", json={"out_path": str(outside)}
    )
    assert r.status_code == 403


def test_export_als_accepts_custom_out_path_in_dir(
    client, app, session, make_track, tmp_path
):
    _set_als_output_dir(app, tmp_path)
    t = _setup_exportable_track(session, make_track, tmp_path)

    target = tmp_path / "sets" / "subdir" / "custom.als"
    r = client.post(
        f"/api/v1/tracks/{t.id}/als", json={"out_path": str(target)}
    )
    assert r.status_code == 200
    body = r.json()
    from pathlib import Path as _P

    assert _P(body["out_path"]).resolve() == target.resolve()
    assert target.exists()


# ---------------------------------------------------------------------------
# Pipeline ops endpoints
# ---------------------------------------------------------------------------


def test_pipeline_status_empty(client: TestClient) -> None:
    """With no tracks: every state present with count 0, in_progress False."""
    r = client.get("/api/v1/pipeline/status")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["in_progress"] is False
    assert body["errors"] == 0
    assert body["complete"] == 0
    assert body["weighted_progress"] == 0.0
    # Every TrackState value must be present (UI relies on this for a stable grid)
    counts = body["counts"]
    for state in (
        "pending", "analyzing", "analyzed", "separating", "separated",
        "analyzing_stems", "stems_analyzed", "detecting_regions",
        "regions_detected", "embedding", "embedded", "complete", "error",
    ):
        assert state in counts
        assert counts[state] == 0


def test_pipeline_status_counts_per_state(
    client: TestClient, session, make_track
) -> None:
    make_track(state="pending")
    make_track(state="analyzing")
    make_track(state="separated")
    make_track(state="separated")
    make_track(state="complete")
    make_track(state="error")
    session.commit()

    r = client.get("/api/v1/pipeline/status")
    body = r.json()
    assert body["total"] == 6
    assert body["counts"]["pending"] == 1
    assert body["counts"]["analyzing"] == 1
    assert body["counts"]["separated"] == 2
    assert body["counts"]["complete"] == 1
    assert body["counts"]["error"] == 1
    assert body["in_progress"] is True  # analyzing is an active stage
    assert body["errors"] == 1
    assert body["complete"] == 1


def test_pipeline_status_in_progress_false_when_only_terminal_states(
    client: TestClient, session, make_track
) -> None:
    make_track(state="complete")
    make_track(state="error")
    make_track(state="pending")  # pending is not "in progress" (no stage running on it)
    session.commit()

    r = client.get("/api/v1/pipeline/status")
    assert r.json()["in_progress"] is False


def test_pipeline_recent_ordered_by_updated_at_desc(
    client: TestClient, session, make_track
) -> None:
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = make_track(title="older", state="analyzed")
    older.updated_at = base
    newer = make_track(title="newer", state="separated")
    newer.updated_at = base + timedelta(seconds=10)
    middle = make_track(title="middle", state="analyzing")
    middle.updated_at = base + timedelta(seconds=5)
    session.commit()

    r = client.get("/api/v1/pipeline/recent")
    assert r.status_code == 200
    body = r.json()
    titles = [row["title"] for row in body]
    assert titles == ["newer", "middle", "older"]
    # Shape check on one row
    row = body[0]
    assert row["state"] == "separated"
    assert row["error_message"] is None
    assert "id" in row and "updated_at" in row


def test_pipeline_recent_respects_limit(
    client: TestClient, session, make_track
) -> None:
    for i in range(25):
        make_track(title=f"t{i}", state="analyzed")
    session.commit()

    r = client.get("/api/v1/pipeline/recent?limit=5")
    assert r.status_code == 200
    assert len(r.json()) == 5


def test_pipeline_recent_surfaces_error_message(
    client: TestClient, session, make_track
) -> None:
    t = make_track(title="broken", state="error")
    t.error_message = "separate: System error."
    session.commit()

    r = client.get("/api/v1/pipeline/recent?limit=1")
    body = r.json()
    assert body[0]["error_message"] == "separate: System error."


def test_pipeline_recent_filters_by_state(
    client: TestClient, session, make_track
) -> None:
    """?state=separated returns only tracks in that state."""
    make_track(title="a", state="separated")
    make_track(title="b", state="separated")
    make_track(title="c", state="analyzed")
    session.commit()

    r = client.get("/api/v1/pipeline/recent?state=separated&limit=100")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert all(row["state"] == "separated" for row in body)


def test_pipeline_status_weighted_progress(
    client: TestClient, session, make_track
) -> None:
    """4 tracks: 1 complete (6/6), 1 separated (2/6), 2 pending (0/6).
    Total stages: 8, max possible: 24, => 33.3%."""
    make_track(state="complete")
    make_track(state="separated")
    make_track(state="pending")
    make_track(state="pending")
    session.commit()

    body = client.get("/api/v1/pipeline/status").json()
    # 6 + 2 + 0 + 0 = 8 stages of 24 = 33.333% rounded to 33.3
    assert body["weighted_progress"] == 33.3
    assert body["complete"] == 1


def test_pipeline_status_weighted_progress_ignores_errors(
    client: TestClient, session, make_track
) -> None:
    """Errored tracks contribute 0 to progress (they need re-running)."""
    make_track(state="error")
    make_track(state="embedded")  # 5/6
    session.commit()

    body = client.get("/api/v1/pipeline/status").json()
    # (0 + 5) / (2 * 6) = 41.666… → 41.7
    assert body["weighted_progress"] == 41.7


# ---------------------------------------------------------------------------
# CSV ingest dedup + endpoints
# ---------------------------------------------------------------------------


_MINI_CSV = """Track URI,Track Name,Album Name,Artist Name(s),Duration (ms)
spotify:track:1,One More Time,Discovery,Daft Punk,320000
spotify:track:2,Sunsleeper,When Will We Land,Barry Can't Swim,260000
spotify:track:3,Faust,Faust EP,Argy;Son of Son,455000
spotify:track:4,,No Artist,Mystery,180000
"""


def test_dedup_finds_token_reordered_match() -> None:
    from dance.api.dedup import find_duplicate, index_existing

    idx = index_existing([(1, "Argy;Son of Son", "Faust")])
    # Incoming has reversed artist order
    assert find_duplicate("Son of Son, Argy", "Faust", idx) == 1


def test_dedup_distinguishes_versions() -> None:
    """Different mixes must NOT be flagged as the same track."""
    from dance.api.dedup import find_duplicate, index_existing

    idx = index_existing([(1, "Daft Punk", "One More Time")])
    assert find_duplicate("Daft Punk", "One More Time (Club Mix)", idx) is None
    assert find_duplicate("Daft Punk", "One More Time - Extended", idx) is None


def test_dedup_handles_diacritics() -> None:
    from dance.api.dedup import find_duplicate, index_existing

    idx = index_existing([(1, "Amélie", "Étoile")])
    assert find_duplicate("Amelie", "Etoile", idx) == 1


def test_ingest_preview_classifies_new_vs_duplicate(
    client: TestClient, session, make_track, tmp_path, app
) -> None:
    # Pre-populate one matching track in the DB
    make_track(title="One More Time", artist="Daft Punk", state="complete")
    session.commit()

    # Point library to an empty tmp dir
    app.state.settings.library_dir = tmp_path

    r = client.post(
        "/api/v1/pipeline/ingest/preview",
        json={"csv_text": _MINI_CSV},
    )
    assert r.status_code == 200
    body = r.json()
    # 4 rows total: 3 valid (one is a dupe), 1 parse error (no title for row 4)
    assert body["total_rows"] == 3
    assert len(body["parse_errors"]) == 1
    # Daft Punk should be flagged as duplicate; the other two are new
    dup_titles = {r["title"] for r in body["duplicates"]}
    new_titles = {r["title"] for r in body["new_rows"]}
    assert "One More Time" in dup_titles
    assert {"Sunsleeper", "Faust"} == new_titles


def test_ingest_preview_detects_existing_file_on_disk(
    client: TestClient, session, make_track, tmp_path, app
) -> None:
    app.state.settings.library_dir = tmp_path
    # Drop a file matching what the importer would write
    (tmp_path / "Daft Punk - One More Time.mp3").write_bytes(b"\x00" * 200_000)

    r = client.post(
        "/api/v1/pipeline/ingest/preview",
        json={"csv_text": _MINI_CSV},
    )
    body = r.json()
    # That row should be in duplicates even though no Track row exists
    targets = {r["target_path"] for r in body["duplicates"]}
    assert any("One More Time" in p for p in targets)


def test_ingest_commit_creates_job_and_returns_pending_items(
    client: TestClient, session, make_track, tmp_path, app, monkeypatch
) -> None:
    """Smoke-test commit endpoint without actually running yt-dlp.

    We monkeypatch ``download_track`` so the background thread completes
    immediately and we can assert the job goes from queued → done.
    """
    import time

    from dance.spotify import csv_importer

    app.state.settings.library_dir = tmp_path

    def _fake_download(row, library, *args, **kwargs):  # noqa: ANN001
        return ("ok", "fake 12 KB")

    monkeypatch.setattr(csv_importer, "download_track", _fake_download)
    # Re-import so the router's reference uses the patched function
    from dance.api.routers import pipeline as pipeline_router
    monkeypatch.setattr(pipeline_router, "download_track", _fake_download)

    r = client.post(
        "/api/v1/pipeline/ingest/commit",
        json={"csv_text": _MINI_CSV, "include_duplicates": True},
    )
    assert r.status_code == 200
    job = r.json()
    assert job["status"] in ("queued", "running", "done")
    assert job["total"] == 3
    job_id = job["id"]

    # Poll for completion (with a small budget)
    for _ in range(50):
        time.sleep(0.05)
        r = client.get(f"/api/v1/pipeline/jobs/{job_id}")
        body = r.json()
        if body["status"] == "done":
            break
    assert body["status"] == "done"
    assert body["counts"]["ok"] == 3
    assert all(it["status"] == "ok" for it in body["items"])


def test_ingest_commit_rejects_empty_after_dedup(
    client: TestClient, session, make_track, tmp_path, app
) -> None:
    """If every CSV row is a duplicate, commit returns 400."""
    app.state.settings.library_dir = tmp_path
    # Make every CSV track look like an existing DB track
    make_track(title="One More Time", artist="Daft Punk", state="complete")
    make_track(title="Sunsleeper", artist="Barry Can't Swim", state="complete")
    make_track(title="Faust", artist="Argy", state="complete")
    session.commit()

    r = client.post(
        "/api/v1/pipeline/ingest/commit",
        json={"csv_text": _MINI_CSV, "include_duplicates": False},
    )
    assert r.status_code == 400


def test_ingest_commit_rejects_bad_csv(client: TestClient) -> None:
    r = client.post(
        "/api/v1/pipeline/ingest/commit",
        json={"csv_text": "not,a,real,header\nfoo,bar,baz,qux"},
    )
    assert r.status_code == 400


def test_get_job_404(client: TestClient) -> None:
    r = client.get("/api/v1/pipeline/jobs/nonexistent")
    assert r.status_code == 404


def test_ingest_commit_per_row_selection(
    client: TestClient, app, tmp_path, monkeypatch
) -> None:
    """Per-row toggle: caller passes selected_keys, only those download."""
    import time
    from dance.api.routers import pipeline as pipeline_router
    from dance.spotify import csv_importer

    app.state.settings.library_dir = tmp_path
    downloaded: list[str] = []

    def _fake_download(row, library, *args, **kwargs):  # noqa: ANN001
        downloaded.append(f"{row.artist}|{row.title}")
        return ("ok", "fake")

    monkeypatch.setattr(csv_importer, "download_track", _fake_download)
    monkeypatch.setattr(pipeline_router, "download_track", _fake_download)

    r = client.post(
        "/api/v1/pipeline/ingest/commit",
        json={
            "csv_text": _MINI_CSV,
            "selected_keys": ["Daft Punk|One More Time"],
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    for _ in range(50):
        time.sleep(0.05)
        body = client.get(f"/api/v1/pipeline/jobs/{job_id}").json()
        if body["status"] == "done":
            break
    assert body["total"] == 1
    assert downloaded == ["Daft Punk|One More Time"]


def test_ingest_commit_per_row_empty_selection(client: TestClient, app, tmp_path) -> None:
    """selected_keys=[] is treated as 'select nothing' → 400."""
    app.state.settings.library_dir = tmp_path
    r = client.post(
        "/api/v1/pipeline/ingest/commit",
        json={"csv_text": _MINI_CSV, "selected_keys": []},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# JobRegistry persistence
# ---------------------------------------------------------------------------


def test_job_registry_round_trips_to_disk(tmp_path) -> None:
    """A registry with persist_path saves on mutation and reloads on init."""
    from dance.api.jobs import Job, JobRegistry

    persist = tmp_path / "jobs.json"
    reg = JobRegistry(persist_path=persist)
    job = reg.create("test_kind", ["a", "b", "c"])
    reg.update_item(job.id, 0, "ok", "first")
    reg.update_item(job.id, 1, "fail", "oops")
    reg.set_status(job.id, "done")

    assert persist.exists()

    # Fresh registry should see the same job
    reg2 = JobRegistry(persist_path=persist)
    loaded = reg2.get(job.id)
    assert loaded is not None
    assert loaded.kind == "test_kind"
    assert loaded.status == "done"
    assert loaded.items[0].status == "ok"
    assert loaded.items[0].message == "first"
    assert loaded.items[1].status == "fail"


def test_job_registry_heals_orphaned_running_jobs(tmp_path) -> None:
    """A job left as 'running' from a crashed previous process is moved to 'error'."""
    from dance.api.jobs import JobRegistry

    persist = tmp_path / "jobs.json"
    reg = JobRegistry(persist_path=persist)
    job = reg.create("crashy", ["x"])
    reg.set_status(job.id, "running")

    # Simulate API restart: new registry reads the persisted file
    reg2 = JobRegistry(persist_path=persist)
    healed = reg2.get(job.id)
    assert healed is not None
    assert healed.status == "error"
    assert healed.error is not None
    assert "restart" in healed.error.lower()


def test_job_registry_caps_history(tmp_path) -> None:
    """Old jobs beyond MAX_PERSISTED are dropped."""
    from dance.api import jobs as jobs_mod
    from dance.api.jobs import JobRegistry

    persist = tmp_path / "jobs.json"
    reg = JobRegistry(persist_path=persist)
    # _MAX_PERSISTED_JOBS = 50; create 52 to confirm the oldest 2 get dropped.
    cap = jobs_mod._MAX_PERSISTED_JOBS
    for i in range(cap + 2):
        reg.create("bulk", [f"item_{i}"])

    reg2 = JobRegistry(persist_path=persist)
    assert len(reg2.list(limit=200)) == cap


# ---------------------------------------------------------------------------
# /pipeline/process endpoint
# ---------------------------------------------------------------------------


def test_pipeline_process_creates_job_with_stage_items(
    client: TestClient, monkeypatch
) -> None:
    """The endpoint pre-populates 6 stage items so the UI has a layout from t=0."""
    import time
    from dance.api.routers import pipeline as pipeline_router

    # Stub the dispatcher path: don't actually load Demucs / CLAP in tests.
    class _FakeDispatcher:
        def __init__(self, *args, **kwargs):  # noqa: ANN001
            from dance.pipeline.events import EventBus

            self.events = EventBus()

        def ingest(self):
            return {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}

        def run(self):
            return {
                "analyze": {"processed": 0, "errors": 0, "skipped": 0},
                "separate": {"processed": 0, "errors": 0, "skipped": 0},
                "analyze_stems": {"processed": 0, "errors": 0, "skipped": 0},
                "detect_regions": {"processed": 0, "errors": 0, "skipped": 0},
                "embed": {"processed": 0, "errors": 0, "skipped": 0},
            }

    import dance.pipeline.dispatcher as dispatcher_mod
    monkeypatch.setattr(dispatcher_mod, "Dispatcher", _FakeDispatcher)

    r = client.post("/api/v1/pipeline/process")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "pipeline_run"
    assert body["total"] == 6
    labels = [it["label"] for it in body["items"]]
    assert labels == [
        "ingest",
        "analyze",
        "separate",
        "analyze_stems",
        "detect_regions",
        "embed",
    ]

    # Wait for the (fast, stubbed) dispatcher to finish
    job_id = body["id"]
    for _ in range(60):
        time.sleep(0.05)
        body = client.get(f"/api/v1/pipeline/jobs/{job_id}").json()
        if body["status"] == "done":
            break
    assert body["status"] == "done"
    assert body["counts"]["ok"] == 6


def test_pipeline_process_returns_409_when_already_running(
    client: TestClient,
) -> None:
    """Two concurrent /process calls must not both run."""
    from dance.api.jobs import get_job_registry

    reg = get_job_registry()
    # Manually plant an active pipeline_run job
    reg.create("pipeline_run", ["x"])

    r = client.post("/api/v1/pipeline/process")
    assert r.status_code == 409
    assert "already" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# DELETE /tracks/{id}
# ---------------------------------------------------------------------------


def test_delete_track_cascades(
    client: TestClient, session, make_track
) -> None:
    """Delete a track and verify it's gone along with its stems / regions / tags."""
    t = make_track(title="goner", state="complete")
    _add_fullmix_analysis(session, t, bpm=120.0)
    _add_tag(session, t, "doomed")
    # A stem with its own analysis row
    sf = StemFile(track_id=t.id, kind="drums", path="/tmp/x.wav")
    session.add(sf)
    session.commit()
    session.add(
        AudioAnalysis(
            track_id=t.id, stem_file_id=sf.id, bpm=120.0, analyzed_at=now_utc()
        )
    )
    session.add(
        Region(
            track_id=t.id,
            position_ms=0,
            length_ms=1000,
            region_type="cue",
            source="auto",
        )
    )
    session.commit()

    track_id = t.id
    sf_id = sf.id

    r = client.delete(f"/api/v1/tracks/{track_id}")
    assert r.status_code == 204

    # The endpoint used a different session; expire ours so we re-query.
    session.expire_all()

    # Everything related is gone
    assert session.get(Track, track_id) is None
    assert session.get(StemFile, sf_id) is None
    assert (
        session.query(Region).filter(Region.track_id == track_id).count() == 0
    )
    assert (
        session.query(AudioAnalysis)
        .filter(AudioAnalysis.track_id == track_id)
        .count()
        == 0
    )
    assert (
        session.query(TrackTag).filter(TrackTag.track_id == track_id).count() == 0
    )


def test_delete_track_404(client: TestClient) -> None:
    r = client.delete("/api/v1/tracks/99999")
    assert r.status_code == 404


def test_delete_track_with_session_play_does_not_fk_violate(
    client: TestClient, session, make_track
) -> None:
    """SessionPlay refs a track but doesn't cascade. Delete must clean up."""
    from datetime import datetime, timezone

    t = make_track(title="played and deleted")
    sess = DjSession(name="set 1", started_at=datetime.now(timezone.utc))
    session.add(sess)
    session.commit()
    play = SessionPlay(
        session_id=sess.id,
        track_id=t.id,
        played_at=datetime.now(timezone.utc),
        position_in_set=1,
    )
    session.add(play)
    session.commit()

    track_id = t.id
    sess_id = sess.id
    r = client.delete(f"/api/v1/tracks/{track_id}")
    assert r.status_code == 204
    session.expire_all()
    # SessionPlay row gone, session still there
    assert (
        session.query(SessionPlay)
        .filter(SessionPlay.track_id == track_id)
        .count()
        == 0
    )
    assert session.get(DjSession, sess_id) is not None


# ---------------------------------------------------------------------------
# /pipeline/scan and /pipeline/watch
# ---------------------------------------------------------------------------


def test_pipeline_scan_returns_counts(client: TestClient, tmp_path, app) -> None:
    """Empty library: 0 new, 0 errors. Verifies the endpoint plumbing without
    needing the heavy ML deps."""
    app.state.settings.library_dir = tmp_path / "empty"
    (tmp_path / "empty").mkdir()
    r = client.post("/api/v1/pipeline/scan")
    assert r.status_code == 200
    body = r.json()
    assert body["new"] == 0
    assert body["errors"] == 0


def test_pipeline_watch_get_defaults_disabled(client: TestClient) -> None:
    r = client.get("/api/v1/pipeline/watch")
    assert r.status_code == 200
    body = r.json()
    assert "enabled" in body
    assert "interval_seconds" in body
    assert isinstance(body["enabled"], bool)


def test_pipeline_watch_post_toggles(client: TestClient) -> None:
    r = client.post("/api/v1/pipeline/watch", json={"enabled": True})
    assert r.status_code == 200
    assert r.json()["enabled"] is True

    r = client.post("/api/v1/pipeline/watch", json={"enabled": False})
    assert r.json()["enabled"] is False


def test_pipeline_watch_post_invalid_400(client: TestClient) -> None:
    r = client.post("/api/v1/pipeline/watch", json={"enabled": "yes"})
    assert r.status_code == 400


def test_pipeline_watch_persists_across_init(client: TestClient, app) -> None:
    """Flag round-trips through watch.json. Uses the test app's data_dir
    so we don't leak to the user's real ~/.dance/."""
    from dance.api.routers import pipeline as pipeline_router

    test_settings = app.state.settings

    # Toggle ON via the endpoint
    client.post("/api/v1/pipeline/watch", json={"enabled": True})
    # Force-reload as a new "init" would
    pipeline_router._WATCH_STATE["enabled"] = False  # pretend fresh process
    pipeline_router._load_watch_state(test_settings)
    assert pipeline_router._WATCH_STATE["enabled"] is True

    # Toggle OFF and verify
    client.post("/api/v1/pipeline/watch", json={"enabled": False})
    pipeline_router._WATCH_STATE["enabled"] = True
    pipeline_router._load_watch_state(test_settings)
    assert pipeline_router._WATCH_STATE["enabled"] is False


# ---------------------------------------------------------------------------
# Per-column rec stream (Phase 3 — live-remixing redesign).
# ---------------------------------------------------------------------------


def _seed_stem(
    session,
    track,
    *,
    kind: str,
    bpm: float,
    key: str | None,
    embedding: list[float] | None = None,
):
    """Helper: create a StemFile + its AudioAnalysis + optional embedding.

    Returns the StemFile id.
    """
    import numpy as np

    from dance.core.database import StemFile, TrackEmbedding
    from dance.core.serialization import encode_embedding

    sf = StemFile(
        track_id=track.id,
        kind=kind,
        path=f"/tmp/{track.id}-{kind}.wav",
        created_at=now_utc(),
    )
    session.add(sf)
    session.flush()
    session.add(
        AudioAnalysis(
            track_id=track.id,
            stem_file_id=sf.id,
            bpm=bpm,
            dominant_pitch_camelot=key,
            floor_energy=5,
            analyzed_at=now_utc(),
            created_at=now_utc(),
        )
    )
    if embedding is not None:
        session.add(
            TrackEmbedding(
                track_id=track.id,
                stem_file_id=sf.id,
                model="test",
                model_version=None,
                dim=len(embedding),
                embedding=encode_embedding(np.asarray(embedding, dtype=np.float32)),
                created_at=now_utc(),
            )
        )
    return sf.id


def test_recommend_by_column_empty_combo_ranks_by_bpm(
    client: TestClient, session, fake_bridge, make_track
):
    """With no combo to compare against, candidates rank by BPM proximity
    to ``master_bpm`` (key weight zeroes out when there's nothing to match)."""
    t1 = make_track(title="exact-bpm")
    t2 = make_track(title="off-by-six")
    t3 = make_track(title="off-by-fifteen")
    _seed_stem(session, t1, kind="drums", bpm=128.0, key="8A")
    _seed_stem(session, t2, kind="drums", bpm=122.0, key="8A")
    _seed_stem(session, t3, kind="drums", bpm=113.0, key="8A")
    session.commit()

    r = client.post(
        "/api/v1/recommend/by-column",
        json={
            "column": "drums",
            "combo_stem_ids": [],
            "master_bpm": 128.0,
            "k": 5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["column"] == "drums"
    assert body["combo_size"] == 0
    titles = [r["track_title"] for r in body["recs"]]
    assert titles == ["exact-bpm", "off-by-six", "off-by-fifteen"]
    assert body["recs"][0]["score"] > body["recs"][1]["score"]


def test_recommend_by_column_uses_combo_embedding(
    client: TestClient, session, fake_bridge, make_track
):
    """Candidate stems get cosine-scored against the averaged combo embedding;
    the closer a candidate's vector is to the combo, the higher it ranks."""
    # Combo: drums + bass of the same direction.
    seed = make_track(title="seed")
    combo_stem_ids = [
        _seed_stem(
            session, seed, kind="drums", bpm=128.0, key="8A",
            embedding=[1.0, 0.0, 0.0],
        ),
        _seed_stem(
            session, seed, kind="bass", bpm=128.0, key="8A",
            embedding=[1.0, 0.0, 0.0],
        ),
    ]
    # Candidates: vocals — one aligned to combo, one orthogonal.
    aligned_track = make_track(title="aligned")
    orth_track = make_track(title="orthogonal")
    _seed_stem(
        session, aligned_track, kind="vocals", bpm=128.0, key="2A",
        embedding=[1.0, 0.0, 0.0],
    )
    _seed_stem(
        session, orth_track, kind="vocals", bpm=128.0, key="2A",
        embedding=[0.0, 1.0, 0.0],
    )
    session.commit()

    r = client.post(
        "/api/v1/recommend/by-column",
        json={
            "column": "vocals",
            "combo_stem_ids": combo_stem_ids,
            "master_bpm": 128.0,
            "k": 5,
        },
    )
    body = r.json()
    titles = [rec["track_title"] for rec in body["recs"]]
    assert titles[0] == "aligned"
    assert titles[1] == "orthogonal"
    assert body["recs"][0]["score"] > body["recs"][1]["score"]
    # The aligned candidate's embedding score should be ~1.0 (cosine 1 → mapped to 1).
    assert body["recs"][0]["score_breakdown"]["embedding"] > 0.9


def test_recommend_by_column_mix_returns_full_tracks(
    client: TestClient, session, fake_bridge, make_track
):
    """Asking for the 'mix' column returns whole tracks (stem_file_id null)."""
    t = make_track(title="full-mix")
    session.add(
        AudioAnalysis(
            track_id=t.id,
            stem_file_id=None,
            bpm=124.0,
            key_camelot="5A",
            floor_energy=6,
            analyzed_at=now_utc(),
            created_at=now_utc(),
        )
    )
    session.commit()

    r = client.post(
        "/api/v1/recommend/by-column",
        json={"column": "mix", "combo_stem_ids": [], "master_bpm": 124.0, "k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["column"] == "mix"
    assert all(rec["stem_file_id"] is None for rec in body["recs"])
    titles = [rec["track_title"] for rec in body["recs"]]
    assert "full-mix" in titles


def test_recommend_by_column_rejects_invalid_column(
    client: TestClient, app, fake_bridge
):
    r = client.post(
        "/api/v1/recommend/by-column",
        json={"column": "guitar", "combo_stem_ids": [], "master_bpm": 128, "k": 3},
    )
    assert r.status_code == 400
    assert "unknown column" in r.json()["detail"]


def test_recommend_by_column_exclude_tracks(
    client: TestClient, session, fake_bridge, make_track
):
    """Tracks listed in exclude_track_ids never appear in the result."""
    t_keep = make_track(title="keeper")
    t_skip = make_track(title="excluded")
    _seed_stem(session, t_keep, kind="drums", bpm=128.0, key="8A")
    _seed_stem(session, t_skip, kind="drums", bpm=128.0, key="8A")
    session.commit()

    r = client.post(
        "/api/v1/recommend/by-column",
        json={
            "column": "drums",
            "combo_stem_ids": [],
            "master_bpm": 128.0,
            "exclude_track_ids": [t_skip.id],
            "k": 5,
        },
    )
    body = r.json()
    titles = [rec["track_title"] for rec in body["recs"]]
    assert "excluded" not in titles
    assert "keeper" in titles


# ---------------------------------------------------------------------------
# Preview / cue endpoints
# ---------------------------------------------------------------------------


def test_preview_stem_routes_to_bridge_with_stem_file_path(
    client: TestClient, session, fake_bridge, make_track, tmp_path
):
    """POST /ableton/preview with column=drums resolves to the drums
    StemFile.path and asks the bridge to audition it on the Cue track."""
    from dance.core.database import StemFile

    track = make_track(title="Probe")
    stem_path = tmp_path / "probe-drums.wav"
    stem_path.write_bytes(b"")
    session.add(
        StemFile(
            track_id=track.id,
            kind="drums",
            path=str(stem_path),
            created_at=now_utc(),
        )
    )
    session.commit()

    r = client.post(
        "/api/v1/ableton/preview",
        json={"track_id": track.id, "column": "drums"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["audio_path"] == str(stem_path)
    assert "drums" in (body["label"] or "")
    assert len(fake_bridge.preview_calls) == 1
    assert fake_bridge.preview_calls[0]["audio_path"] == str(stem_path)


def test_preview_mix_uses_full_track_file_path(
    client: TestClient, session, fake_bridge, make_track, tmp_path
):
    """column='mix' previews the original full-track audio file, not a stem."""
    fpath = tmp_path / "full-mix.mp3"
    fpath.write_bytes(b"")
    track = make_track(title="Full Mix Probe", file_path=str(fpath))
    session.commit()

    r = client.post(
        "/api/v1/ableton/preview",
        json={"track_id": track.id, "column": "mix"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audio_path"] == str(fpath)
    assert fake_bridge.preview_calls[-1]["audio_path"] == str(fpath)


def test_preview_unknown_column_rejected(
    client: TestClient, session, fake_bridge, make_track
):
    track = make_track(title="X")
    session.commit()
    r = client.post(
        "/api/v1/ableton/preview",
        json={"track_id": track.id, "column": "guitar"},
    )
    assert r.status_code == 400
    assert "unknown column" in r.json()["detail"]


def test_preview_missing_audio_returns_404(
    client: TestClient, session, fake_bridge, make_track
):
    """A track with no matching StemFile for the requested column → 404."""
    track = make_track(title="Headless")  # no stems for any column
    session.commit()
    r = client.post(
        "/api/v1/ableton/preview",
        json={"track_id": track.id, "column": "drums"},
    )
    assert r.status_code == 404


def test_preview_stop_clears_bridge(
    client: TestClient, fake_bridge
):
    r = client.post("/api/v1/ableton/preview/stop")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert fake_bridge.stop_preview_calls == 1


# ---------------------------------------------------------------------------
# Cell-level loads (kinds filter) — Phase 7
# ---------------------------------------------------------------------------


def test_load_track_full_song_default_loads_all_stems(
    client: TestClient, session, fake_bridge, make_track, tmp_path
):
    """No ``kinds`` field → backward-compat: whole-song load (all 4 stems)."""
    from dance.core.database import StemFile

    track = make_track(title="Full Song Probe")
    for kind in ("drums", "bass", "vocals", "other"):
        p = tmp_path / f"{kind}.wav"
        p.write_bytes(b"")
        session.add(
            StemFile(
                track_id=track.id, kind=kind, path=str(p), created_at=now_utc()
            )
        )
    session.commit()

    r = client.post(
        "/api/v1/ableton/load-track",
        json={"track_id": track.id, "include_stems": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # FakeAbletonBridge default returns indices for all stems supplied.
    assert "drums" in body["track_indices"]
    assert "bass" in body["track_indices"]
    assert "vocals" in body["track_indices"]
    assert "other" in body["track_indices"]


def test_load_track_kinds_filter_passed_through_to_bridge(
    client: TestClient, session, fake_bridge, make_track, tmp_path
):
    """``kinds=["drums"]`` only asks the bridge to load drums — no other
    stems. Verified by inspecting the recorded bridge call."""
    from dance.core.database import StemFile

    track = make_track(title="Drum Only Probe")
    for kind in ("drums", "bass", "vocals", "other"):
        p = tmp_path / f"{kind}.wav"
        p.write_bytes(b"")
        session.add(
            StemFile(
                track_id=track.id, kind=kind, path=str(p), created_at=now_utc()
            )
        )
    session.commit()

    r = client.post(
        "/api/v1/ableton/load-track",
        json={"track_id": track.id, "include_stems": True, "kinds": ["drums"]},
    )
    assert r.status_code == 200
    # The fake bridge records every push_track_to_live call; we ensure the
    # kinds filter went through to the bridge layer.
    last = fake_bridge.push_calls[-1]
    assert last["kinds"] == ["drums"]


def test_deck_map_returns_cells_not_scenes(
    client: TestClient, fake_bridge
):
    """GET /ableton/decks responds with a ``cells`` list (cell-level shape),
    not a legacy ``scenes`` list."""
    fake_bridge.deck_state_return = {
        "columns": {"mix": 0, "drums": 1, "bass": 2, "vocals": 3, "other": 4},
        "cells": [
            {"scene_index": 0, "kind": "drums", "track_id": 99},
        ],
    }
    r = client.get("/api/v1/ableton/decks")
    assert r.status_code == 200
    body = r.json()
    assert "cells" in body
    assert "scenes" not in body
    assert len(body["cells"]) == 1
    assert body["cells"][0]["kind"] == "drums"
    assert body["cells"][0]["scene_index"] == 0


# ---------------------------------------------------------------------------
# Waveform peaks
# ---------------------------------------------------------------------------


def test_track_waveform_endpoint_returns_peaks(
    client: TestClient, session, make_track, tmp_path
):
    """GET /tracks/{id}/waveform decodes the track audio and returns a
    normalized envelope of length num_peaks."""
    from tests.audio_fixtures import TrackSpec, write_track

    spec = TrackSpec(bpm=128.0, bars=2)
    audio_path = tmp_path / "track-wave.wav"
    write_track(audio_path, spec)

    track = make_track(title="WaveTrack", file_path=str(audio_path))
    session.commit()

    r = client.get(f"/api/v1/tracks/{track.id}/waveform")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["num_peaks"] == 200
    assert len(body["peaks"]) == 200
    assert all(0.0 <= p <= 1.0 for p in body["peaks"])
    assert body["duration_seconds"] > 0
    # The synthetic track is non-silent so at least one window should peak.
    assert max(body["peaks"]) > 0.0


def test_stem_waveform_endpoint_returns_peaks(
    client: TestClient, session, make_track, tmp_path
):
    """GET /stems/{id}/waveform decodes a stem and returns its envelope."""
    from tests.audio_fixtures import TrackSpec, write_track

    spec = TrackSpec(bpm=128.0, bars=2)
    stem_path = tmp_path / "drums-wave.wav"
    write_track(stem_path, spec)

    track = make_track(title="StemWaveHost")
    stem = StemFile(
        track_id=track.id,
        kind="drums",
        path=str(stem_path),
        created_at=now_utc(),
    )
    session.add(stem)
    session.commit()

    r = client.get(f"/api/v1/stems/{stem.id}/waveform")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["num_peaks"] == 200
    assert len(body["peaks"]) == 200
    assert all(0.0 <= p <= 1.0 for p in body["peaks"])
    assert body["duration_seconds"] > 0


def test_waveform_missing_audio_404(
    client: TestClient, session, make_track, tmp_path
):
    """A DB row pointing at a path that doesn't exist on disk → 404."""
    bogus = tmp_path / "does-not-exist.wav"
    track = make_track(title="Ghost", file_path=str(bogus))

    stem = StemFile(
        track_id=track.id,
        kind="vocals",
        path=str(bogus),
        created_at=now_utc(),
    )
    session.add(stem)
    session.commit()

    r = client.get(f"/api/v1/tracks/{track.id}/waveform")
    assert r.status_code == 404

    r = client.get(f"/api/v1/stems/{stem.id}/waveform")
    assert r.status_code == 404


def test_waveform_caches_sidecar_json(
    client: TestClient, session, make_track, tmp_path
):
    """After the first call, a ``.waveform.json`` lands next to the audio
    file and the second call returns the same peaks (cache hit)."""
    from tests.audio_fixtures import TrackSpec, write_track

    spec = TrackSpec(bpm=128.0, bars=2)
    audio_path = tmp_path / "cached-wave.wav"
    write_track(audio_path, spec)
    sidecar = audio_path.with_name(audio_path.name + ".waveform.json")
    assert not sidecar.exists()

    track = make_track(title="Cacheable", file_path=str(audio_path))
    session.commit()

    r1 = client.get(f"/api/v1/tracks/{track.id}/waveform")
    assert r1.status_code == 200
    assert sidecar.exists(), "sidecar cache should have been written"
    peaks1 = r1.json()["peaks"]

    r2 = client.get(f"/api/v1/tracks/{track.id}/waveform")
    assert r2.status_code == 200
    assert r2.json()["peaks"] == peaks1
