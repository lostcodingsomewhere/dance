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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_bridge() -> FakeAbletonBridge:
    return FakeAbletonBridge()


@pytest.fixture
def app(session_factory: sessionmaker, fake_bridge: FakeAbletonBridge):
    return create_app(
        settings=get_settings(),
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
