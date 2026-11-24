from __future__ import annotations

import gzip
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from lxml import etree
from pythonosc import udp_client

from dance.api import create_app
from dance.config import get_settings
from dance.core.database import Track, _reset_engine_for_tests, get_session, get_session_factory
from dance.osc.bridge import AbletonBridge
from dance.osc.listener import AbletonOSCListener
from tests.test_cue_pads_e2e import API, BLOCK_MS
from tests.test_export_set_e2e import (
    PADS_PER_PAGE,
    _pads,
    _planned_set,
    _stem_path,
    three_songs_through_the_pipeline,
)

STEM_ROLES = ("drums", "bass", "vocals", "other")
SONG_ROLES = (*STEM_ROLES, "mix")
DECK_KIND_BY_TRACK_NAME = {
    f"{role.capitalize()} {side.upper()}": f"{role}_{side}"
    for side in ("a", "b")
    for role in SONG_ROLES
}
SETTLE_SECONDS = 5.0


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class LiveStandIn:
    def __init__(self) -> None:
        self.receive_port = _free_port()
        self.reply_port = _free_port()
        self.received: list[tuple[str, tuple[Any, ...]]] = []
        self._track_names: list[str] = []
        self._clips: dict[tuple[int, int], dict[str, Any]] = {}
        self._scene_count = 0
        self._tempo_listener_registered = False
        self._lock = threading.Lock()
        self._listener = AbletonOSCListener(port=self.receive_port)
        self._reply = udp_client.SimpleUDPClient("127.0.0.1", self.reply_port)

    def __enter__(self) -> LiveStandIn:
        self._listener.on_any(self._answer)
        self._listener.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._listener.stop()

    def open(self, set_path: str | Path) -> None:
        live_set = etree.fromstring(gzip.decompress(Path(set_path).read_bytes()))
        tracks = list(live_set.find(".//Tracks"))
        with self._lock:
            self._track_names = [t.find("Name/EffectiveName").get("Value") for t in tracks]
            self._clips = {}
            self._scene_count = len(live_set.findall(".//Scenes/Scene"))
            self._tempo_listener_registered = False
            for track_index, track in enumerate(tracks):
                slots = track.findall("DeviceChain/MainSequencer/ClipSlotList/ClipSlot")
                for row, slot in enumerate(slots):
                    clip = slot.find("ClipSlot/Value/AudioClip")
                    if clip is None:
                        continue
                    self._clips[(track_index, row)] = {
                        "name": clip.find("Name").get("Value"),
                        "start_marker": float(clip.find("Loop/LoopStart").get("Value"))
                        + float(clip.find("Loop/StartRelative").get("Value")),
                        "file_path": clip.find("SampleRef/FileRef/Path").get("Value"),
                    }
        self._reply.send_message("/live/startup", [])

    def track_index(self, track_name: str) -> int:
        return self._track_names.index(track_name)

    def clip(self, track_name: str, row: int) -> dict[str, Any]:
        return self._clips[(self.track_index(track_name), row)]

    def put_clip(self, track_name: str, row: int, **clip: Any) -> None:
        with self._lock:
            self._clips[(self.track_index(track_name), row)] = clip

    def change_tempo(self, bpm: float) -> None:
        if self._tempo_listener_registered:
            self._reply.send_message("/live/song/get/tempo", [bpm])

    def fired(self) -> list[tuple[int, int]]:
        with self._lock:
            return [
                (int(args[0]), int(args[1]))
                for address, args in self.received
                if address == "/live/clip_slot/fire"
            ]

    def _answer(self, address: str, args: tuple[Any, ...]) -> None:
        with self._lock:
            self.received.append((address, args))
            if address == "/live/song/start_listen/tempo":
                self._tempo_listener_registered = True
            elif address == "/live/song/get/track_names":
                self._reply.send_message(address, list(self._track_names))
            elif address == "/live/song/get/num_tracks":
                self._reply.send_message(address, [len(self._track_names)])
            elif address == "/live/song/get/num_scenes":
                self._reply.send_message(address, [self._scene_count])
            elif address == "/live/track/get/clips/name":
                track = int(args[0])
                names = [
                    self._clips.get((track, row), {}).get("name")
                    for row in range(self._scene_count)
                ]
                self._reply.send_message(address, [track, *names])
            elif address.startswith("/live/clip/get/"):
                key = (int(args[0]), int(args[1]))
                prop = address.rsplit("/", 1)[1]
                if key in self._clips and prop in self._clips[key]:
                    self._reply.send_message(address, [*key, self._clips[key][prop]])


@pytest.fixture
def booth(monkeypatch):
    track_ids = three_songs_through_the_pipeline(monkeypatch)
    settings = get_settings()
    with LiveStandIn() as live:
        bridge = AbletonBridge(
            send_port=live.receive_port,
            listen_port=live.reply_port,
            state_file=settings.data_dir / "deck_state.json",
        )
        app = create_app(
            settings=settings,
            bridge=bridge,
            session_factory=get_session_factory(settings.db_url),
        )
        bridge.start()
        try:
            with TestClient(app) as client:
                yield client, track_ids, live
        finally:
            bridge.stop()
    _reset_engine_for_tests()


def _cells(client: TestClient) -> dict[tuple[int, str], dict]:
    deck_map = client.get(f"{API}/ableton/decks").json()
    return {(cell["scene_index"], cell["kind"]): cell for cell in deck_map["cells"]}


def _cells_once_there_are(client: TestClient, count: int) -> dict[tuple[int, str], dict]:
    deadline = time.monotonic() + SETTLE_SECONDS
    while time.monotonic() < deadline:
        cells = _cells(client)
        if len(cells) == count:
            return cells
        time.sleep(0.05)
    raise AssertionError(f"the Booth shows {len(_cells(client))} cells, expected {count}")


def _settles(condition) -> bool:
    deadline = time.monotonic() + SETTLE_SECONDS
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def _cue(cell: dict) -> tuple[int | None, str | None, bool]:
    return cell["cue"], cell["cue_label"], cell["cue_silent"]


def _exported(client: TestClient, queues: dict[str, list[int]]) -> dict:
    exported = client.post(f"{API}/sets/{_planned_set(client, queues)}/als", json={})
    assert exported.status_code == 200, exported.text
    assert exported.json()["skipped"] == []
    return exported.json()


def test_the_booth_shows_a_prepared_set_as_soon_as_live_opens_it(booth):
    client, ids, live = booth
    first, second, third = ids["first_song"], ids["second_song"], ids["third_song"]
    exported = _exported(client, {"song": [first, second, third], "vocals": [second]})
    assert client.get(f"{API}/ableton/decks").json() == {"columns": None, "cells": []}

    live.open(exported["out_path"])
    cells = _cells_once_there_are(client, exported["clip_count"])

    columns = client.get(f"{API}/ableton/decks").json()["columns"]
    assert columns == {
        kind: live.track_index(name) for name, kind in DECK_KIND_BY_TRACK_NAME.items()
    }

    loads = [
        (first, "a", 0, SONG_ROLES),
        (second, "b", 0, SONG_ROLES),
        (third, "a", 1, SONG_ROLES),
        (second, "a", 2, ("vocals",)),
    ]
    silent_cues = 0
    for track_id, side, page, roles in loads:
        for role in roles:
            pads = _pads(client, track_id, role)
            assert [pad["slot"] for pad in pads] == [1, 2, 3, 4, 5]
            for pad in pads:
                cell = cells[(page * PADS_PER_PAGE + pad["slot"] - 1, f"{role}_{side}")]
                assert cell["track_id"] == track_id
                assert _cue(cell) == (pad["slot"], pad["section_label"], pad["silent"])
                silent_cues += pad["silent"]
    assert silent_cues > 0

    cue_three_on_deck_b = PADS_PER_PAGE * 0 + 2
    played = client.post(f"{API}/ableton/transport/fire-deck/b/{cue_three_on_deck_b}")
    assert played.status_code == 200 and played.json()["ok"] is True
    deck_b_pads = sorted((columns[f"{role}_b"], cue_three_on_deck_b) for role in SONG_ROLES)
    assert _settles(lambda: sorted(live.fired()) == deck_b_pads), live.fired()

    spare_vocal_cue_two = 2 * PADS_PER_PAGE + 1
    played = client.post(
        f"{API}/ableton/transport/fire-deck/a/{spare_vocal_cue_two}",
        params={"loaded_only": True},
    )
    assert played.status_code == 200 and played.json()["fired"] == 1
    with_the_spare_pad = sorted([*deck_b_pads, (columns["vocals_a"], spare_vocal_cue_two)])
    assert _settles(lambda: sorted(live.fired()) == with_the_spare_pad), live.fired()

    live.change_tempo(131.5)
    assert _settles(lambda: client.get(f"{API}/ableton/state").json()["tempo"] == 131.5)


def test_a_pad_is_called_a_cue_only_where_live_starts_it_at_that_cue(booth):
    client, ids, live = booth
    first = ids["first_song"]
    exported = _exported(client, {"song": [first]})
    live.open(exported["out_path"])
    _cells_once_there_are(client, exported["clip_count"])

    nudged = live.clip("Drums A", 2)
    live.put_clip("Drums A", 2, **{**nudged, "start_marker": nudged["start_marker"] + 8.0})
    title = client.get(f"{API}/tracks/{first}").json()["title"]
    live.put_clip(
        "Bass B",
        3,
        name=f"{title} (bass B)",
        start_marker=0.0,
        file_path=str(Path(_stem_path(client, first, "bass")).resolve()),
    )

    resynced = client.post(f"{API}/ableton/decks/resync")
    assert resynced.status_code == 200, resynced.text
    assert resynced.json()["unmatched"] == []
    assert resynced.json()["adopted"] == exported["clip_count"] + 1

    cells = _cells(client)
    assert _cue(cells[(2, "drums_a")]) == (None, None, False)
    assert cells[(1, "drums_a")]["cue"] == 2
    assert cells[(2, "bass_a")]["cue"] == 3
    assert cells[(3, "bass_b")]["track_id"] == first
    assert _cue(cells[(3, "bass_b")]) == (None, None, False)

    moved = client.put(f"{API}/tracks/{first}/pads/vocals/1", json={"position_ms": 5 * BLOCK_MS})
    assert moved.status_code == 200, moved.text
    cells = _cells(client)
    assert _cue(cells[(0, "vocals_a")]) == (None, None, False)
    assert cells[(0, "other_a")]["cue"] == 1


def test_two_songs_with_one_title_are_told_apart_by_their_audio_files(booth):
    client, ids, live = booth
    earlier, later = sorted([ids["first_song"], ids["second_song"]])
    session = get_session(get_settings().db_url)
    for track in session.query(Track).filter(Track.id.in_([earlier, later])):
        track.title = "Losing Control"
    session.commit()
    session.close()

    exported = _exported(client, {"song": [later]})
    live.open(exported["out_path"])
    cells = _cells_once_there_are(client, exported["clip_count"])

    assert {cell["track_id"] for cell in cells.values()} == {later}
    assert {cell["cue"] for cell in cells.values()} == {1, 2, 3, 4, 5}
