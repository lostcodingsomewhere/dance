from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dance.config import get_settings
from dance.core.database import Track, get_session
from tests.live_stand_in import booth_with_live
from tests.test_cue_pads_e2e import API, BLOCK_MS
from tests.test_export_set_e2e import PADS_PER_PAGE, _pads, _planned_set, _stem_path

STEM_ROLES = ("drums", "bass", "vocals", "other")
SONG_ROLES = (*STEM_ROLES, "mix")
DECK_KIND_BY_TRACK_NAME = {
    f"{role.capitalize()} {side.upper()}": f"{role}_{side}"
    for side in ("a", "b")
    for role in SONG_ROLES
}
SETTLE_SECONDS = 5.0


@pytest.fixture
def booth(monkeypatch):
    yield from booth_with_live(monkeypatch)


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
