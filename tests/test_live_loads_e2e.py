from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dance.config import get_settings
from dance.core.database import Region, RegionType, get_session
from tests.live_stand_in import GRID_GUARD_DELAYS, LiveStandIn, SetFile, booth_with_live, clip_tempo
from tests.test_booth_pages_e2e import (
    SONG_ROLES,
    _cells,
    _cells_once_there_are,
    _cue,
    _exported,
    _settles,
)
from tests.test_cue_pads_e2e import API
from tests.test_export_set_e2e import (
    PADS_PER_PAGE,
    WARP_BEATS,
    WARP_COMPLEX,
    WARP_COMPLEX_PRO,
    _pads,
    _stem_path,
)

HALF_TEMPO_FROM_A_LATE_FIRST_BEAT = [(0.0, 1.63), (4.0, 5.38), (8.0, 9.13), (8.03125, 9.16)]
WARP_MODE_BY_ROLE = {
    "drums": WARP_BEATS,
    "bass": WARP_BEATS,
    "vocals": WARP_COMPLEX_PRO,
    "other": WARP_COMPLEX,
    "mix": WARP_COMPLEX,
}


@pytest.fixture
def booth(monkeypatch):
    yield from booth_with_live(monkeypatch)


def _deck_track(role: str, side: str) -> str:
    return f"{role.capitalize()} {side.upper()}"


def _load(client: TestClient, **request: Any) -> dict:
    loaded = client.post(f"{API}/ableton/load-track", json=request)
    assert loaded.status_code == 200, loaded.text
    return loaded.json()


def _landed(loaded: dict) -> tuple[int, str, int]:
    return loaded["page"], loaded["side"], loaded["scene_index"]


def _open_in_live(client: TestClient, live: LiveStandIn, queues: dict[str, list[int]]) -> dict:
    exported = _exported(client, queues)
    live.open(exported["out_path"])
    _cells_once_there_are(client, exported["clip_count"])
    return exported


def _differences(got: dict, want: dict) -> list[str]:
    found = [
        f"{field} is {got[field]!r}, a prepared Set has {want[field]!r}"
        for field in ("name", "looping", "warp_mode", "color_index")
        if got[field] != want[field]
    ]
    if got["muted"]:
        found.append("the clip is deactivated")
    if Path(got["file_path"]).resolve() != Path(want["file_path"]).resolve():
        found.append(f"plays {got['file_path']}, not {want['file_path']}")
    edges = ("start_marker", "loop_start", "loop_end") if want["looping"] else ("start_marker",)
    found += [
        f"{edge} is {got[edge]:.3f}, a prepared Set has {want[edge]:.3f}"
        for edge in edges
        if abs(got[edge] - want[edge]) > 1e-3
    ]
    return found + _off_the_grid(got, clip_tempo(want))


def _off_the_grid(clip: dict, bpm: float) -> list[str]:
    first_beat, first_sec = clip["markers"][0]
    if abs(first_beat) > 1e-6 or abs(first_sec) > 1e-6:
        return [f"its first warp marker is at {clip['markers'][0]}"]
    if abs(clip_tempo(clip) - bpm) > 1e-3:
        return [f"it is warped at {clip_tempo(clip):.3f} BPM, the song is {bpm:.3f}"]
    return []


def _unlike_a_prepared_set(live: LiveStandIn, prepared: SetFile, side: str) -> list[str]:
    found = []
    for role in SONG_ROLES:
        deck_track = _deck_track(role, side)
        for row in range(PADS_PER_PAGE):
            if not live.has_clip(deck_track, row):
                found.append(f"{deck_track} row {row}: no clip")
                continue
            found += [
                f"{deck_track} row {row}: {difference}"
                for difference in _differences(
                    live.clip(deck_track, row), prepared.clip(deck_track, row)
                )
            ]
    return found


def test_a_song_loaded_during_the_set_gets_the_pads_a_prepared_set_would_give_it(booth):
    client, ids, live = booth
    first, second = ids["first_song"], ids["second_song"]
    prepared = SetFile(_exported(client, {"song": [first, second]})["out_path"])
    opened = _open_in_live(client, live, {"song": [first]})

    loaded = _load(client, track_id=second, side="b", page=0)

    assert (_landed(loaded), loaded["stems_loaded"], loaded["warnings"]) == ((0, "b", 0), 4, [])
    _settles(lambda: not _unlike_a_prepared_set(live, prepared, "b"))
    assert _unlike_a_prepared_set(live, prepared, "b") == []
    made_from_a_file = live.sent_to("/live/clip_slot/create_audio_clip")
    grid_points_written = live.sent_to("/live/clip/add_warp_marker")
    assert len(made_from_a_file) == len(SONG_ROLES)
    assert len(grid_points_written) == 2 * len(SONG_ROLES)

    cells = _cells(client)
    assert len(cells) == 2 * opened["clip_count"]
    for role in SONG_ROLES:
        pads = _pads(client, second, role)
        assert [pad["slot"] for pad in pads] == [1, 2, 3, 4, 5]
        for pad in pads:
            cell = cells[(pad["slot"] - 1, f"{role}_b")]
            assert cell["track_id"] == second
            assert _cue(cell) == (pad["slot"], pad["section_label"], pad["silent"])


def test_a_load_takes_the_next_page_with_room_and_replaces_nothing(booth):
    client, ids, live = booth
    first, second, third = ids["first_song"], ids["second_song"], ids["third_song"]
    _open_in_live(client, live, {"song": [first, second]})
    prepared_clips = live.clips()

    onto_a_full_deck = _load(client, track_id=third, side="a", page=0)
    assert _landed(onto_a_full_deck) == (1, "a", PADS_PER_PAGE)
    assert _settles(lambda: live.has_clip("Mix A", 2 * PADS_PER_PAGE - 1))
    assert live.scene_count == 2 * PADS_PER_PAGE

    no_deck_named = _load(client, track_id=first, page=0)
    assert _landed(no_deck_named) == (1, "b", PADS_PER_PAGE)

    one_stem = _load(client, track_id=second, kinds=["vocals"], side="a", page=0)
    assert (_landed(one_stem), one_stem["stems_loaded"]) == ((2, "a", 2 * PADS_PER_PAGE), 1)
    third_page = range(2 * PADS_PER_PAGE, 3 * PADS_PER_PAGE)
    assert _settles(lambda: all(live.has_clip("Vocals A", row) for row in third_page))
    assert {slot for slot in live.clips() if slot[1] in third_page} == {
        (live.track_index("Vocals A"), row) for row in third_page
    }

    live.put_clip(
        "Drums B",
        2 * PADS_PER_PAGE + 2,
        name="dropped in by hand",
        start_marker=0.0,
        file_path=_stem_path(client, third, "drums"),
    )
    beside_a_clip_the_app_never_saw = _load(
        client, track_id=third, kinds=["drums"], side="b", page=2
    )
    assert _landed(beside_a_clip_the_app_never_saw) == (3, "b", 3 * PADS_PER_PAGE)
    assert _settles(lambda: live.has_clip("Drums B", 4 * PADS_PER_PAGE - 1))

    assert live.sent_to("/live/clip_slot/delete_clip") == []
    assert live.sent_to("/live/clip_slot/stop") == []
    in_live_now = live.clips()
    assert {slot: in_live_now[slot] for slot in prepared_clips} == prepared_clips
    by_hand = in_live_now[(live.track_index("Drums B"), 2 * PADS_PER_PAGE + 2)]
    assert by_hand["name"] == "dropped in by hand"


def test_live_plays_the_songs_own_grid_when_its_auto_warp_is_wrong_or_late(booth):
    client, ids, live = booth
    first, second = ids["first_song"], ids["second_song"]
    prepared = SetFile(_exported(client, {"song": [first, second]})["out_path"])
    opened = _open_in_live(client, live, {"song": [first]})
    song_file = client.get(f"{API}/tracks/{second}").json()["file_path"]
    live.auto_warps(_stem_path(client, second, "drums"), HALF_TEMPO_FROM_A_LATE_FIRST_BEAT)
    live.auto_warps(
        _stem_path(client, second, "bass"), HALF_TEMPO_FROM_A_LATE_FIRST_BEAT, after=0.35
    )
    live.auto_warps(song_file, HALF_TEMPO_FROM_A_LATE_FIRST_BEAT, after=0.9)

    _load(client, track_id=second, side="b", page=0)
    time.sleep(GRID_GUARD_DELAYS[-1])

    _settles(lambda: not _unlike_a_prepared_set(live, prepared, "b"))
    assert _unlike_a_prepared_set(live, prepared, "b") == []

    live.rewarp("Vocals B", 3, HALF_TEMPO_FROM_A_LATE_FIRST_BEAT)
    checked = client.post(f"{API}/ableton/warp-check/0")
    assert checked.status_code == 200, checked.text
    assert (checked.json()["checked"], checked.json()["warnings"]) == (opened["clip_count"], [])
    _settles(lambda: not _unlike_a_prepared_set(live, prepared, "b"))
    assert _unlike_a_prepared_set(live, prepared, "b") == []


def test_the_check_says_when_live_no_longer_answers(booth):
    client, ids, live = booth
    first, second = ids["first_song"], ids["second_song"]
    opened = _open_in_live(client, live, {"song": [first]})
    _load(client, track_id=second, side="b", page=0)
    assert _settles(lambda: live.has_clip("Mix B", PADS_PER_PAGE - 1))

    live.stop_answering()
    checked = client.post(f"{API}/ableton/warp-check/0")

    assert checked.status_code == 200, checked.text
    assert checked.json()["checked"] == opened["clip_count"]
    (warning,) = checked.json()["warnings"]
    assert "did not report" in warning and f"{opened['clip_count']} pads" in warning


def test_a_song_with_no_pads_yet_loads_whole_on_its_own_grid_and_says_so(booth):
    client, ids, live = booth
    first, third = ids["first_song"], ids["third_song"]
    _open_in_live(client, live, {"song": [first]})
    session = get_session(get_settings().db_url)
    session.query(Region).filter(
        Region.track_id == third, Region.region_type == RegionType.PAD.value
    ).delete()
    session.commit()
    session.close()
    song_bpm = client.get(f"{API}/tracks/{third}").json()["analysis"]["bpm"]

    loaded = _load(client, track_id=third, side="b", page=0)

    assert (_landed(loaded), loaded["stems_loaded"]) == ((0, "b", 0), 4)
    assert len(loaded["warnings"]) == 1 and "pads" in loaded["warnings"][0]

    def whole_stems() -> list[dict]:
        return [live.clip(_deck_track(role, "b"), 0) for role in SONG_ROLES]

    assert _settles(lambda: all(live.has_clip(_deck_track(role, "b"), 0) for role in SONG_ROLES))
    assert _settles(lambda: not any(_off_the_grid(clip, song_bpm) for clip in whole_stems()))
    for role, clip in zip(SONG_ROLES, whole_stems(), strict=True):
        assert (clip["start_marker"], clip["looping"]) == (0.0, False)
        assert clip["warp_mode"] == WARP_MODE_BY_ROLE[role]
        assert not clip["muted"]
    below_the_first_row = range(1, PADS_PER_PAGE)
    assert not any(live.has_clip("Drums B", row) for row in below_the_first_row)

    cells = _cells(client)
    for role in SONG_ROLES:
        assert cells[(0, f"{role}_b")]["track_id"] == third
        assert _cue(cells[(0, f"{role}_b")]) == (None, None, False)
