from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dance.api import create_app
from dance.config import get_settings
from dance.core.database import _reset_engine_for_tests, get_session_factory
from dance.osc.bridge import AbletonBridge
from tests.live_stand_in import GRID_GUARD_DELAYS, LiveStandIn, booth_with_live, free_port
from tests.test_booth_pages_e2e import _cells_once_there_are, _exported, _settles
from tests.test_cue_pads_e2e import API
from tests.test_export_set_e2e import three_songs_through_the_pipeline

HEALTH = f"{API}/ableton/health"
STATE = f"{API}/ableton/state"
TRACK_COUNT_QUESTION = "/live/song/get/num_tracks"
MAKE_A_CLIP = "/live/clip_slot/create_audio_clip"
PORT_NOTHING_CAN_BE_SENT_TO = 0
SECONDS_LIVE_TAKES_UNDER_A_LOAD = 0.5
SECONDS_LIVE_TAKES_TO_MAKE_A_CLIP = 0.5
SECONDS_OF_SILENCE_A_LOAD_EXPLAINS = 0.5


@contextmanager
def _api_sending_to(live_port: int, reply_port: int, session_factory) -> Iterator[TestClient]:
    settings = get_settings()
    settings.ensure_directories()
    bridge = AbletonBridge(send_port=live_port, listen_port=reply_port)
    app = create_app(settings=settings, bridge=bridge, session_factory=session_factory)
    bridge.start()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        bridge.stop()


@pytest.fixture
def booth(session_factory):
    with LiveStandIn() as live:
        with _api_sending_to(live.receive_port, live.reply_port, session_factory) as client:
            yield client, live


@pytest.fixture
def booth_with_songs(monkeypatch):
    yield from booth_with_live(monkeypatch)


@pytest.fixture
def booth_with_little_patience(monkeypatch):
    track_ids = three_songs_through_the_pipeline(monkeypatch)
    settings = get_settings()
    with LiveStandIn() as live:
        bridge = AbletonBridge(
            send_port=live.receive_port,
            listen_port=live.reply_port,
            state_file=settings.data_dir / "deck_state.json",
            grid_guard_delays=GRID_GUARD_DELAYS,
            seconds_to_wait_for_new_clips=SECONDS_OF_SILENCE_A_LOAD_EXPLAINS,
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


def _says_when_it_starts_making_a_clip(live: LiveStandIn, monkeypatch) -> threading.Event:
    started = threading.Event()
    answer = live._answer

    def answer_and_say(address: str, args: tuple[Any, ...]) -> None:
        if address == MAKE_A_CLIP:
            started.set()
        answer(address, args)

    monkeypatch.setattr(live, "_answer", answer_and_say)
    return started


def _goes_quiet_at_its_next_track_count(live: LiveStandIn, monkeypatch) -> threading.Event:
    quiet = threading.Event()
    answer = live._answer

    def answer_until_asked(address: str, args: tuple[Any, ...]) -> None:
        if address == TRACK_COUNT_QUESTION:
            live.stop_answering()
            quiet.set()
        answer(address, args)

    monkeypatch.setattr(live, "_answer", answer_until_asked)
    return quiet


def _answers_its_next_track_count_late(
    live: LiveStandIn, monkeypatch, seconds: float
) -> threading.Event:
    asked = threading.Event()
    answer = live._answer

    def answer_the_first_one_late(address: str, args: tuple[Any, ...]) -> None:
        if address == TRACK_COUNT_QUESTION and not asked.is_set():
            asked.set()
            time.sleep(seconds)
        answer(address, args)

    monkeypatch.setattr(live, "_answer", answer_the_first_one_late)
    return asked


def test_the_health_check_says_live_is_reachable_while_live_answers(booth):
    client, live = booth

    checked = client.get(HEALTH)

    assert checked.status_code == 200, checked.text
    assert checked.json() == {"live_reachable": True}
    assert live.sent_to(TRACK_COUNT_QUESTION) != []


def test_the_health_check_says_so_once_live_stops_answering(booth):
    client, live = booth
    assert client.get(HEALTH).json() == {"live_reachable": True}
    live.change_tempo(131.5)
    assert _settles(lambda: client.get(STATE).json()["tempo"] == 131.5)

    live.stop_answering()
    checked = client.get(HEALTH)

    assert checked.status_code == 200, checked.text
    assert checked.json() == {"live_reachable": False}
    last_known_state = client.get(STATE)
    assert last_known_state.status_code == 200
    assert last_known_state.json()["tempo"] == 131.5


@pytest.mark.parametrize(
    "live_port",
    [free_port, lambda: PORT_NOTHING_CAN_BE_SENT_TO],
    ids=["live is not running", "nothing can be sent to live"],
)
def test_the_health_check_says_live_is_unreachable_without_live(
    live_port: Callable[[], int], session_factory
):
    with _api_sending_to(live_port(), free_port(), session_factory) as client:
        checked = client.get(HEALTH)

    assert checked.status_code == 200, checked.text
    assert checked.json() == {"live_reachable": False}


def test_a_health_check_leaves_a_load_the_answer_it_is_waiting_for(booth_with_songs, monkeypatch):
    client, ids, live = booth_with_songs
    exported = _exported(client, {"song": [ids["first_song"]]})
    live.open(exported["out_path"])
    _cells_once_there_are(client, exported["clip_count"])
    load_is_waiting_for_live = _answers_its_next_track_count_late(
        live, monkeypatch, SECONDS_LIVE_TAKES_UNDER_A_LOAD
    )

    with ThreadPoolExecutor(max_workers=1) as beside:
        load = beside.submit(
            client.post,
            f"{API}/ableton/load-track",
            json={"track_id": ids["second_song"], "side": "b", "page": 0},
        )
        assert load_is_waiting_for_live.wait(timeout=5.0)
        checked = client.get(HEALTH)

    assert checked.json() == {"live_reachable": True}
    assert load.result().status_code == 200, load.result().text
    loaded = load.result().json()
    assert (loaded["stems_loaded"], loaded["warnings"]) == (4, [])


def test_the_health_check_trusts_live_while_it_is_silent_making_the_clips_of_a_load(
    booth_with_songs, monkeypatch
):
    client, ids, live = booth_with_songs
    exported = _exported(client, {"song": [ids["first_song"]]})
    live.open(exported["out_path"])
    _cells_once_there_are(client, exported["clip_count"])
    live.takes_to_make_a_clip(SECONDS_LIVE_TAKES_TO_MAKE_A_CLIP)
    live_is_making_clips = _says_when_it_starts_making_a_clip(live, monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as beside:
        load = beside.submit(
            client.post,
            f"{API}/ableton/load-track",
            json={"track_id": ids["second_song"], "side": "b", "page": 0},
        )
        assert live_is_making_clips.wait(timeout=5.0)
        checked = client.get(HEALTH)
        live_was_still_silent = not load.done()

    assert live_was_still_silent
    assert checked.json() == {"live_reachable": True}
    assert load.result().status_code == 200, load.result().text
    loaded = load.result().json()
    assert (loaded["stems_loaded"], loaded["warnings"]) == (4, [])


def test_a_load_stops_vouching_for_a_live_that_has_been_silent_too_long(
    booth_with_little_patience, monkeypatch
):
    client, ids, live = booth_with_little_patience
    exported = _exported(client, {"song": [ids["first_song"]]})
    live.open(exported["out_path"])
    _cells_once_there_are(client, exported["clip_count"])
    live_went_quiet = _goes_quiet_at_its_next_track_count(live, monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as beside:
        load = beside.submit(
            client.post,
            f"{API}/ableton/load-track",
            json={"track_id": ids["second_song"], "side": "b", "page": 0},
        )
        assert live_went_quiet.wait(timeout=5.0)
        checked = client.get(HEALTH)
        load_was_still_waiting = not load.done()

    assert load_was_still_waiting
    assert checked.json() == {"live_reachable": False}
    assert load.result().json()["stems_loaded"] == 0
