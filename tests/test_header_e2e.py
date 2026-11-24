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
from dance.osc.bridge import AbletonBridge
from tests.live_stand_in import LiveStandIn, booth_with_live, free_port
from tests.test_booth_pages_e2e import _cells_once_there_are, _exported, _settles
from tests.test_cue_pads_e2e import API

HEALTH = f"{API}/ableton/health"
STATE = f"{API}/ableton/state"
TRACK_COUNT_QUESTION = "/live/song/get/num_tracks"
PORT_NOTHING_CAN_BE_SENT_TO = 0
SECONDS_LIVE_TAKES_UNDER_A_LOAD = 0.5


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
