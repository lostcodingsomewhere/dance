from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dance.api import create_app
from dance.config import Settings
from tests.test_api import FakeAbletonBridge
from tests.test_cue_pads_e2e import API

SUMMARY_FIELDS = {"id", "name", "notes", "is_active", "track_count", "created_at", "updated_at"}


@pytest.fixture
def api(session_factory, tmp_path):
    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def songs(session, make_track) -> list[int]:
    ids = [make_track().id for _ in range(4)]
    session.commit()
    return ids


def _new_set(api: TestClient, name: str) -> int:
    created = api.post(f"{API}/sets", json={"name": name})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _plan(api: TestClient, set_id: int, queues: dict[str, list[int]]) -> None:
    planned = api.put(f"{API}/sets/{set_id}/plan", json={"queues": queues})
    assert planned.status_code == 200, planned.text


def _picks_by_set_name(api: TestClient) -> dict[str, int]:
    listed = api.get(f"{API}/sets")
    assert listed.status_code == 200, listed.text
    return {row["name"]: row["track_count"] for row in listed.json()}


def test_the_set_list_counts_the_picks_in_each_plan(api, songs):
    first, second, third, fourth = songs
    friday = _new_set(api, "Friday night")
    _new_set(api, "Nothing planned")
    seven_picks = {
        "drums": [first, fourth],
        "bass": [second],
        "other": [third],
        "song": [first, second, third],
    }

    _plan(api, friday, seven_picks)

    assert _picks_by_set_name(api) == {"Friday night": 7, "Nothing planned": 0}
    assert all(set(row) == SUMMARY_FIELDS for row in api.get(f"{API}/sets").json())

    appended = api.post(
        f"{API}/sets/{friday}/plan/append", json={"role": "vocals", "track_id": fourth}
    )
    assert appended.status_code == 200, appended.text
    assert _picks_by_set_name(api)["Friday night"] == 8

    _plan(api, friday, {"song": [first]})
    assert _picks_by_set_name(api)["Friday night"] == 1


def test_a_row_of_the_old_set_tracks_table_is_not_a_pick(api, songs):
    friday = _new_set(api, "Friday night")
    old_row = api.post(f"{API}/sets/{friday}/tracks", json={"track_id": songs[0]})
    assert old_row.status_code == 200, old_row.text
    assert api.get(f"{API}/sets/{friday}/plan").json()["queues"]["song"] == []

    assert _picks_by_set_name(api) == {"Friday night": 0}
