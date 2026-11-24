from __future__ import annotations

import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from dance.api import create_app
from dance.config import Settings
from dance.core.database import Track
from dance.spotify import csv_importer, download_pool
from tests.spotify_stand_in import (
    APP_ID,
    APP_SECRET,
    APP_TOKEN,
    PLAYLIST_LIMIT_MAX,
    SEARCH_LIMIT_MAX,
    CatalogTrack,
    Playlist,
    SpotifyStandIn,
)
from tests.test_acquisition import _FAKE_YT_DLP, FakeYtDlp
from tests.test_api import FakeAbletonBridge
from tests.test_cue_pads_e2e import API

PLAYLIST_ID = "testplaylist0000000001"
PLAYLIST_URL = f"https://open.spotify.com/playlist/{PLAYLIST_ID}?si=abc"
OWNERS_TOKEN = "token-of-the-owner"
SOMEONE_ELSES_TOKEN = "token-of-someone-else"

IN_THE_PLAYLIST = [
    CatalogTrack("song-one", "Night Drive", ("Mira",), 201.0, album="Night Drive EP"),
    CatalogTrack("song-two", "Half Light", ("Mira", "Oda"), 240.5),
    CatalogTrack("song-three", "Low Tide - Extended Mix", ("Oda",), 421.0),
]
A_PODCAST_EPISODE = CatalogTrack("episode-one", "Episode 12", ("A Show",), 1800.0, kind="episode")
A_LOCAL_FILE = None


@pytest.fixture
def spotify(monkeypatch) -> SpotifyStandIn:
    stand_in = SpotifyStandIn(
        catalog=[
            CatalogTrack(f"strobe-{n}", f"Strobe {n}", ("deadmau5",), 300.0 + n) for n in range(17)
        ],
        playlists={
            PLAYLIST_ID: Playlist(
                owner="the-owner",
                entries=[
                    IN_THE_PLAYLIST[0],
                    A_PODCAST_EPISODE,
                    IN_THE_PLAYLIST[1],
                    A_LOCAL_FILE,
                    IN_THE_PLAYLIST[2],
                    IN_THE_PLAYLIST[0],
                ],
            )
        },
        user_of_token={OWNERS_TOKEN: "the-owner", SOMEONE_ELSES_TOKEN: "someone-else"},
        most_playlist_entries_a_page=2,
    )
    stand_in.stands_in_for_spotify(monkeypatch)
    return stand_in


@pytest.fixture
def api(spotify, session_factory, tmp_path, monkeypatch):
    youtube_finds_nothing = FakeYtDlp({})
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    monkeypatch.setattr(subprocess, "run", youtube_finds_nothing)
    download_pool._reset_download_pool_for_tests()
    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        spotify_client_id=APP_ID,
        spotify_client_secret=APP_SECRET,
        youtube_cookies_file=tmp_path / "cookies.txt",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory)
    with TestClient(app) as client:
        yield client
    download_pool._reset_download_pool_for_tests()


def _read_playlist(api: TestClient, token: str | None):
    return api.post(
        f"{API}/pipeline/ingest/playlist",
        json={"playlist_url": PLAYLIST_URL, "user_token": token},
    )


def _downloads_finished(api: TestClient, job_ids: list[str]) -> None:
    for _ in range(400):
        jobs = [api.get(f"{API}/pipeline/jobs/{job_id}").json() for job_id in job_ids]
        if all(job["status"] in ("done", "error") for job in jobs):
            return
        time.sleep(0.05)
    raise AssertionError("download jobs never finished")


def test_a_search_for_more_than_spotify_gives_at_once_reads_several_pages(api, spotify):
    found = api.get(f"{API}/spotify/search", params={"q": "strobe deadmau5", "limit": 20})

    assert found.status_code == 200, found.text
    assert [hit["spotify_id"] for hit in found.json()["hits"]] == [
        f"strobe-{n}" for n in range(17)
    ]
    asked = spotify.asked("/v1/search")
    assert [(a["limit"], a.get("offset", "0")) for a in asked] == [("10", "0"), ("10", "10")]
    assert all(int(a["limit"]) <= SEARCH_LIMIT_MAX for a in asked)

    spotify.requests.clear()
    few = api.get(f"{API}/spotify/search", params={"q": "strobe deadmau5", "limit": 5})
    assert [hit["title"] for hit in few.json()["hits"]] == [f"Strobe {n}" for n in range(5)]
    assert len(spotify.asked("/v1/search")) == 1


def test_a_search_gets_past_an_error_spotify_says_to_retry(api, spotify):
    spotify.passing_search_failures = 2

    found = api.get(f"{API}/spotify/search", params={"q": "strobe deadmau5", "limit": 5})

    assert found.status_code == 200, found.text
    assert len(found.json()["hits"]) == 5
    assert len(spotify.asked("/v1/search")) == 3

    spotify.search_fails_with = 503
    still_down = api.get(f"{API}/spotify/search", params={"q": "strobe deadmau5", "limit": 5})
    assert still_down.status_code == 502


def test_the_owners_token_reads_a_playlist_and_queues_its_songs(api, spotify, session_factory):
    read = _read_playlist(api, OWNERS_TOKEN)

    assert read.status_code == 200, read.text
    body = read.json()
    assert (body["total_in_playlist"], body["added"], body["skipped_existing"]) == (3, 3, 0)
    _downloads_finished(api, body["job_ids"])
    with session_factory() as session:
        queued = {
            track.spotify_id: (track.title, track.artist, track.duration_seconds)
            for track in session.query(Track)
        }
    assert queued == {
        "song-one": ("Night Drive", "Mira", 201.0),
        "song-two": ("Half Light", "Mira, Oda", 240.5),
        "song-three": ("Low Tide - Extended Mix", "Oda", 421.0),
    }
    pages = [request for request in spotify.requests if "/v1/playlists/" in request.url.path]
    assert [request.url.path for request in pages] == [f"/v1/playlists/{PLAYLIST_ID}/items"] * 3
    assert all(request.headers["Authorization"] == f"Bearer {OWNERS_TOKEN}" for request in pages)
    assert int(pages[0].url.params["limit"]) <= PLAYLIST_LIMIT_MAX

    again = _read_playlist(api, OWNERS_TOKEN).json()
    assert (again["added"], again["skipped_existing"]) == (0, 3)


def test_a_playlist_read_says_what_spotify_needs(api, spotify, session_factory):
    without_a_user = _read_playlist(api, None)
    assert without_a_user.status_code == 401
    assert "user token" in without_a_user.json()["detail"]
    assert not any(
        request.headers.get("Authorization") == f"Bearer {APP_TOKEN}"
        and "/v1/playlists/" in request.url.path
        for request in spotify.requests
    )

    not_the_owner = _read_playlist(api, SOMEONE_ELSES_TOKEN)
    assert not_the_owner.status_code == 401
    assert "owns" in not_the_owner.json()["detail"]
    assert 'Spotify said "Forbidden"' in not_the_owner.json()["detail"]

    expired = _read_playlist(api, "a-token-spotify-no-longer-knows")
    assert expired.status_code == 401
    assert "expired" in expired.json()["detail"]

    unknown = api.post(
        f"{API}/pipeline/ingest/playlist",
        json={
            "playlist_url": "https://open.spotify.com/playlist/0000000000000000000000",
            "user_token": OWNERS_TOKEN,
        },
    )
    assert unknown.status_code == 502
    assert "not found" in unknown.json()["detail"]
    with session_factory() as session:
        assert session.query(Track).count() == 0
