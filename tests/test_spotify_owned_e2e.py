from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from dance.api import create_app
from dance.core.database import Track, _reset_engine_for_tests, get_session_factory
from dance.spotify import csv_importer, download_pool
from tests.spotify_stand_in import APP_ID, APP_SECRET, CatalogTrack, SpotifyStandIn
from tests.test_acquisition import _FAKE_YT_DLP, FFMPEG, FakeYtDlp, _entry, needs_ffmpeg
from tests.test_api import FakeAbletonBridge
from tests.test_cue_pads_e2e import API
from tests.test_find_versions_e2e import Booth, _as_mp3
from tests.test_versions_e2e import Song, build_library

SONG_SECONDS = 15.0
LONG_CUT_SECONDS = 200.0
REMIX_UPLOAD = "TINLICKERrm"
LONG_CUT_UPLOAD = "LONGCUTup01"

ONE_SONG = [Song("owned", "Shallow Water", "Elderbrook")]
TWO_COPIES = [
    Song("one_copy", "Shallow Water", "Elderbrook"),
    Song("another_copy", "Shallow Water", "Elderbrook"),
]

THE_SONG = CatalogTrack("the-song", "Shallow Water", ("Elderbrook",), SONG_SECONDS)
ITS_ALBUM_RELEASE = CatalogTrack(
    "the-song-on-the-album", "Shallow Water", ("Elderbrook",), SONG_SECONDS + 1.0, album="Album"
)
REMIX = CatalogTrack(
    "tinlicker-remix", "Shallow Water - Tinlicker Remix", ("Elderbrook", "Tinlicker"), SONG_SECONDS
)
REMIX_ON_ANOTHER_RELEASE = CatalogTrack(
    "tinlicker-remix-again", REMIX.name, REMIX.artists, SONG_SECONDS + 1.0, album="Remixes"
)
EXTENDED = CatalogTrack("extended", "Shallow Water - Extended Mix", ("Elderbrook",), 240.0)
LONG_CUT = CatalogTrack("long-cut", "Shallow Water", ("Elderbrook",), LONG_CUT_SECONDS)
ANOTHER_BANDS = CatalogTrack("another-bands", "Shallow Water", ("Other Band",), SONG_SECONDS)
NOT_OWNED = CatalogTrack("not-owned", "Deep End", ("Elderbrook",), SONG_SECONDS)
WHAT_SPOTIFY_HAS = [
    THE_SONG,
    ITS_ALBUM_RELEASE,
    REMIX,
    REMIX_ON_ANOTHER_RELEASE,
    EXTENDED,
    LONG_CUT,
    ANOTHER_BANDS,
    NOT_OWNED,
]


def _booth(monkeypatch, songs: list[Song]):
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    monkeypatch.setenv("DANCE_SPOTIFY_CLIENT_ID", APP_ID)
    monkeypatch.setenv("DANCE_SPOTIFY_CLIENT_SECRET", APP_SECRET)
    settings, ids = build_library(monkeypatch, songs)
    spotify = SpotifyStandIn(catalog=list(WHAT_SPOTIFY_HAS))
    spotify.stands_in_for_spotify(monkeypatch)
    download_pool._reset_download_pool_for_tests()
    app = create_app(
        settings=settings,
        bridge=FakeAbletonBridge(),
        session_factory=get_session_factory(settings.db_url),
    )
    with TestClient(app) as client:
        yield Booth(client, ids, spotify, settings)
    download_pool._reset_download_pool_for_tests()
    _reset_engine_for_tests()


@pytest.fixture
def booth(monkeypatch):
    yield from _booth(monkeypatch, ONE_SONG)


@pytest.fixture
def booth_with_a_marked_duplicate(monkeypatch):
    for booth in _booth(monkeypatch, TWO_COPIES):
        added_first, added_second = sorted(booth.ids.values())
        booth.ids = {"marked_duplicate": added_first, "kept": added_second}
        with get_session_factory(booth.settings.db_url)() as session:
            duplicate = session.get(Track, added_first)
            duplicate.duplicate_of = added_second
            duplicate.spotify_id = THE_SONG.spotify_id
            session.commit()
        yield booth


def _youtube(monkeypatch, results: dict[str, list[dict]], media: dict | None = None) -> FakeYtDlp:
    youtube = FakeYtDlp(results, media=media)
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    monkeypatch.setattr(subprocess, "run", youtube)
    return youtube


def _searched(booth: Booth, query: str) -> list[tuple[str, int | None, str | None]]:
    found = booth.client.get(f"{API}/spotify/search", params={"q": query, "limit": 20})
    assert found.status_code == 200, found.text
    return [
        (hit["spotify_id"], hit["library_track_id"], hit["library_state"])
        for hit in found.json()["hits"]
    ]


def _asked_for(booth: Booth, song: CatalogTrack, only_the_id: bool = False) -> dict:
    wanted: dict = {"spotify_id": song.spotify_id}
    if not only_the_id:
        wanted |= {
            "title": song.name,
            "artist": ", ".join(song.artists),
            "duration_ms": int(song.seconds * 1000),
        }
    queued = booth.client.post(f"{API}/pipeline/ingest/track", json=wanted)
    assert queued.status_code == 200, queued.text
    return queued.json()


def _rows(booth: Booth) -> list[tuple[int, str | None]]:
    with get_session_factory(booth.settings.db_url)() as session:
        tracks = session.query(Track).order_by(Track.id)
        return [(int(track.id), track.spotify_id) for track in tracks]


def _download_jobs(booth: Booth) -> list[str]:
    jobs = booth.client.get(f"{API}/pipeline/jobs", params={"limit": 100}).json()
    return [job["id"] for job in jobs if job["kind"] == "spotify_ingest"]


@needs_ffmpeg
def test_a_search_marks_the_song_the_library_holds_and_none_of_its_other_versions(booth):
    owned = booth.ids["owned"]
    assert _rows(booth) == [(owned, None)]

    assert _searched(booth, "shallow water") == [
        (THE_SONG.spotify_id, owned, "complete"),
        (ITS_ALBUM_RELEASE.spotify_id, owned, "complete"),
        (REMIX.spotify_id, None, None),
        (REMIX_ON_ANOTHER_RELEASE.spotify_id, None, None),
        (EXTENDED.spotify_id, None, None),
        (LONG_CUT.spotify_id, None, None),
        (ANOTHER_BANDS.spotify_id, None, None),
    ]
    assert _searched(booth, "deep end") == [(NOT_OWNED.spotify_id, None, None)]


@needs_ffmpeg
def test_asking_for_a_song_the_library_holds_answers_with_it_and_downloads_nothing(
    booth, monkeypatch
):
    owned = booth.ids["owned"]
    youtube = _youtube(monkeypatch, {})
    jobs_before = _download_jobs(booth)

    for answer in (
        _asked_for(booth, THE_SONG),
        _asked_for(booth, ITS_ALBUM_RELEASE),
        _asked_for(booth, THE_SONG, only_the_id=True),
    ):
        assert answer == {
            "track_id": owned,
            "state": "complete",
            "job_id": None,
            "already_existed": True,
        }

    assert _rows(booth) == [(owned, None)]
    assert _download_jobs(booth) == jobs_before
    assert (youtube.search_cmds, youtube.download_cmds) == ([], [])
    assert len(booth.spotify.asked(f"/v1/tracks/{THE_SONG.spotify_id}")) == 1


@needs_ffmpeg
def test_asking_for_a_remix_or_another_cut_of_a_held_song_still_downloads_it(
    booth, monkeypatch, tmp_path
):
    owned = booth.ids["owned"]
    youtube = _youtube(
        monkeypatch,
        {
            "Elderbrook Tinlicker Shallow Water - Tinlicker Remix": [
                _entry(
                    REMIX_UPLOAD,
                    "Elderbrook - Shallow Water (Tinlicker Remix)",
                    "Anjunadeep",
                    SONG_SECONDS,
                )
            ],
            "Elderbrook Shallow Water": [
                _entry(
                    LONG_CUT_UPLOAD, "Elderbrook - Shallow Water", "Elderbrook", LONG_CUT_SECONDS
                )
            ],
        },
        media={
            REMIX_UPLOAD: _as_mp3(tmp_path / "remix-upload.mp3", seed=9),
            LONG_CUT_UPLOAD: None,
        },
    )

    remix = _asked_for(booth, REMIX)
    assert booth.finished(remix["job_id"])["items"][0]["status"] == "ok"
    long_cut = _asked_for(booth, LONG_CUT)
    booth.finished(long_cut["job_id"])

    assert (remix["already_existed"], long_cut["already_existed"]) == (False, False)
    assert _rows(booth) == [
        (owned, None),
        (remix["track_id"], REMIX.spotify_id),
        (long_cut["track_id"], LONG_CUT.spotify_id),
    ]
    assert youtube.downloads == [REMIX_UPLOAD, LONG_CUT_UPLOAD]
    assert _searched(booth, "shallow water tinlicker") == [
        (REMIX.spotify_id, remix["track_id"], "pending"),
        (REMIX_ON_ANOTHER_RELEASE.spotify_id, remix["track_id"], "pending"),
    ]


@needs_ffmpeg
def test_a_song_held_as_a_marked_duplicate_answers_with_the_track_that_was_kept(
    booth_with_a_marked_duplicate, monkeypatch
):
    booth = booth_with_a_marked_duplicate
    kept, duplicate = booth.ids["kept"], booth.ids["marked_duplicate"]
    youtube = _youtube(monkeypatch, {})

    assert _searched(booth, "shallow water")[:2] == [
        (THE_SONG.spotify_id, kept, "complete"),
        (ITS_ALBUM_RELEASE.spotify_id, kept, "complete"),
    ]
    for song in (THE_SONG, ITS_ALBUM_RELEASE):
        answer = _asked_for(booth, song)
        assert (answer["track_id"], answer["already_existed"], answer["job_id"]) == (
            kept,
            True,
            None,
        )
    assert _asked_for(booth, THE_SONG, only_the_id=True)["track_id"] == kept

    assert _rows(booth) == [(duplicate, THE_SONG.spotify_id), (kept, None)]
    assert (youtube.search_cmds, youtube.download_cmds) == ([], [])


@needs_ffmpeg
def test_asking_for_another_release_of_a_failed_download_tries_that_download_again(
    booth, monkeypatch, tmp_path
):
    upload = _entry(
        REMIX_UPLOAD, "Elderbrook - Shallow Water (Tinlicker Remix)", "Anjunadeep", SONG_SECONDS
    )
    youtube = _youtube(
        monkeypatch,
        {"Elderbrook Tinlicker Shallow Water - Tinlicker Remix": [upload]},
        media={REMIX_UPLOAD: None},
    )
    first = _asked_for(booth, REMIX)
    assert booth.finished(first["job_id"])["items"][0]["status"] == "fail"
    assert _searched(booth, "shallow water tinlicker") == [
        (REMIX.spotify_id, first["track_id"], "error"),
        (REMIX_ON_ANOTHER_RELEASE.spotify_id, first["track_id"], "error"),
    ]

    youtube.media[REMIX_UPLOAD] = _as_mp3(tmp_path / "remix-upload.mp3", seed=9)
    again = _asked_for(booth, REMIX_ON_ANOTHER_RELEASE)

    assert (again["track_id"], again["already_existed"]) == (first["track_id"], True)
    assert again["job_id"] is not None
    assert booth.finished(again["job_id"])["items"][0]["status"] == "ok"
    assert _rows(booth) == [(booth.ids["owned"], None), (first["track_id"], REMIX.spotify_id)]
    assert youtube.downloads == [REMIX_UPLOAD, REMIX_UPLOAD]
