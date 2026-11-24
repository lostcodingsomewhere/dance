from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from dance.api import create_app
from dance.config import Settings
from dance.core.database import (
    Track,
    TrackState,
    _reset_engine_for_tests,
    get_session,
    get_session_factory,
)
from dance.pipeline.dispatcher import Dispatcher
from dance.spotify import csv_importer, download_pool
from dance.spotify import search as spotify_search
from tests.audio_fixtures import TrackSpec, synth_track
from tests.spotify_stand_in import APP_ID, APP_SECRET, CatalogTrack, SpotifyStandIn
from tests.test_acquisition import (
    _FAKE_YT_DLP,
    _REAL_RUN,
    FFMPEG,
    FakeYtDlp,
    _entry,
    _ffprobe,
    needs_ffmpeg,
)
from tests.test_api import FakeAbletonBridge
from tests.test_cue_pads_e2e import API, _heavy_models_faked
from tests.test_versions_e2e import Song, build_library

SONG_SECONDS = 15.0
LONG_CUT_SECONDS = 200.0
TRIBUTE_SECONDS = 120.0
ORIGINAL_UPLOAD = "ORIGINALup1"
REMIX_UPLOAD = "TINLICKERrm"

LIBRARY = [
    Song("original", "Shallow Water", "Elderbrook"),
    Song("club_mix", "Shallow Water - Club Mix", "Elderbrook"),
    Song("no_artist", "Untitled Jam", ""),
]


def _release_of_the_original(number: int) -> CatalogTrack:
    return CatalogTrack(
        f"original-release-{number}", "Shallow Water", ("Elderbrook",), SONG_SECONDS
    )


REMIX = CatalogTrack(
    "tinlicker-remix", "Shallow Water - Tinlicker Remix", ("Elderbrook", "Tinlicker"), SONG_SECONDS
)
WHAT_SPOTIFY_HAS = [
    _release_of_the_original(1),
    CatalogTrack("club-mix", "Shallow Water - Club Mix", ("Elderbrook",), SONG_SECONDS),
    *(_release_of_the_original(number) for number in range(2, 11)),
    REMIX,
    CatalogTrack(
        "tinlicker-remix-again", REMIX.name, REMIX.artists, SONG_SECONDS + 1.0, album="Remixes"
    ),
    CatalogTrack("long-cut", "Shallow Water", ("Elderbrook",), LONG_CUT_SECONDS),
    CatalogTrack(
        "extended",
        "Shallow Water (feat. Someone) - Extended Mix",
        ("Elderbrook", "Someone"),
        240.0,
    ),
    CatalogTrack(
        "edit-of-the-remix",
        "Shallow Water (Tinlicker Remix) - Tinlicker Remix - Edit",
        ("Elderbrook", "Tinlicker"),
        100.0,
    ),
    CatalogTrack("tribute", "Shallow Water", ("Elderbrook Tribute Band",), TRIBUTE_SECONDS),
    CatalogTrack("other-title", "Shallow Waters", ("Elderbrook",), SONG_SECONDS),
    CatalogTrack("unrelated", "Deep End", ("Elderbrook",), SONG_SECONDS),
]


def _as_mp3(path: Path, seed: int) -> Path:
    spec = TrackSpec(bpm=128.0, bars=8, seed=seed)
    wav = path.with_suffix(".wav")
    sf.write(str(wav), synth_track(spec), spec.sample_rate)
    _REAL_RUN(
        [FFMPEG, "-hide_banner", "-v", "error", "-y", "-i", str(wav), "-b:a", "128k", str(path)],
        check=True,
        capture_output=True,
    )
    return path


class YouTubeThatCanBeHeld(FakeYtDlp):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.answers = threading.Event()
        self.answers.set()

    def __call__(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if cmd[0] == _FAKE_YT_DLP:
            assert self.answers.wait(timeout=20)
        return super().__call__(cmd, **kwargs)


class Booth:
    def __init__(
        self, client: TestClient, ids: dict[str, int], spotify: SpotifyStandIn, settings: Settings
    ) -> None:
        self.client = client
        self.ids = ids
        self.spotify = spotify
        self.settings = settings

    def versions_on_spotify(self, track_id: int):
        return self.client.get(f"{API}/spotify/versions", params={"track_id": track_id})

    def listed(self, track_id: int) -> list[tuple[str, str, int | None, str | None]]:
        found = self.versions_on_spotify(track_id)
        assert found.status_code == 200, found.text
        return [
            (v["spotify_id"], v["label"], v["library_track_id"], v["library_state"])
            for v in found.json()["versions"]
        ]

    def finished(self, job_id: str) -> dict:
        for _ in range(400):
            job = self.client.get(f"{API}/pipeline/jobs/{job_id}").json()
            if job["status"] in ("done", "error"):
                return job
            time.sleep(0.05)
        raise AssertionError(f"download job {job_id} never finished")


@pytest.fixture
def booth(monkeypatch, tmp_path):
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    monkeypatch.setenv("DANCE_SPOTIFY_CLIENT_ID", APP_ID)
    monkeypatch.setenv("DANCE_SPOTIFY_CLIENT_SECRET", APP_SECRET)
    settings, ids = build_library(monkeypatch, LIBRARY)
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


@needs_ffmpeg
def test_a_remix_found_on_spotify_downloads_and_joins_the_songs_versions(
    booth, monkeypatch, tmp_path
):
    original, club_mix = booth.ids["original"], booth.ids["club_mix"]

    found = booth.versions_on_spotify(original)

    assert found.status_code == 200, found.text
    assert [
        (v["spotify_id"], v["label"], v["library_track_id"], v["library_state"])
        for v in found.json()["versions"]
    ] == [
        ("club-mix", "Club Mix", club_mix, "complete"),
        ("tinlicker-remix", "Tinlicker Remix", None, None),
        ("long-cut", "longer cut", None, None),
        ("extended", "Extended Mix", None, None),
        ("edit-of-the-remix", "Tinlicker Remix, Edit", None, None),
    ]
    remix = found.json()["versions"][1]
    assert (remix["title"], remix["artist"]) == (REMIX.name, "Elderbrook, Tinlicker")
    assert remix["duration_ms"] == int(SONG_SECONDS * 1000)
    assert remix["image_url"] == "https://img.test/tinlicker-remix"
    searches = booth.spotify.asked("/v1/search")
    assert [(s["q"], s["limit"], s.get("offset", "0")) for s in searches] == [
        ('track:"Shallow Water" artist:"Elderbrook"', "10", "0"),
        ('track:"Shallow Water" artist:"Elderbrook"', "10", "10"),
    ]

    youtube = FakeYtDlp(
        {
            "Elderbrook Tinlicker Shallow Water - Tinlicker Remix": [
                _entry(ORIGINAL_UPLOAD, "Elderbrook - Shallow Water", "Elderbrook", SONG_SECONDS),
                _entry(
                    REMIX_UPLOAD,
                    "Elderbrook - Shallow Water (Tinlicker Remix)",
                    "Anjunadeep",
                    SONG_SECONDS,
                ),
            ]
        },
        media={
            ORIGINAL_UPLOAD: _as_mp3(tmp_path / "original-upload.mp3", seed=1),
            REMIX_UPLOAD: _as_mp3(tmp_path / "remix-upload.mp3", seed=9),
        },
    )
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    monkeypatch.setattr(subprocess, "run", youtube)

    queued = booth.client.post(
        f"{API}/pipeline/ingest/track",
        json={
            "spotify_id": remix["spotify_id"],
            "title": remix["title"],
            "artist": remix["artist"],
            "duration_ms": remix["duration_ms"],
        },
    )
    assert queued.status_code == 200, queued.text
    remix_id = queued.json()["track_id"]
    job = booth.finished(queued.json()["job_id"])

    assert job["status"] == "done", job
    assert job["items"][0]["status"] == "ok", job
    assert youtube.downloads == [REMIX_UPLOAD]
    downloaded = (
        Path(booth.settings.library_dir)
        / "Elderbrook, Tinlicker - Shallow Water - Tinlicker Remix.mp3"
    )
    assert f"watch?v={REMIX_UPLOAD}" in _ffprobe(downloaded)["format"]["tags"]["comment"]
    assert booth.listed(original)[1] == ("tinlicker-remix", "Tinlicker Remix", remix_id, "pending")

    session = get_session(booth.settings.db_url)
    separate, ensure_model, encode = _heavy_models_faked()
    with separate, ensure_model, encode:
        dispatcher = Dispatcher(booth.settings, session)
        assert dispatcher.ingest()["new"] == 0
        dispatcher.run()
    remix_row = session.get(Track, remix_id)
    assert remix_row.state == TrackState.COMPLETE.value, remix_row.error_message
    session.close()

    versions = booth.client.get(f"{API}/tracks/{original}/versions").json()
    assert [(v["track_id"], v["label"]) for v in versions] == [
        (club_mix, "Club Mix"),
        (remix_id, "Tinlicker Remix"),
    ]
    assert booth.listed(original)[1] == ("tinlicker-remix", "Tinlicker Remix", remix_id, "complete")
    assert booth.listed(club_mix) == [
        ("original-release-1", "original", original, "complete"),
        ("tinlicker-remix", "Tinlicker Remix", remix_id, "complete"),
        ("long-cut", "original", None, None),
        ("extended", "Extended Mix", None, None),
        ("edit-of-the-remix", "Tinlicker Remix, Edit", None, None),
    ]


@needs_ffmpeg
def test_a_version_whose_download_failed_can_be_asked_for_again(booth, monkeypatch, tmp_path):
    original = booth.ids["original"]
    remix = booth.versions_on_spotify(original).json()["versions"][1]
    ask = {
        "spotify_id": remix["spotify_id"],
        "title": remix["title"],
        "artist": remix["artist"],
        "duration_ms": remix["duration_ms"],
    }
    upload = _entry(
        REMIX_UPLOAD, "Elderbrook - Shallow Water (Tinlicker Remix)", "Anjunadeep", SONG_SECONDS
    )
    youtube = YouTubeThatCanBeHeld(
        {"Elderbrook Tinlicker Shallow Water - Tinlicker Remix": [upload]},
        media={REMIX_UPLOAD: None},
    )
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    monkeypatch.setattr(subprocess, "run", youtube)

    first = booth.client.post(f"{API}/pipeline/ingest/track", json=ask).json()
    refused = booth.finished(first["job_id"])
    assert refused["items"][0]["status"] == "fail", refused
    remix_id = first["track_id"]
    assert booth.listed(original)[1] == ("tinlicker-remix", "Tinlicker Remix", remix_id, "error")
    assert booth.versions_on_spotify(original).json()["versions"][1]["download_failed"] is True

    youtube.media[REMIX_UPLOAD] = _as_mp3(tmp_path / "remix-upload.mp3", seed=9)
    youtube.answers.clear()
    second = booth.client.post(f"{API}/pipeline/ingest/track", json=ask).json()
    while_it_downloads = booth.client.post(f"{API}/pipeline/ingest/track", json=ask).json()
    youtube.answers.set()

    assert (second["track_id"], second["already_existed"]) == (remix_id, True)
    assert second["job_id"] is not None
    assert (while_it_downloads["state"], while_it_downloads["job_id"]) == ("pending", None)
    assert booth.finished(second["job_id"])["items"][0]["status"] == "ok"
    assert booth.listed(original)[1] == ("tinlicker-remix", "Tinlicker Remix", remix_id, "pending")
    assert not any(
        version["download_failed"]
        for version in booth.versions_on_spotify(original).json()["versions"]
    )
    with get_session_factory(booth.settings.db_url)() as session:
        row = session.get(Track, remix_id)
        assert (row.state, row.error_message) == (TrackState.PENDING.value, None)
        assert Path(row.file_path).exists()

    third = booth.client.post(f"{API}/pipeline/ingest/track", json=ask).json()
    assert (third["track_id"], third["job_id"]) == (remix_id, None)
    assert youtube.downloads == [REMIX_UPLOAD, REMIX_UPLOAD]


@needs_ffmpeg
def test_find_versions_says_what_stops_it(booth):
    original = booth.ids["original"]

    assert booth.versions_on_spotify(999_999).status_code == 404

    no_artist = booth.versions_on_spotify(booth.ids["no_artist"])
    assert no_artist.status_code == 422
    assert "artist" in no_artist.json()["detail"]
    assert booth.spotify.asked("/v1/search") == []

    booth.spotify.search_fails_with = 500
    spotify_down = booth.versions_on_spotify(original)
    assert spotify_down.status_code == 502
    assert "Spotify" in spotify_down.json()["detail"]

    booth.spotify.search_fails_with = None
    booth.spotify.catalog.clear()
    assert booth.listed(original) == []

    booth.settings.spotify_client_id = None
    booth.settings.spotify_client_secret = None
    spotify_search._reset_default_client_for_tests()
    not_set_up = booth.versions_on_spotify(original)
    assert not_set_up.status_code == 503
    assert "DANCE_SPOTIFY_CLIENT_ID" in not_set_up.json()["detail"]
