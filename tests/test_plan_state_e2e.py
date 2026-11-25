from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from dance.api import create_app
from dance.config import Settings
from dance.core.database import Track, _reset_engine_for_tests, get_session, get_session_factory
from dance.pipeline.dispatcher import Dispatcher
from dance.pipeline.placeholders import is_placeholder_hash
from dance.spotify import csv_importer, download_pool
from tests.spotify_stand_in import APP_ID, APP_SECRET, CatalogTrack, SpotifyStandIn
from tests.test_acquisition import _FAKE_YT_DLP, _entry, needs_ffmpeg
from tests.test_api import FakeAbletonBridge
from tests.test_cue_pads_e2e import API, _heavy_models_faked
from tests.test_find_versions_e2e import YouTubeThatCanBeHeld, _as_mp3
from tests.test_versions_e2e import Song, build_library

SONG_SECONDS = 15.0
REMIX_UPLOAD = "TINLICKERrm"

LIBRARY = [
    Song("original", "Shallow Water", "Elderbrook"),
    Song("another_song", "Night Drive", "Mira"),
]
REMIX = CatalogTrack(
    "tinlicker-remix", "Shallow Water - Tinlicker Remix", ("Elderbrook", "Tinlicker"), SONG_SECONDS
)
WHAT_SPOTIFY_HAS = [
    CatalogTrack("original-release", "Shallow Water", ("Elderbrook",), SONG_SECONDS),
    REMIX,
]
WHAT_YOUTUBE_FINDS = {
    "Elderbrook Tinlicker Shallow Water - Tinlicker Remix": [
        _entry(
            REMIX_UPLOAD, "Elderbrook - Shallow Water (Tinlicker Remix)", "Anjunadeep", SONG_SECONDS
        )
    ]
}


class Booth:
    def __init__(
        self,
        client: TestClient,
        ids: dict[str, int],
        settings: Settings,
        youtube: YouTubeThatCanBeHeld,
    ) -> None:
        self.client = client
        self.ids = ids
        self.settings = settings
        self.youtube = youtube

    def new_set(self) -> int:
        created = self.client.post(f"{API}/sets", json={"name": "Friday night"})
        assert created.status_code == 200, created.text
        return created.json()["id"]

    def plan(self, set_id: int):
        return self.client.get(f"{API}/sets/{set_id}/plan")

    def replace_songs(self, set_id: int, track_ids: list[int]):
        return self.client.put(f"{API}/sets/{set_id}/plan", json={"queues": {"song": track_ids}})

    def queue_song(self, set_id: int, track_id: int, after: int | None = None):
        return self.client.post(
            f"{API}/sets/{set_id}/plan/append",
            json={"role": "song", "track_id": track_id, "after_track_id": after},
        )

    def get_the_remix(self) -> tuple[int, str]:
        asked = self.client.post(
            f"{API}/pipeline/ingest/track",
            json={
                "spotify_id": REMIX.spotify_id,
                "title": REMIX.name,
                "artist": ", ".join(REMIX.artists),
                "duration_ms": int(REMIX.seconds * 1000),
            },
        )
        assert asked.status_code == 200, asked.text
        return asked.json()["track_id"], asked.json()["job_id"]

    def download_ended(self, job_id: str) -> str:
        for _ in range(400):
            job = self.client.get(f"{API}/pipeline/jobs/{job_id}").json()
            if job["status"] in ("done", "error"):
                return job["items"][0]["status"]
            time.sleep(0.05)
        raise AssertionError(f"download job {job_id} never finished")

    def every_download_ended(self) -> None:
        self.youtube.answers.set()
        for job in self.client.get(f"{API}/pipeline/jobs").json():
            self.download_ended(job["id"])

    def audio_arrived(self, track_id: int) -> bool:
        with get_session_factory(self.settings.db_url)() as session:
            return not is_placeholder_hash(session.get(Track, track_id).file_hash)

    def as_find_versions_shows(self, track_id: int) -> tuple[str | None, bool]:
        found = self.client.get(
            f"{API}/spotify/versions", params={"track_id": self.ids["original"]}
        )
        assert found.status_code == 200, found.text
        (version,) = [v for v in found.json()["versions"] if v["library_track_id"] == track_id]
        return version["library_state"], version["download_failed"]

    def run_the_pipeline(self, separation=None) -> None:
        faked_separation, ensure_model, encode = _heavy_models_faked()
        session = get_session(self.settings.db_url)
        with separation or faked_separation, ensure_model, encode:
            Dispatcher(self.settings, session).run()
        session.close()


@pytest.fixture
def booth(monkeypatch, tmp_path):
    monkeypatch.setenv("DANCE_SPOTIFY_CLIENT_ID", APP_ID)
    monkeypatch.setenv("DANCE_SPOTIFY_CLIENT_SECRET", APP_SECRET)
    # Unset, the app looks for a cookie file in the real ~/.dance.
    monkeypatch.setenv("DANCE_YOUTUBE_COOKIES_FILE", str(tmp_path / "cookies.txt"))
    settings, ids = build_library(monkeypatch, LIBRARY)
    SpotifyStandIn(catalog=list(WHAT_SPOTIFY_HAS)).stands_in_for_spotify(monkeypatch)
    youtube = YouTubeThatCanBeHeld(dict(WHAT_YOUTUBE_FINDS), media={REMIX_UPLOAD: None})
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    monkeypatch.setattr(subprocess, "run", youtube)
    download_pool._reset_download_pool_for_tests()
    app = create_app(
        settings=settings,
        bridge=FakeAbletonBridge(),
        session_factory=get_session_factory(settings.db_url),
    )
    with TestClient(app) as client:
        booth = Booth(client, ids, settings, youtube)
        yield booth
        booth.every_download_ended()
    download_pool._reset_download_pool_for_tests()
    _reset_engine_for_tests()


@contextmanager
def _downloads_held(youtube: YouTubeThatCanBeHeld) -> Iterator[None]:
    youtube.answers.clear()
    try:
        yield
    finally:
        youtube.answers.set()


def _separation_runs_out_of_memory(self, session, track, settings) -> bool:
    raise RuntimeError("MPS backend out of memory")


def _songs(answer) -> list[tuple[int, str | None, bool]]:
    assert answer.status_code == 200, answer.text
    return [
        (pick["track_id"], pick["state"], pick["download_failed"])
        for pick in answer.json()["queues"]["song"]
    ]


def test_a_song_that_finished_the_pipeline_is_queued_as_complete(booth):
    original = booth.ids["original"]
    set_id = booth.new_set()

    planned = booth.replace_songs(set_id, [original])

    assert _songs(planned) == [(original, "complete", False)]
    assert _songs(booth.plan(set_id)) == [(original, "complete", False)]


@needs_ffmpeg
def test_a_pick_still_downloading_is_pending_and_has_not_failed(booth):
    original = booth.ids["original"]
    set_id = booth.new_set()
    booth.replace_songs(set_id, [original])

    with _downloads_held(booth.youtube):
        remix, _ = booth.get_the_remix()
        queued = booth.queue_song(set_id, remix, after=original)

        assert not booth.audio_arrived(remix)
        assert _songs(queued) == [(original, "complete", False), (remix, "pending", False)]
        assert _songs(booth.plan(set_id)) == _songs(queued)
        assert booth.as_find_versions_shows(remix) == ("pending", False)


@needs_ffmpeg
def test_a_pick_whose_download_failed_says_so_until_another_try_brings_the_audio(booth, tmp_path):
    set_id = booth.new_set()
    remix, first_try = booth.get_the_remix()
    assert booth.download_ended(first_try) == "fail"
    booth.queue_song(set_id, remix)

    assert not booth.audio_arrived(remix)
    assert _songs(booth.plan(set_id)) == [(remix, "error", True)]
    assert booth.as_find_versions_shows(remix) == ("error", True)

    booth.youtube.media[REMIX_UPLOAD] = _as_mp3(tmp_path / "remix-upload.mp3", seed=9)
    _, second_try = booth.get_the_remix()
    assert booth.download_ended(second_try) == "ok"

    assert booth.audio_arrived(remix)
    assert _songs(booth.plan(set_id)) == [(remix, "pending", False)]

    booth.run_the_pipeline()

    assert _songs(booth.plan(set_id)) == [(remix, "complete", False)]


@needs_ffmpeg
def test_a_pick_that_failed_in_the_pipeline_is_not_a_failed_download(booth, tmp_path):
    booth.youtube.media[REMIX_UPLOAD] = _as_mp3(tmp_path / "remix-upload.mp3", seed=9)
    set_id = booth.new_set()
    remix, job_id = booth.get_the_remix()
    assert booth.download_ended(job_id) == "ok"
    booth.queue_song(set_id, remix)

    booth.run_the_pipeline(
        separation=patch(
            "dance.pipeline.stages.separate.StemSeparationStage.process",
            new=_separation_runs_out_of_memory,
        )
    )

    assert booth.audio_arrived(remix)
    assert _songs(booth.plan(set_id)) == [(remix, "error", False)]
    assert booth.as_find_versions_shows(remix) == ("error", False)


def test_a_plan_that_names_a_deleted_track_still_answers(booth):
    original, deleted = booth.ids["original"], booth.ids["another_song"]
    set_id = booth.new_set()
    booth.replace_songs(set_id, [deleted, original])
    assert booth.client.delete(f"{API}/tracks/{deleted}").status_code == 204

    read = booth.plan(set_id)
    added_to = booth.queue_song(set_id, original)

    assert _songs(read) == [(deleted, None, False), (original, "complete", False)]
    assert _songs(added_to) == _songs(read)


@needs_ffmpeg
def test_replacing_and_adding_to_a_plan_answer_as_reading_it_does(booth):
    original = booth.ids["original"]
    set_id = booth.new_set()
    remix, job_id = booth.get_the_remix()
    assert booth.download_ended(job_id) == "fail"

    replaced = booth.replace_songs(set_id, [remix])

    assert _songs(replaced) == [(remix, "error", True)]
    assert replaced.json() == booth.plan(set_id).json()

    added_to = booth.queue_song(set_id, original, after=remix)

    assert _songs(added_to) == [(remix, "error", True), (original, "complete", False)]
    assert added_to.json() == booth.plan(set_id).json()
