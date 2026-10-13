"""One bounded download pool for every ingest path.

Every API ingest path queues on ONE shared pool sized by
``settings.download_workers`` (default 3). Before, the CSV job had a private
4-worker pool, single-track ingest spawned a thread, and playlist ingest
spawned a thread — and a yt-dlp process — per track: a 300-track playlist
meant 300 concurrent downloads against YouTube.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from dance.spotify import download_pool
from dance.spotify.download_pool import DownloadPool, get_download_pool


@pytest.fixture(autouse=True)
def _fresh_pool():
    download_pool._reset_download_pool_for_tests()
    yield
    download_pool._reset_download_pool_for_tests()


class _ConcurrencyProbe:
    """Callable that records how many copies of itself run at once."""

    def __init__(self, hold_s: float = 0.05) -> None:
        self.hold_s = hold_s
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._lock = threading.Lock()
        self.done = threading.Event()
        self.expected = 0

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.hold_s)
        with self._lock:
            self.active -= 1
            if self.expected and self.calls >= self.expected and self.active == 0:
                self.done.set()
        return ("ok", "probe")


# ---------------------------------------------------------------------------
# Shared download pool
# ---------------------------------------------------------------------------


def test_pool_never_exceeds_its_size() -> None:
    pool = DownloadPool(2)
    probe = _ConcurrencyProbe()
    futures = [pool.submit(probe, i) for i in range(8)]
    assert [f.result(timeout=5) for f in futures] == [("ok", "probe")] * 8
    assert probe.peak == 2
    pool.shutdown()


def test_pool_surfaces_exceptions_through_the_future() -> None:
    pool = DownloadPool(1)

    def boom() -> None:
        raise RuntimeError("yt-dlp exploded")

    fut = pool.submit(boom)
    with pytest.raises(RuntimeError, match="exploded"):
        fut.result(timeout=5)
    # The worker survives and keeps serving.
    assert pool.submit(lambda: 42).result(timeout=5) == 42
    pool.shutdown()


def test_pool_threads_are_daemons() -> None:
    """Stopping the API must never block on an in-flight download."""
    pool = DownloadPool(2, name="probe-pool")
    names = [t for t in threading.enumerate() if t.name.startswith("probe-pool")]
    assert len(names) == 2 and all(t.daemon for t in names)
    pool.shutdown()


def test_get_download_pool_is_one_shared_instance() -> None:
    a = get_download_pool(3)
    b = get_download_pool(5)  # a second size never spawns a second pool
    assert a is b and a.max_workers == 3


def test_csv_job_downloads_through_the_shared_pool(tmp_path, monkeypatch) -> None:
    from dance.api.jobs import get_job_registry
    from dance.api.routers import pipeline as pipeline_mod
    from dance.spotify.csv_importer import CsvRow

    probe = _ConcurrencyProbe()
    monkeypatch.setattr(pipeline_mod, "download_track", probe)
    rows = [CsvRow(artist=f"A{i}", title=f"T{i}", album="", duration_s=200) for i in range(6)]
    job = get_job_registry().create("csv_ingest", [f"A{i} - T{i}" for i in range(6)])

    pipeline_mod._run_csv_job(job.id, rows, tmp_path, None, "mp3", 2)

    assert probe.calls == 6
    assert probe.peak == 2
    done = get_job_registry().get(job.id)
    assert done is not None and done.status == "done"
    assert done.counts["ok"] == 6


@pytest.fixture
def api(session_factory, tmp_path):
    from dance.api import create_app
    from dance.config import Settings
    from tests.test_api import FakeAbletonBridge

    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        download_workers=2,
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory)
    with TestClient(app) as client:
        yield client


def _fake_playlist(monkeypatch, n: int) -> None:
    from dance.api.routers import spotify as spotify_router
    from dance.spotify import search as spotify_search
    from dance.spotify.search import SpotifyTrackHit

    hits = [
        SpotifyTrackHit(
            spotify_id=f"pl{i:03d}",
            title=f"Song {i}",
            artist=f"Act {i}",
            album=None,
            duration_ms=200_000 + i * 1000,
            preview_url=None,
            image_url=None,
            explicit=False,
            popularity=None,
        )
        for i in range(n)
    ]

    class _Client:
        configured = True

        def get_playlist_tracks(self, url, user_token=None):  # noqa: ARG002
            return list(hits)

    monkeypatch.setattr(spotify_search, "get_default_client", lambda _s: _Client())
    monkeypatch.setattr(spotify_router, "get_default_client", lambda _s: _Client())


def test_playlist_ingest_is_bounded_by_download_workers(api, monkeypatch) -> None:
    """Six new tracks, ``download_workers=2`` → never more than two at once
    (the old code started six threads immediately)."""
    from dance.api.routers import pipeline as pipeline_mod

    probe = _ConcurrencyProbe(hold_s=0.1)
    probe.expected = 6
    monkeypatch.setattr(pipeline_mod, "_run_spotify_ingest_job", probe)
    _fake_playlist(monkeypatch, 6)

    r = api.post(
        "/api/v1/pipeline/ingest/playlist",
        json={"playlist_url": "https://open.spotify.com/playlist/bounded1"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["added"] == 6
    assert probe.done.wait(5)
    assert probe.calls == 6
    assert probe.peak == 2


def test_single_track_ingest_queues_on_the_same_pool(api, monkeypatch) -> None:
    from dance.api.routers import pipeline as pipeline_mod

    seen: list[tuple] = []
    ran = threading.Event()

    def _job(*args, **kwargs):
        seen.append((threading.current_thread().name, args))
        ran.set()

    monkeypatch.setattr(pipeline_mod, "_run_spotify_ingest_job", _job)
    r = api.post(
        "/api/v1/pipeline/ingest/track",
        json={"spotify_id": "single01", "title": "Song", "artist": "Act", "duration_ms": 200_000},
    )
    assert r.status_code == 200, r.text
    assert ran.wait(5)
    ((thread_name, args),) = seen
    assert thread_name.startswith("dance-download")
    # audio_format + the claimed destination ride along to the worker.
    assert args[-2] == "mp3"
    assert args[-1].name == "Act - Song.mp3"
