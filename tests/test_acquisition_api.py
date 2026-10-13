"""The whole single-track ingest path, with only the yt-dlp binary faked.

POST /ingest/track -> shared download pool -> scored search -> download by
video id -> real ffprobe/ffmpeg verification -> tags -> Track row finalized.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dance.spotify import csv_importer
from tests.test_acquisition import (
    _FAKE_YT_DLP,
    FFMPEG,
    FakeYtDlp,
    _encode,
    _entry,
    _ffprobe,
    needs_ffmpeg,
)


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("e2e_audio")
    out = {
        "mp3_20": _encode(d / "s20.mp3", 20, "libmp3lame"),
        "mp3_60": _encode(d / "s60.mp3", 60, "libmp3lame"),
    }
    try:
        out["aac_60"] = _encode(d / "aac60.m4a", 60, "aac")
    except subprocess.CalledProcessError:
        pass
    return out


# ---------------------------------------------------------------------------
# Whole API path: POST /ingest/track -> pool -> acquire -> verify -> DB
# ---------------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["mp3", "m4a"])
def test_api_single_track_ingest_end_to_end(
    session_factory, tmp_path, monkeypatch, clips, fmt
) -> None:
    """Only the yt-dlp binary is faked. The endpoint creates the row, the
    shared pool runs the worker, the real scorer picks the right upload over
    a wrong-length top hit, real ffprobe/ffmpeg verify it, tags are written,
    and the Track row ends up pointing at the verified file."""
    import hashlib
    import time

    from fastapi.testclient import TestClient

    from dance.api import create_app
    from dance.config import Settings
    from dance.core.database import Track
    from dance.spotify import download_pool
    from tests.test_api import FakeAbletonBridge

    if fmt == "m4a" and "aac_60" not in clips:
        pytest.skip("this ffmpeg cannot write AAC")
    download_pool._reset_download_pool_for_tests()
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: _FAKE_YT_DLP)
    fake = FakeYtDlp(
        {
            "CamelPhat Arodes Cycles": [
                # Shaped like YouTube's real top hit for this query on
                # 2026-10-13: the 414 s cut the old code downloaded.
                _entry("R1vEXTENDED", "CamelPhat & Arodes - Cycles", "Obsessive Progressive", 414),
                _entry("RIGHTlength", "CamelPhat & Arodes - Cycles", "THE VOID", 61),
            ]
        },
        media={
            "R1vEXTENDED": clips["mp3_20"],
            "RIGHTlength": clips["mp3_60"] if fmt == "mp3" else clips["aac_60"],
        },
    )
    monkeypatch.setattr(subprocess, "run", fake)

    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        audio_format=fmt,
        youtube_cookies_file=tmp_path / "cookies.txt",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/api/v1/pipeline/ingest/track",
                json={
                    "spotify_id": "2bzCnD7nJuuS4yDNCAtLYz",
                    "title": "Cycles",
                    "artist": "CamelPhat, Arodes",
                    "duration_ms": 60_000,
                },
            )
            assert r.status_code == 200, r.text
            job_id, track_id = r.json()["job_id"], r.json()["track_id"]
            for _ in range(200):
                job = client.get(f"/api/v1/pipeline/jobs/{job_id}").json()
                if job["status"] in ("done", "error"):
                    break
                time.sleep(0.05)
    finally:
        download_pool._reset_download_pool_for_tests()

    assert job["status"] == "done", job
    (item,) = job["items"]
    assert item["status"] == "ok", item
    assert "yt:RIGHTlength score" in item["message"]
    # The wrong-length top hit was gated out before any download.
    assert fake.downloads == ["RIGHTlength"]

    dest = tmp_path / "library" / f"CamelPhat, Arodes - Cycles.{fmt}"
    with session_factory() as s:
        track = s.get(Track, track_id)
        assert track.file_path == str(dest)
        assert track.file_hash == hashlib.sha256(dest.read_bytes()).hexdigest()
        assert track.file_size_bytes == dest.stat().st_size
        assert track.state == "pending"
    probe = _ffprobe(dest)
    assert probe["streams"][0]["codec_name"] == ("mp3" if fmt == "mp3" else "aac")
    assert "watch?v=RIGHTlength" in probe["format"]["tags"]["comment"]


# ---------------------------------------------------------------------------
# /health/deps: ffprobe is required too (every download is verified with it)
# ---------------------------------------------------------------------------


def test_deps_check_requires_ffprobe(tmp_path, monkeypatch) -> None:
    from dance.api.app import _deps_status
    from dance.config import Settings

    settings = Settings(library_dir=tmp_path / "library", data_dir=tmp_path / "data")
    monkeypatch.setattr(csv_importer, "_resolve_ffmpeg", lambda _x: str(tmp_path / "ffmpeg"))
    (tmp_path / "ffmpeg").write_text("")
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: "/usr/bin/true")

    monkeypatch.setattr(csv_importer, "_resolve_ffprobe", lambda _ff: "/nope/ffprobe")
    report = _deps_status(settings)
    ff = next(c for c in report["checks"] if c["key"] == "ffmpeg")
    assert ff["status"] == "missing" and "ffprobe not found" in ff["detail"]
    assert report["ok"] is False

    (tmp_path / "ffprobe").write_text("")
    monkeypatch.setattr(csv_importer, "_resolve_ffprobe", lambda _ff: str(tmp_path / "ffprobe"))
    report = _deps_status(settings)
    ff = next(c for c in report["checks"] if c["key"] == "ffmpeg")
    assert ff["status"] == "ok" and "ffprobe" in ff["detail"]
    assert report["ok"] is True
