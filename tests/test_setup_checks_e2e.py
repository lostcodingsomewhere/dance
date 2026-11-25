from __future__ import annotations

import json
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dance import config
from dance.api import create_app
from dance.config import get_settings
from dance.core.database import TrackState
from dance.spotify import csv_importer
from tests.test_api import FakeAbletonBridge

SETUP = "/api/v1/health/deps"
DAYS_BEFORE_YT_DLP_IS_OLD = 90


def _yt_dlp_released(tmp_path: Path, days_ago: int) -> Path:
    released = date.today() - timedelta(days=days_ago)
    stand_in = tmp_path / f"yt-dlp-{days_ago}"
    stand_in.write_text(f"#!/bin/sh\necho {released:%Y.%m.%d}\n")
    stand_in.chmod(stand_in.stat().st_mode | stat.S_IXUSR)
    return stand_in


@pytest.fixture
def server_started(session_factory) -> Callable[[], contextmanager]:
    @contextmanager
    def start() -> Iterator[TestClient]:
        config._settings = None
        settings = get_settings()
        settings.ensure_directories()
        app = create_app(
            settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory
        )
        with TestClient(app) as client:
            yield client

    return start


@pytest.fixture
def setup_checks(server_started) -> Iterator[TestClient]:
    with server_started() as client:
        yield client


def _check(client: TestClient, key: str) -> dict:
    answer = client.get(SETUP)
    assert answer.status_code == 200, answer.text
    return next(check for check in answer.json()["checks"] if check["key"] == key)


def test_cookies_are_read_from_the_data_folder_and_nowhere_else(
    server_started, monkeypatch, tmp_path
):
    home = tmp_path / "someone_else"
    (home / ".dance").mkdir(parents=True)
    (home / ".dance" / "cookies.txt").write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("HOME", str(home))

    with server_started() as client:
        assert _check(client, "cookies")["status"] == "optional"
        in_the_data_folder = get_settings().data_dir / "cookies.txt"

    in_the_data_folder.write_text("# Netscape HTTP Cookie File\n")
    with server_started() as client:
        cookies = _check(client, "cookies")
    assert (cookies["status"], cookies["detail"]) == ("ok", str(in_the_data_folder))


def test_an_old_yt_dlp_is_flagged_with_how_to_update_it(setup_checks, monkeypatch, tmp_path):
    old = _yt_dlp_released(tmp_path, DAYS_BEFORE_YT_DLP_IS_OLD + 160)
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: str(old))

    flagged = _check(setup_checks, "yt_dlp_current")

    assert flagged["status"] == "optional"
    assert f"{DAYS_BEFORE_YT_DLP_IS_OLD + 160} days old" in flagged["detail"]
    assert "-U yt-dlp" in flagged["detail"] or "brew upgrade yt-dlp" in flagged["detail"]
    assert _check(setup_checks, "yt_dlp")["status"] == "ok"

    current = _yt_dlp_released(tmp_path, 10)
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: str(current))

    assert _check(setup_checks, "yt_dlp_current")["status"] == "ok"


def test_failed_downloads_are_counted_until_they_are_retried(
    setup_checks, session, make_track
):
    assert _check(setup_checks, "downloads")["status"] == "ok"
    for n in range(2):
        make_track(
            file_hash=f"pending:never-arrived-{n}",
            state=TrackState.ERROR.value,
            error_message="no upload matched",
        )
    make_track(state=TrackState.ERROR.value, error_message="separate: MPS backend out of memory")
    session.commit()

    downloads = _check(setup_checks, "downloads")

    assert downloads["status"] == "optional"
    assert downloads["detail"].startswith("2 downloads failed.")


def test_the_doctor_warns_about_an_old_yt_dlp(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from dance.cli import main

    old = _yt_dlp_released(tmp_path, DAYS_BEFORE_YT_DLP_IS_OLD + 1)
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: str(old))

    ran = CliRunner().invoke(main, ["doctor", "--json"])

    report = json.loads(ran.output)
    by_key = {check["key"]: check for check in report["checks"]}
    assert (by_key["yt_dlp_current"]["status"], by_key["yt_dlp_current"]["summary"]) == (
        "warn",
        "out of date",
    )
    assert f"{DAYS_BEFORE_YT_DLP_IS_OLD + 1} days old" in by_key["yt_dlp_current"]["fix"]
