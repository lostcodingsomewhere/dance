"""Tests for the ffmpeg preflight check."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from dance.core.database import _reset_engine_for_tests
from dance.pipeline import preflight


@pytest.fixture(autouse=True)
def _isolate_db_engine():
    """`dance main()` caches a process-global DB engine. Reset before/after so
    CLI invocations here don't leak it into other suites (e.g. the e2e test)."""
    _reset_engine_for_tests()
    yield
    _reset_engine_for_tests()


def test_require_ffmpeg_passes_when_present(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    # Should not raise.
    preflight.require_ffmpeg()
    assert preflight.ffmpeg_available() is True


def test_require_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    assert preflight.ffmpeg_available() is False
    with pytest.raises(preflight.PreflightError) as exc:
        preflight.require_ffmpeg()
    msg = str(exc.value)
    assert "ffmpeg not found" in msg
    assert "brew install ffmpeg" in msg


def test_process_command_exits_when_ffmpeg_missing(monkeypatch, tmp_path):
    """`dance process` must fail fast with a non-zero exit and the actionable
    message, BEFORE constructing a dispatcher or doing any work."""
    from dance import cli

    # ffmpeg absent everywhere the CLI looks.
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

    # If the preflight is skipped, this would explode — prove it's never reached.
    def _boom(*a, **k):  # pragma: no cover - should never run
        raise AssertionError("dispatcher work ran despite missing ffmpeg")

    monkeypatch.setattr(cli, "_run_process", _boom)

    runner = CliRunner()
    # Point data_dir/library at tmp so we don't touch the real DB.
    env = {
        "DANCE_DATA_DIR": str(tmp_path / "data"),
        "DANCE_LIBRARY_DIR": str(tmp_path / "lib"),
        "DANCE_STEMS_DIR": str(tmp_path / "stems"),
        "DANCE_ALS_OUTPUT_DIR": str(tmp_path / "als"),
    }
    result = runner.invoke(cli.main, ["process"], env=env)

    assert result.exit_code != 0
    assert "ffmpeg not found" in result.output
    assert "brew install ffmpeg" in result.output
