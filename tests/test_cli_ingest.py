"""Tests for the `dance ingest` wrapper command.

These mock out the heavy phase implementations so we exercise the wrapper's
orchestration + summary, not the real pipeline.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from dance import cli
from dance.core.database import _reset_engine_for_tests
from dance.pipeline import preflight


@pytest.fixture(autouse=True)
def _isolate_db_engine():
    """`dance main()` caches a process-global DB engine bound to the first URL
    it sees. Reset before and after each test so these CLI invocations neither
    inherit nor leak a stale engine into other suites (e.g. the e2e test)."""
    _reset_engine_for_tests()
    yield
    _reset_engine_for_tests()


def _env(tmp_path) -> dict[str, str]:
    return {
        "DANCE_DATA_DIR": str(tmp_path / "data"),
        "DANCE_LIBRARY_DIR": str(tmp_path / "lib"),
        "DANCE_STEMS_DIR": str(tmp_path / "stems"),
        "DANCE_ALS_OUTPUT_DIR": str(tmp_path / "als"),
    }


def test_ingest_skip_sync_runs_remaining_phases_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    calls: list[str] = []
    monkeypatch.setattr(cli, "_run_sync", lambda *a, **k: calls.append("sync") or None)
    monkeypatch.setattr(
        cli, "_run_process", lambda *a, **k: calls.append("process") or {}
    )

    # build_graph / tag / export_als are invoked via ctx.invoke — patch their
    # callbacks so we don't touch real models or the recommender.
    monkeypatch.setattr(cli.build_graph, "callback", lambda *a, **k: calls.append("build-graph"))
    monkeypatch.setattr(cli.tag, "callback", lambda *a, **k: calls.append("tag"))
    monkeypatch.setattr(cli.export_als, "callback", lambda *a, **k: calls.append("export-als"))

    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest", "--skip-sync"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    # Sync skipped; everything else ran in order.
    assert calls == ["process", "build-graph", "tag", "export-als"]
    assert "Skipping Spotify sync" in result.output
    assert "Ingest complete" in result.output


def test_ingest_exits_when_ffmpeg_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

    def _boom(*a, **k):  # pragma: no cover - should never run
        raise AssertionError("phase ran despite missing ffmpeg")

    monkeypatch.setattr(cli, "_run_process", _boom)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest", "--skip-sync"], env=_env(tmp_path))

    assert result.exit_code != 0
    assert "ffmpeg not found" in result.output


def test_ingest_summary_reports_playable_and_stuck(monkeypatch, tmp_path):
    """The final summary counts COMPLETE tracks as playable and flags stuck
    ones."""
    from dance.config import Settings
    from dance.core.database import Track, TrackState, get_session, init_db, now_utc

    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli, "_run_process", lambda *a, **k: {})
    monkeypatch.setattr(cli.build_graph, "callback", lambda *a, **k: None)
    monkeypatch.setattr(cli.tag, "callback", lambda *a, **k: None)
    monkeypatch.setattr(cli.export_als, "callback", lambda *a, **k: None)

    env = _env(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "data",
        library_dir=tmp_path / "lib",
        stems_dir=tmp_path / "stems",
        als_output_dir=tmp_path / "als",
    )
    settings.ensure_directories()
    init_db(settings.db_url)
    session = get_session(settings.db_url)
    try:
        n = 0
        for state in (
            TrackState.COMPLETE,
            TrackState.COMPLETE,
            TrackState.ERROR,
            TrackState.SEPARATED,
        ):
            n += 1
            session.add(
                Track(
                    file_hash=f"{n:064d}",
                    file_path=f"/tmp/t{n}.mp3",
                    file_name=f"t{n}.mp3",
                    file_size_bytes=1000,
                    state=state.value,
                    created_at=now_utc(),
                    updated_at=now_utc(),
                )
            )
        session.commit()
    finally:
        session.close()

    runner = CliRunner()
    result = runner.invoke(cli.main, ["ingest", "--skip-sync"], env=env)

    assert result.exit_code == 0, result.output
    assert "2 track(s) now playable" in result.output
    # 1 errored + 1 mid-pipeline (SEPARATED) = 2 stuck.
    assert "2 still stuck" in result.output
