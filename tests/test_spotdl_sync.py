"""``spotdl sync`` must never delete or rename library files.

Without ``--sync-without-deleting``, spotdl 4.4's ``console/sync.py`` deletes
the local file of every song that left the playlist and renames files whose
Spotify metadata changed — orphaning Track rows, stems and every generated
``.als`` that references the file by absolute path.
"""

from __future__ import annotations

import sys

import pytest


def _captured_spotdl_cmd(tmp_path, monkeypatch) -> list[str]:
    from dance.config import Settings
    from dance.spotify import downloader as dl

    captured: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, **kwargs):
        captured.append(list(cmd))
        return _Done()

    monkeypatch.setattr(dl.subprocess, "run", _run)
    monkeypatch.setattr(dl.SpotifyDownloader, "_check_spotdl_installed", lambda self: True)
    settings = Settings(
        library_dir=tmp_path / "library",
        data_dir=tmp_path / "data",
        spotify_playlist_url="https://open.spotify.com/playlist/abcdefghij",
    )
    dl.SpotifyDownloader(settings).sync_playlist()
    (cmd,) = captured
    return cmd


def test_spotdl_sync_runs_without_deleting(tmp_path, monkeypatch) -> None:
    cmd = _captured_spotdl_cmd(tmp_path, monkeypatch)
    assert cmd[:2] == ["spotdl", "sync"]
    assert "--sync-without-deleting" in cmd


def test_installed_spotdl_accepts_the_sync_command(tmp_path, monkeypatch) -> None:
    """Parse our exact argv with the installed spotdl's own parser, so a
    renamed/removed flag fails here instead of silently at ingest time."""
    arguments = pytest.importorskip("spotdl.utils.arguments")
    cmd = _captured_spotdl_cmd(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", cmd)
    ns = arguments.create_parser().parse_args(cmd[1:])
    assert ns.operation == "sync"
    assert ns.sync_without_deleting is True
    assert ns.overwrite == "skip"
