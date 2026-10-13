"""Tests for settings resolution under the test suite."""

from __future__ import annotations

from pathlib import Path

from dance.config import get_settings


def test_settings_never_resolve_to_the_real_home():
    """Regression: a cached ``get_settings()`` used to pin the real ``~/.dance``
    for the rest of the session, so a later CLI test ran ``init_db()`` against
    the real library DB. conftest's autouse fixture must keep every path in a
    temp dir and stop ``~/.dance/.env`` from being read."""

    settings = get_settings()
    home = Path.home()

    assert settings.db_url != f"sqlite:///{home / '.dance' / 'dance.db'}"
    for path in (
        settings.data_dir,
        settings.library_dir,
        settings.stems_dir,
        settings.als_output_dir,
    ):
        assert not path.is_relative_to(home / ".dance")
        assert not path.is_relative_to(home / "Music")
    # Only ~/.dance/.env could set this; the test env never does.
    assert settings.spotify_client_id is None
