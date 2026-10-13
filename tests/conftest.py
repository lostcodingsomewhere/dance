"""Shared pytest fixtures for the Dance test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from dance import config
from dance.core import database as db
from dance.core.database import now_utc


@pytest.fixture(autouse=True)
def _isolate_from_real_home(tmp_path_factory, monkeypatch):
    """Keep every test away from the real ``~/.dance`` and ``~/Music``.

    ``get_settings()`` and ``get_engine()`` cache process-global singletons,
    and ``Settings`` reads ``~/.dance/.env`` on top of defaults rooted in the
    real home directory. Without this, the first test to call
    ``get_settings()`` pins the real paths for the rest of the session, and a
    later CLI test's ``init_db()`` runs against the real ``~/.dance/dance.db``.
    """

    home = tmp_path_factory.mktemp("dance_home")
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    for name in [n for n in os.environ if n.startswith("DANCE_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("DANCE_DATA_DIR", str(home / "data"))
    monkeypatch.setenv("DANCE_LIBRARY_DIR", str(home / "library"))
    monkeypatch.setenv("DANCE_STEMS_DIR", str(home / "stems"))
    monkeypatch.setenv("DANCE_ALS_OUTPUT_DIR", str(home / "sets"))
    # Starlette's TestClient sends ``Host: testserver``; the API's origin
    # guard only answers to loopback names. tests/test_origin_guard.py pins
    # the real defaults explicitly.
    monkeypatch.setenv(
        "DANCE_API_ALLOWED_HOSTS",
        ",".join([*config.DEFAULT_API_ALLOWED_HOSTS, "testserver"]),
    )
    config._settings = None
    db._reset_engine_for_tests()
    yield
    config._settings = None
    db._reset_engine_for_tests()


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """Return a fresh ``sqlite:///`` URL for a test-scoped database file."""

    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def engine(db_url: str):
    """Create a fresh engine + schema for each test, isolated from globals."""

    eng = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    db.Base.metadata.create_all(bind=eng)
    # Apply the partial unique indexes used by audio_analysis.
    db._create_partial_unique_indexes(eng)

    yield eng

    eng.dispose()


@pytest.fixture
def session_factory(engine) -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def session(session_factory) -> Session:
    s = session_factory()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def make_track(session: Session):
    """Factory for creating Track rows with sensible defaults."""

    counter = {"n": 0}

    def _make(**overrides) -> db.Track:
        counter["n"] += 1
        n = counter["n"]
        defaults = dict(
            file_hash=f"{n:064d}",
            file_path=f"/tmp/track{n}.mp3",
            file_name=f"track{n}.mp3",
            file_size_bytes=1024 * n,
            title=f"Track {n}",
            artist=f"Artist {n}",
            state=db.TrackState.PENDING.value,
            created_at=now_utc(),
            updated_at=now_utc(),
        )
        defaults.update(overrides)
        track = db.Track(**defaults)
        session.add(track)
        session.flush()
        return track

    return _make
