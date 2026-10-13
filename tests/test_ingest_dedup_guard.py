"""Ingest dedup must not merge different versions of a song.

The fuzzy (artist, title) matcher had no duration or version guard, so on a
long credit line ``Alicante`` and ``Alicante (Extended)`` cleared the 0.92
ratio — and playlist ingest then copied the incoming Spotify ID onto the
*other* version's row, so every later ingest trusted the wrong file.

Now: both durations known → they must agree within 5 s; differing version
words → never a duplicate; and a name-only match (a duration missing on one
side) can still skip a re-download but never receives a Spotify ID.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pytest
from fastapi.testclient import TestClient

from dance.api.dedup import (
    find_duplicate,
    index_existing,
    match_duplicate,
    normalize,
    version_words,
)
from dance.core.database import Track

ALICANTE_ARTIST = "HUGEL, Tom Enzy, Danny Roma x Manrix, Yuliesky Gonzalez"


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def test_long_credit_extended_version_is_not_a_duplicate() -> None:
    idx = index_existing([(1, ALICANTE_ARTIST, "Alicante (Extended)")])
    # Prove the guard is what blocks it: the old ratio test alone passes.
    a = f"{normalize(ALICANTE_ARTIST)} {normalize('Alicante')}"
    b = f"{normalize(ALICANTE_ARTIST)} {normalize('Alicante (Extended)')}"
    assert SequenceMatcher(None, a, b).ratio() >= 0.92
    assert find_duplicate(ALICANTE_ARTIST, "Alicante", idx) is None


def test_version_words() -> None:
    # "Original Mix" names the default version, so it never blocks a match.
    assert version_words("Aria (Original Mix)") == version_words("Aria") == frozenset()
    assert version_words("Surrender - TH;EN Remix") == {"remix"}
    assert version_words("One More Time (Radio Edit)") == {"radio", "edit"}
    assert version_words("Alive") == frozenset()  # whole words only


def test_durations_more_than_5s_apart_are_different_recordings() -> None:
    idx = index_existing([(7, "CamelPhat, Arodes", "Cycles", 414.0)])
    assert find_duplicate("CamelPhat, Arodes", "Cycles", idx, incoming_duration_s=185.8) is None
    assert find_duplicate("CamelPhat, Arodes", "Cycles", idx, incoming_duration_s=410.0) == 7


@pytest.mark.parametrize(
    ("existing_dur", "incoming_dur", "confirmed"),
    [(200.0, 202.0, True), (None, 202.0, False), (200.0, None, False), (None, None, False)],
)
def test_match_reports_whether_duration_confirmed_it(existing_dur, incoming_dur, confirmed) -> None:
    idx = index_existing([(3, "Bonobo", "Kerala", existing_dur)])
    m = match_duplicate("Bonobo", "Kerala", idx, incoming_duration_s=incoming_dur)
    assert m is not None and m.track_id == 3
    assert m.duration_confirmed is confirmed


def test_confirmed_match_beats_a_name_only_twin() -> None:
    """Two same-named rows: the one whose duration proves identity wins, so
    the backfill isn't blocked by a duration-less twin listed first."""
    idx = index_existing([(1, "Bonobo", "Kerala", None), (2, "Bonobo", "Kerala", 230.0)])
    m = match_duplicate("Bonobo", "Kerala", idx, incoming_duration_s=231.0)
    assert m is not None and (m.track_id, m.duration_confirmed) == (2, True)


def test_three_field_rows_still_index() -> None:
    idx = index_existing([(1, "Daft Punk", "One More Time")])
    assert find_duplicate("Daft Punk", "One More Time", idx, incoming_duration_s=320) == 1


# ---------------------------------------------------------------------------
# Playlist ingest
# ---------------------------------------------------------------------------


@pytest.fixture
def api(session_factory, tmp_path, monkeypatch):
    from dance.api import create_app
    from dance.api.routers import pipeline as pipeline_mod
    from dance.config import Settings
    from tests.test_api import FakeAbletonBridge

    monkeypatch.setattr(pipeline_mod, "_run_spotify_ingest_job", lambda *a, **kw: None)
    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory)
    with TestClient(app) as client:
        yield client


def _playlist(monkeypatch, *hits) -> None:
    from dance.api.routers import spotify as spotify_router
    from dance.spotify import search as spotify_search
    from dance.spotify.search import SpotifyTrackHit

    rows = [
        SpotifyTrackHit(
            spotify_id=sid,
            title=title,
            artist=artist,
            album=None,
            duration_ms=ms,
            preview_url=None,
            image_url=None,
            explicit=False,
            popularity=None,
        )
        for sid, title, artist, ms in hits
    ]

    class _Client:
        configured = True

        def get_playlist_tracks(self, url, user_token=None):  # noqa: ARG002
            return list(rows)

    monkeypatch.setattr(spotify_search, "get_default_client", lambda _s: _Client())
    monkeypatch.setattr(spotify_router, "get_default_client", lambda _s: _Client())


def _ingest(api) -> dict:
    r = api.post(
        "/api/v1/pipeline/ingest/playlist",
        json={"playlist_url": "https://open.spotify.com/playlist/dedupguard1"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_name_only_match_is_skipped_but_never_gets_the_spotify_id(
    api, session, make_track, monkeypatch
) -> None:
    existing = make_track(title="Kerala", artist="Bonobo", spotify_id=None, duration_seconds=None)
    session.commit()
    _playlist(monkeypatch, ("kerala01", "Kerala", "Bonobo", 230_000))

    body = _ingest(api)

    assert body["added"] == 0 and body["skipped_existing"] == 1
    assert body["backfilled"] == 0
    session.expire_all()
    assert session.get(Track, existing.id).spotify_id is None


def test_duration_confirmed_match_backfills(api, session, make_track, monkeypatch) -> None:
    existing = make_track(title="Kerala", artist="Bonobo", spotify_id=None, duration_seconds=231.0)
    session.commit()
    _playlist(monkeypatch, ("kerala01", "Kerala", "Bonobo", 230_000))

    body = _ingest(api)

    assert body["backfilled"] == 1
    session.expire_all()
    assert session.get(Track, existing.id).spotify_id == "kerala01"


def test_other_version_is_ingested_separately_on_its_own_path(
    api, session, make_track, monkeypatch, tmp_path
) -> None:
    """The real ``Cycles`` case: a 414 s row for a 186 s Spotify track. It is
    not a duplicate, keeps no new ID, and the new row never claims its file."""
    library = tmp_path / "library"
    wrong = make_track(
        title="Cycles",
        artist="CamelPhat, Arodes",
        spotify_id=None,
        duration_seconds=414.0,
        file_path=str(library / "CamelPhat, Arodes - Cycles.mp3"),
    )
    session.commit()
    _playlist(monkeypatch, ("2bzCnD7nJuuS4yDNCAtLYz", "Cycles", "CamelPhat, Arodes", 185_800))

    body = _ingest(api)

    assert body["added"] == 1 and body["backfilled"] == 0
    session.expire_all()
    assert session.get(Track, wrong.id).spotify_id is None
    new = session.query(Track).filter(Track.spotify_id == "2bzCnD7nJuuS4yDNCAtLYz").one()
    assert new.id != wrong.id
    assert new.file_path.endswith("CamelPhat, Arodes - Cycles [2bzCnD7nJuuS4yDNCAtLYz].mp3")


def test_unowned_file_on_disk_is_still_adopted(api, session, monkeypatch, tmp_path) -> None:
    """A downloaded-but-not-yet-scanned file belongs to no row: keep the
    plain path so the worker's ``skip`` reuses it instead of re-downloading."""
    _playlist(monkeypatch, ("fresh0001", "Fresh", "Someone", 200_000))
    body = _ingest(api)
    assert body["added"] == 1
    new = session.query(Track).filter(Track.spotify_id == "fresh0001").one()
    assert new.file_path.endswith("Someone - Fresh.mp3")


def test_csv_preview_respects_the_duration_guard(api, session, make_track) -> None:
    make_track(title="Cycles", artist="CamelPhat", duration_seconds=414.0)
    make_track(title="Kerala", artist="Bonobo", duration_seconds=230.0)
    session.commit()
    csv = (
        "Track URI,Track Name,Album Name,Artist Name(s),Duration (ms)\n"
        "spotify:track:1,Cycles,Single,CamelPhat,185800\n"
        "spotify:track:2,Kerala,Migration,Bonobo,231000\n"
    )
    body = api.post("/api/v1/pipeline/ingest/preview", json={"csv_text": csv}).json()
    assert [r["title"] for r in body["new_rows"]] == ["Cycles"]
    assert [r["title"] for r in body["duplicates"]] == ["Kerala"]
