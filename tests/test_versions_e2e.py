from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from mutagen.flac import FLAC

from dance.api import create_app
from dance.config import get_settings
from dance.core.database import (
    Track,
    TrackState,
    _reset_engine_for_tests,
    get_session,
    get_session_factory,
    init_db,
)
from dance.pipeline import dispatcher as dispatcher_mod
from dance.pipeline import preflight
from dance.pipeline.dispatcher import Dispatcher
from tests.audio_fixtures import TrackSpec, synth_track
from tests.test_api import FakeAbletonBridge
from tests.test_cue_pads_e2e import API, _heavy_models_faked

SHORT_BARS, LONG_BARS, LONGEST_BARS = 8, 12, 16


@dataclass(frozen=True)
class Song:
    key: str
    title: str
    artist: str
    bars: int = SHORT_BARS


LIBRARY = [
    Song("original", "Shallow Water", "Elderbrook"),
    Song("club_mix", "Shallow Water - Club Mix", "Elderbrook"),
    Song("extended", "Shallow Water (Extended Mix)", "Elderbrook, Friend", bars=LONG_BARS),
    Song("another_bands", "Shallow Water", "Other Band", bars=LONGEST_BARS),
    Song("plain_spelling", "Alive", "Kx5"),
    Song("credited_spelling", "Alive (feat. The Moth & The Flame)", "Kx5/deadmau5"),
    Song("short_cut", "Cycles", "CamelPhat;Arodes"),
    Song("long_cut", "Cycles", "CamelPhat/Arodes", bars=LONG_BARS),
]


SECOND_COPY_OF_THE_ORIGINAL = Song("original_again", "Shallow Water", "Elderbrook")


@pytest.fixture
def library(monkeypatch):
    yield from _through_the_pipeline(monkeypatch, LIBRARY)


@pytest.fixture
def library_with_a_second_copy(monkeypatch):
    songs = [LIBRARY[0], SECOND_COPY_OF_THE_ORIGINAL, LIBRARY[1]]
    yield from _through_the_pipeline(monkeypatch, songs)


def _through_the_pipeline(monkeypatch, songs: list[Song]):
    monkeypatch.setattr(dispatcher_mod, "_IDLE_GRACE_S", 0.3)
    monkeypatch.setattr(dispatcher_mod, "_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    settings = get_settings()
    settings.ensure_directories()
    init_db(settings.db_url)
    for seed, song in enumerate(songs, start=1):
        spec = TrackSpec(bpm=128.0, bars=song.bars, seed=seed)
        path = Path(settings.library_dir) / f"{song.key}.flac"
        sf.write(str(path), synth_track(spec), spec.sample_rate, format="FLAC", subtype="PCM_16")
        tags = FLAC(str(path))
        tags["title"], tags["artist"] = song.title, song.artist
        tags.save()

    session = get_session(settings.db_url)
    separate, ensure_model, encode = _heavy_models_faked()
    with separate, ensure_model, encode:
        dispatcher = Dispatcher(settings, session)
        assert dispatcher.ingest()["new"] == len(songs)
        dispatcher.run()
    ids = {}
    for track in session.query(Track):
        assert track.state == TrackState.COMPLETE.value, track.error_message
        ids[Path(track.file_path).stem] = track.id
    session.close()

    app = create_app(
        settings=settings,
        bridge=FakeAbletonBridge(),
        session_factory=get_session_factory(settings.db_url),
    )
    with TestClient(app) as client:
        yield client, ids
    _reset_engine_for_tests()


def _versions(client: TestClient, track_id: int) -> list[tuple[int, str]]:
    found = client.get(f"{API}/tracks/{track_id}/versions")
    assert found.status_code == 200, found.text
    return [(version["track_id"], version["label"]) for version in found.json()]


def _track(client: TestClient, track_id: int) -> dict:
    return client.get(f"{API}/tracks/{track_id}").json()


def _set_with_songs(client: TestClient, song_ids: list[int]) -> int:
    created = client.post(f"{API}/sets", json={"name": "Friday night"})
    assert created.status_code == 200, created.text
    set_id = created.json()["id"]
    planned = client.put(f"{API}/sets/{set_id}/plan", json={"queues": {"song": song_ids}})
    assert planned.status_code == 200, planned.text
    return set_id


def _song_recs(client: TestClient, set_id: int) -> list[dict]:
    recs = client.get(f"{API}/sets/{set_id}/plan-recs", params={"role": "song", "k": 8})
    assert recs.status_code == 200, recs.text
    return recs.json()["recs"]


def _marked_as_versions(recs: list[dict]) -> list[tuple[int, int, str]]:
    return [
        (rec["track_id"], rec["version_of"], rec["version_label"])
        for rec in recs
        if rec["version_of"] is not None
    ]


def test_a_songs_versions_are_its_other_cuts_by_the_same_artists(library):
    client, ids = library
    original, club_mix, extended = ids["original"], ids["club_mix"], ids["extended"]

    assert _versions(client, original) == [(club_mix, "Club Mix"), (extended, "Extended Mix")]
    assert _versions(client, club_mix) == [(original, "original"), (extended, "Extended Mix")]
    assert _versions(client, ids["another_bands"]) == []
    assert _versions(client, ids["plain_spelling"]) == []
    assert _versions(client, ids["credited_spelling"]) == []
    assert _versions(client, ids["short_cut"]) == [(ids["long_cut"], "longer cut")]
    assert _versions(client, ids["long_cut"]) == [(ids["short_cut"], "shorter cut")]
    assert client.get(f"{API}/tracks/999999/versions").status_code == 404

    listed = client.get(f"{API}/tracks/{original}/versions").json()[0]
    club_mix_track = _track(client, club_mix)
    assert (listed["title"], listed["artist"]) == ("Shallow Water - Club Mix", "Elderbrook")
    assert listed["duration_seconds"] == pytest.approx(club_mix_track["duration_seconds"])
    assert listed["bpm"] == pytest.approx(club_mix_track["analysis"]["bpm"])

    assert [_track(client, ids[key])["version_count"] for key in ("original", "extended")] == [2, 2]
    assert _track(client, ids["another_bands"])["version_count"] == 0
    family = _track(client, original)["version_family"]
    assert family is not None
    assert _track(client, club_mix)["version_family"] == family
    assert _track(client, ids["another_bands"])["version_family"] is None

    found = client.get(f"{API}/tracks/search", params={"q": "shallow", "limit": 8}).json()
    families_in_order = [track["version_family"] for track in found]
    assert sorted(track["id"] for track in found) == sorted(
        [original, club_mix, extended, ids["another_bands"]]
    )
    together = [index for index, value in enumerate(families_in_order) if value == family]
    assert together == list(range(together[0], together[0] + 3))


def test_versions_of_planned_songs_lead_the_song_column_and_queue_next_to_them(library):
    client, ids = library
    original, club_mix, extended = ids["original"], ids["club_mix"], ids["extended"]
    set_id = _set_with_songs(client, [original, ids["plain_spelling"]])

    recs = _song_recs(client, set_id)

    assert _marked_as_versions(recs) == [
        (club_mix, original, "Club Mix"),
        (extended, original, "Extended Mix"),
    ]
    assert [rec["track_id"] for rec in recs[:2]] == [club_mix, extended]
    assert recs[0]["version_of_title"] == "Shallow Water"
    assert recs[1]["duration_seconds"] == pytest.approx(
        _track(client, extended)["duration_seconds"]
    )
    assert ids["another_bands"] in [rec["track_id"] for rec in recs[2:]]
    drum_recs = client.get(f"{API}/sets/{set_id}/plan-recs", params={"role": "drums", "k": 8})
    assert _marked_as_versions(drum_recs.json()["recs"]) == []

    queued = client.post(
        f"{API}/sets/{set_id}/plan/append",
        json={"role": "song", "track_id": club_mix, "after_track_id": original},
    )
    assert queued.status_code == 200, queued.text
    assert [item["track_id"] for item in queued.json()["queues"]["song"]] == [
        original,
        club_mix,
        ids["plain_spelling"],
    ]
    assert _marked_as_versions(_song_recs(client, set_id)) == [
        (extended, original, "Extended Mix")
    ]


def test_versions_of_the_songs_on_the_decks_lead_the_booths_song_column(library):
    client, ids = library
    original, club_mix, extended = ids["original"], ids["club_mix"], ids["extended"]

    def booth_recs(column: str, on_the_decks: list[int]) -> list[dict]:
        recs = client.post(
            f"{API}/recommend/by-column",
            json={
                "column": column,
                "combo_stem_ids": [],
                "master_bpm": 128.0,
                "k": 4,
                "exclude_track_ids": on_the_decks,
                "deck_track_ids": on_the_decks,
            },
        )
        assert recs.status_code == 200, recs.text
        return recs.json()["recs"]

    with_the_original_loaded = booth_recs("mix", [original])
    assert _marked_as_versions(with_the_original_loaded) == [
        (club_mix, original, "Club Mix"),
        (extended, original, "Extended Mix"),
    ]
    assert [rec["track_id"] for rec in with_the_original_loaded[:2]] == [club_mix, extended]
    assert len(with_the_original_loaded) == 4

    assert _marked_as_versions(booth_recs("mix", [])) == []
    assert _marked_as_versions(booth_recs("drums", [original])) == []
    assert _marked_as_versions(booth_recs("mix", [original, club_mix])) == [
        (extended, original, "Extended Mix")
    ]


def test_a_second_copy_of_a_planned_song_is_not_offered_as_its_version(
    library_with_a_second_copy,
):
    client, ids = library_with_a_second_copy
    original, second_copy, club_mix = ids["original"], ids["original_again"], ids["club_mix"]

    assert _versions(client, original) == [(club_mix, "Club Mix")]
    assert _versions(client, second_copy) == [(club_mix, "Club Mix")]
    assert len(_versions(client, club_mix)) == 1

    for planned, other_copy in ((original, second_copy), (second_copy, original)):
        set_id = _set_with_songs(client, [planned, club_mix])
        assert _marked_as_versions(_song_recs(client, set_id)) == []
        set_id = _set_with_songs(client, [planned])
        assert _marked_as_versions(_song_recs(client, set_id)) == [
            (club_mix, planned, "Club Mix")
        ]
        assert other_copy != planned
