from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from dance.core.database import Track, TrackState
from dance.recommender.versions import Cut, first_credited_artist
from dance.spotify.matching import passes_duration_gate
from dance.spotify.search import SpotifySearchClient, SpotifyTrackHit

MOST_RESULTS_READ = 50


class NothingToSearchWithError(ValueError):
    pass


NEVER_DOWNLOADED = "pending:"


@dataclass(frozen=True)
class LibraryCut(Cut):
    track_id: int
    spotify_id: str | None
    state: str
    has_a_file: bool = True

    @property
    def download_failed(self) -> bool:
        return self.state == TrackState.ERROR.value and not self.has_a_file


@dataclass(frozen=True)
class SpotifyVersion:
    hit: SpotifyTrackHit
    label: str
    in_library: LibraryCut | None


@dataclass(frozen=True)
class VersionsOnSpotify:
    query: str
    versions: list[SpotifyVersion]


@dataclass
class _Release:
    hit: SpotifyTrackHit
    cut: Cut
    spotify_ids: set[str] = field(default_factory=set)


def _quoted(phrase: str) -> str:
    return '"' + " ".join(phrase.replace('"', " ").split()) + '"'


def versions_query(song: Cut) -> str:
    title, artist = song.parts.plain, first_credited_artist(song.artist)
    if not title or not artist:
        raise NothingToSearchWithError(
            "this track needs a title and an artist before Spotify can be searched for its versions"
        )
    return f"track:{_quoted(title)} artist:{_quoted(artist)}"


def _cut_of(hit: SpotifyTrackHit) -> Cut:
    seconds = hit.duration_ms / 1000.0 if hit.duration_ms else None
    return Cut(title=hit.title, artist=hit.artist, duration_seconds=seconds)


def _one_per_recording(song: Cut, hits: Sequence[SpotifyTrackHit]) -> list[_Release]:
    releases: list[_Release] = []
    for hit in hits:
        cut = _cut_of(hit)
        if not cut.is_the_same_song_as(song):
            continue
        earlier = next((r for r in releases if r.cut.is_the_same_recording_as(cut)), None)
        if earlier is None:
            earlier = _Release(hit, cut)
            releases.append(earlier)
        earlier.spotify_ids.add(hit.spotify_id)
    return releases


def _is_the_file_of(mine: LibraryCut, release: _Release) -> bool:
    if mine.spotify_id in release.spotify_ids:
        return True
    return (
        mine.is_the_same_song_as(release.cut)
        and mine.parts.note_key == release.cut.parts.note_key
        and passes_duration_gate(release.cut.duration_seconds, mine.duration_seconds)
    )


def _file_of(release: _Release, library: Sequence[LibraryCut]) -> LibraryCut | None:
    return next((cut for cut in library if _is_the_file_of(cut, release)), None)


def versions_among(
    song: LibraryCut, hits: Sequence[SpotifyTrackHit], library: Sequence[LibraryCut]
) -> list[SpotifyVersion]:
    versions: list[SpotifyVersion] = []
    for release in _one_per_recording(song, hits):
        if _is_the_file_of(song, release):
            continue
        mine = _file_of(release, library)
        versions.append(SpotifyVersion(release.hit, release.cut.label_seen_from(song), mine))
    return versions


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _library_cut(track: Track, stands_for: Track | None = None) -> LibraryCut:
    kept = stands_for or track
    return LibraryCut(
        title=_text(track.title),
        artist=_text(track.artist),
        duration_seconds=float(track.duration_seconds) if track.duration_seconds else None,
        track_id=int(kept.id),
        spotify_id=_text(track.spotify_id),
        state=str(kept.state),
        has_a_file=not str(kept.file_hash).startswith(NEVER_DOWNLOADED),
    )


def _whole_library(session: Session) -> list[LibraryCut]:
    tracks = {int(track.id): track for track in session.query(Track)}
    finished_first = sorted(
        tracks.values(),
        key=lambda t: (t.duplicate_of is not None, t.state != TrackState.COMPLETE.value, t.id),
    )
    return [
        _library_cut(track, tracks.get(int(track.duplicate_of or -1))) for track in finished_first
    ]


def versions_on_spotify(
    session: Session, client: SpotifySearchClient, track: Track
) -> VersionsOnSpotify:
    song = _library_cut(track)
    query = versions_query(song)
    hits = client.search_tracks(query, limit=MOST_RESULTS_READ)
    return VersionsOnSpotify(query, versions_among(song, hits, _whole_library(session)))


def _holder_of(hit: SpotifyTrackHit, library: Sequence[LibraryCut]) -> LibraryCut | None:
    if not hit.duration_ms:
        return next((cut for cut in library if cut.spotify_id == hit.spotify_id), None)
    return _file_of(_Release(hit, _cut_of(hit), {hit.spotify_id}), library)


def held_in_library(session: Session, hits: Sequence[SpotifyTrackHit]) -> list[LibraryCut | None]:
    library = _whole_library(session)
    return [_holder_of(hit, library) for hit in hits]
