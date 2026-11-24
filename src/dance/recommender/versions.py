from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

from sqlalchemy.orm import Session

from dance.core.database import Track, TrackState
from dance.recommender.dedup import SAME_RECORDING_TOLERANCE_S

VERSION_WORDS = frozenset(
    {
        "remix", "rmx", "extended", "edit", "mix", "mixed", "live", "acoustic",
        "instrumental", "acapella", "vip", "dub", "rework", "bootleg", "flip",
        "radio", "club", "version", "remaster", "remastered", "reprise", "demo",
        "unplugged", "slowed", "sped", "nightcore", "cover", "karaoke",
    }
)
ORIGINAL_LABEL = "original"
LONGER_LABEL = "longer cut"
SHORTER_LABEL = "shorter cut"

_BRACKETED = re.compile(r"\(([^)]*)\)|\[([^\]]*)\]")
_CREDIT = re.compile(r"^\s*(?:feat|ft|featuring|with|prod)\b", re.IGNORECASE)
_TRAILING_CREDIT = re.compile(r"\b(?:feat|ft|featuring)\b\.?\s.*$", re.IGNORECASE)
_DEFAULT_VERSION_WORDS = frozenset({"original", "mix", "version"})
_ARTIST_SEPARATORS = re.compile(
    r";|,|/|&|\bx\b|\bvs\b\.?|\bfeat\b\.?|\bft\b\.?|\bfeaturing\b|\band\b", re.IGNORECASE
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_TAIL_SEPARATOR = " - "
_SESSION_CACHE_KEY = "dance_version_index"


def _folded(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def _words(text: str) -> set[str]:
    return set(_NON_ALNUM.sub(" ", _folded(text)).split())


def _key(text: str) -> str:
    return _NON_ALNUM.sub("", _folded(text))


def _names_a_version(text: str) -> bool:
    return bool(_words(text) & VERSION_WORDS)


def _is_the_default_version(text: str) -> bool:
    words = _words(text)
    return "original" in words and words <= _DEFAULT_VERSION_WORDS


@dataclass(frozen=True)
class TitleParts:
    base: str
    notes: tuple[str, ...]

    @property
    def note(self) -> str:
        return ", ".join(self.notes)

    @property
    def note_key(self) -> str:
        return _key(" ".join(self.notes))


def title_parts(title: str | None) -> TitleParts:
    notes: list[str] = []

    def without_version_or_credit(bracket: re.Match[str]) -> str:
        inside = (bracket.group(1) or bracket.group(2) or "").strip()
        if _CREDIT.match(inside) or _is_the_default_version(inside):
            return " "
        if _names_a_version(inside):
            notes.append(inside)
            return " "
        return f" {inside} "

    segments = _BRACKETED.sub(without_version_or_credit, title or "").split(_TAIL_SEPARATOR)
    tails: list[str] = []
    while len(segments) > 1 and (
        _names_a_version(segments[-1]) or _is_the_default_version(segments[-1])
    ):
        tails.append(segments.pop().strip())
    notes += [tail for tail in reversed(tails) if not _is_the_default_version(tail)]
    base = _key(_TRAILING_CREDIT.sub("", _TAIL_SEPARATOR.join(segments)))
    return TitleParts(base=base, notes=tuple(notes))


def credited_artists(artist: str | None) -> frozenset[str]:
    without_tags = _BRACKETED.sub(" ", artist or "")
    return frozenset(
        name for name in (_key(part) for part in _ARTIST_SEPARATORS.split(without_tags)) if name
    )


@dataclass(frozen=True)
class Recording:
    track_id: int
    title: str | None
    artist: str | None
    duration_seconds: float | None
    file_size_bytes: int = 0

    @cached_property
    def parts(self) -> TitleParts:
        return title_parts(self.title)

    @cached_property
    def artists(self) -> frozenset[str]:
        return credited_artists(self.artist)

    def is_the_same_recording_as(self, other: Recording) -> bool:
        if self.duration_seconds is None or other.duration_seconds is None:
            return False
        close = abs(self.duration_seconds - other.duration_seconds) <= SAME_RECORDING_TOLERANCE_S
        return close and self.parts.note_key == other.parts.note_key


@dataclass(frozen=True)
class Version:
    track_id: int
    label: str


@dataclass(frozen=True)
class LeadingVersion:
    of_track_id: int
    label: str


def _label(version: Recording, seen_from: Recording) -> str:
    if version.parts.note_key != seen_from.parts.note_key:
        return version.parts.note or ORIGINAL_LABEL
    longer = (version.duration_seconds or 0.0) > (seen_from.duration_seconds or 0.0)
    return LONGER_LABEL if longer else SHORTER_LABEL


def _sharing_an_artist(recordings: Sequence[Recording]) -> list[list[Recording]]:
    groups: list[list[Recording]] = []
    for recording in recordings:
        joined = [group for group in groups if any(recording.artists & r.artists for r in group)]
        merged = [recording, *(r for group in joined for r in group)]
        groups = [group for group in groups if group not in joined] + [merged]
    return groups


def _distinct_recordings(group: Sequence[Recording]) -> dict[int, Recording]:
    kept: list[Recording] = []
    stands_for: dict[int, Recording] = {}
    for recording in sorted(group, key=lambda r: (-r.file_size_bytes, r.track_id)):
        same = next((k for k in kept if k.is_the_same_recording_as(recording)), None)
        if same is None:
            kept.append(recording)
        stands_for[recording.track_id] = same or recording
    return stands_for


def _family_order(recording: Recording) -> tuple[bool, str, float, int]:
    parts = recording.parts
    return (
        bool(parts.notes),
        parts.note.lower(),
        -(recording.duration_seconds or 0.0),
        recording.track_id,
    )


class VersionIndex:
    def __init__(self, recordings: Iterable[Recording]) -> None:
        self._recordings = {recording.track_id: recording for recording in recordings}
        self._family: dict[int, list[Recording]] = {}
        self._stands_for: dict[int, Recording] = {}
        by_base: dict[str, list[Recording]] = {}
        for recording in self._recordings.values():
            if recording.parts.base:
                by_base.setdefault(recording.parts.base, []).append(recording)
        for same_base in by_base.values():
            for group in _sharing_an_artist(same_base):
                stands_for = _distinct_recordings(group)
                distinct = sorted(
                    {r.track_id: r for r in stands_for.values()}.values(), key=_family_order
                )
                if len(distinct) < 2:
                    continue
                for recording in group:
                    self._family[recording.track_id] = distinct
                    self._stands_for[recording.track_id] = stands_for[recording.track_id]

    def versions_of(self, track_id: int) -> list[Version]:
        asked = self._recordings.get(track_id)
        if asked is None or track_id not in self._family:
            return []
        itself = self._stands_for[track_id]
        return [
            Version(track_id=other.track_id, label=_label(other, asked))
            for other in self._family[track_id]
            if other is not itself
        ]

    def recording_of(self, track_id: int) -> int:
        kept = self._stands_for.get(track_id)
        return kept.track_id if kept else track_id

    def family_id(self, track_id: int) -> int | None:
        family = self._family.get(track_id)
        return min(recording.track_id for recording in family) if family else None


def version_index(session: Session) -> VersionIndex:
    index = session.info.get(_SESSION_CACHE_KEY)
    if index is None:
        rows = session.query(
            Track.id, Track.title, Track.artist, Track.duration_seconds, Track.file_size_bytes
        ).filter(
            Track.duplicate_of.is_(None),
            Track.state == TrackState.COMPLETE.value,
            Track.title.isnot(None),
        )
        index = VersionIndex(
            Recording(
                track_id=int(track_id),
                title=title,
                artist=artist,
                duration_seconds=float(duration) if duration is not None else None,
                file_size_bytes=int(size or 0),
            )
            for track_id, title, artist, duration, size in rows
        )
        session.info[_SESSION_CACHE_KEY] = index
    return index


def leading_versions(
    index: VersionIndex, of_track_ids: Sequence[int], *, skip: Iterable[int], limit: int
) -> dict[int, LeadingVersion]:
    skipped = {index.recording_of(track_id) for track_id in skip}
    leading: dict[int, LeadingVersion] = {}
    for of_track_id in of_track_ids:
        for version in index.versions_of(of_track_id):
            if len(leading) >= limit:
                return leading
            if version.track_id not in skipped and version.track_id not in leading:
                leading[version.track_id] = LeadingVersion(of_track_id, version.label)
    return leading
