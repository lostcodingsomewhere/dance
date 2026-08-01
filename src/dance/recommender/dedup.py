"""Collapse duplicate recordings in a recommendation list.

The library was built up over three ingest runs (2026-05-17 / 05-26 / 06-13)
which pulled overlapping sets of songs under slightly different filenames —
``Dot Major;Kitty Amor - Navi``, ``Dot Major, Kitty Amor - Navi``,
``Dot Major - Navi``. Different bytes, different ``file_hash``, so ingest's
hash-based dedup never fired and each became its own ``Track`` row.

Measured on the live library: **82 groups, 145 redundant tracks of 353 (41%)**,
of which 114 pairs are the *same recording* (durations within 2 s). The cost
lands squarely on the surface the DJ uses to build a set — 5 of 40 rec slots
across the five roles were literal repeats of another card in the same list.

Two classes, and only one of them should be collapsed:

* **Same recording** — same title, near-identical duration. One of these is
  pure noise in a rec list. Collapse.
* **Genuine variant** — an extended mix next to a radio edit (``Manifesto``:
  348 s vs 224 s). Different tools for different moments in a set. **Keep
  both** — hiding the extended mix behind a radio edit that happened to score
  higher would be worse than showing two cards.

The duration guard is what separates them, so it is deliberately tight.
"""

from __future__ import annotations

import re
from typing import Protocol, TypeVar

from sqlalchemy.orm import Session

from dance.core.database import Track

# Two recordings sharing a normalized title, within this many seconds of each
# other, are treated as the same recording. This guard — not the artist field —
# is what keeps distinct songs apart, so it is deliberately tight: real
# extended-vs-radio pairs here differ by 60 s or more, genuine re-rips by <1 s.
SAME_RECORDING_TOLERANCE_S = 5.0

# Mix-name suffixes that do NOT make a track a different recording — they are
# how the same file gets named by different sources.
_NOISE_SUFFIX = re.compile(
    r"\((original mix|original|extended mix|extended|radio edit|radio)\)",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def title_key(title: str | None) -> str:
    """Normalize a title so filename-level variation collapses.

    Lowercase, drop mix-name noise, strip everything non-alphanumeric — so
    ``JOY (In Me All The Time) (Original Mix)`` agrees with
    ``JOY (In Me All The Time)``.

    **Artist is deliberately not part of the key.** It looks like the obvious
    second field, but on this library it actively breaks the match: the same
    release is credited three different ways across the three ingest runs —
    ``Dot Major;Kitty Amor`` / ``Dot Major, Kitty Amor`` / ``Dot Major`` — so
    keying on artist collapsed only two of those three copies.

    Measured before making this choice: title+duration collapses 165 pairs
    across the library, and exactly ONE of them has artist strings that are
    not substrings of one another — ``Layton Giordani, Linney, Sarah de
    Warren`` vs ``Layton Giordani/Sarah de Warren/Linney``, the same three
    people with a different separator. Zero genuine false positives. The
    duration guard, not the artist, is what keeps distinct songs apart.
    """
    return _NON_ALNUM.sub("", _NOISE_SUFFIX.sub("", (title or "").lower()))


def dedupe_track_rows(tracks: list[Track]) -> list[Track]:
    """Same collapse, for a list of ``Track`` rows (order preserved).

    Separate from :func:`dedupe_by_recording` because a Track already carries
    its own title and duration — no lookup query is needed, which matters on
    the ⌘K path where the palette re-queries on every keystroke.

    Searching "navi" on this library returns the same recording three times,
    burning three of the palette's eight slots on one song.
    """
    kept: dict[str, list[float]] = {}
    out: list[Track] = []
    for t in tracks:
        # str() past SQLAlchemy's Column[str] declarative typing — the ORM
        # hands back a plain str at runtime.
        key = title_key(str(t.title) if t.title is not None else None)
        duration = t.duration_seconds
        if not key or duration is None:
            out.append(t)
            continue
        seen = kept.setdefault(key, [])
        if any(abs(float(duration) - d) <= SAME_RECORDING_TOLERANCE_S for d in seen):
            continue
        seen.append(float(duration))
        out.append(t)
    return out


def find_duplicate_groups(tracks: list[Track]) -> list[tuple[Track, list[Track]]]:
    """Group tracks into ``(canonical, redundant_copies)`` pairs.

    Only groups that are the SAME RECORDING are returned — same normalized
    title, durations within :data:`SAME_RECORDING_TOLERANCE_S`. Extended and
    radio versions of one song land in separate groups and are both canonical.

    Canonical selection, in order:

    1. **Fully processed wins.** ``state == 'complete'`` — never demote a
       working copy in favour of one that failed to separate.
    2. **Largest file wins.** For the same recording at the same duration,
       more bytes means a higher bitrate, and this library is already
       source-quality-bound (the stems are separated from lossy MP3s, which
       is the standing complaint about how they sound).
    3. **Lowest id** as a deterministic tie-break, so repeat runs agree.
    """
    buckets: dict[str, list[Track]] = {}
    for t in tracks:
        key = title_key(str(t.title) if t.title is not None else None)
        if not key or t.duration_seconds is None:
            continue
        buckets.setdefault(key, []).append(t)

    groups: list[tuple[Track, list[Track]]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        # Split a title bucket into same-recording clusters by duration, so an
        # extended mix never absorbs the radio edit.
        clusters: list[list[Track]] = []
        for t in sorted(members, key=lambda x: float(x.duration_seconds or 0.0)):
            dur = float(t.duration_seconds or 0.0)
            for c in clusters:
                if abs(dur - float(c[0].duration_seconds or 0.0)) <= SAME_RECORDING_TOLERANCE_S:
                    c.append(t)
                    break
            else:
                clusters.append([t])
        for c in clusters:
            if len(c) < 2:
                continue
            ranked = sorted(
                c,
                key=lambda t: (
                    0 if str(t.state) == "complete" else 1,
                    -int(t.file_size_bytes or 0),
                    int(t.id),
                ),
            )
            groups.append((ranked[0], ranked[1:]))
    return groups


class _HasTrackId(Protocol):
    track_id: int


T = TypeVar("T", bound=_HasTrackId)


def dedupe_by_recording(session: Session, items: list[T]) -> list[T]:
    """Keep the first item per distinct recording, preserving order.

    Callers pass an already-SORTED list, so "first" means "best" and no
    tie-breaking policy is invented here: whichever copy the scorer preferred
    for this context is the one that survives.

    Items whose track is missing metadata are never collapsed — with no title
    to compare we cannot claim two things are the same recording, and dropping
    a candidate on a guess is worse than showing one card too many.
    """
    if not items:
        return items

    track_ids = {int(i.track_id) for i in items}
    rows = (
        session.query(Track.id, Track.title, Track.duration_seconds)
        .filter(Track.id.in_(track_ids))
        .all()
    )
    meta = {int(r[0]): (r[1], r[2]) for r in rows}

    # normalized title -> durations already kept under it
    kept_durations: dict[str, list[float]] = {}
    out: list[T] = []
    for item in items:
        info = meta.get(int(item.track_id))
        if info is None:
            out.append(item)
            continue
        title, duration = info
        key = title_key(title)
        if not key:
            # No usable metadata — can't claim a duplicate.
            out.append(item)
            continue
        if duration is None:
            out.append(item)
            kept_durations.setdefault(key, [])
            continue
        seen = kept_durations.setdefault(key, [])
        if any(abs(float(duration) - d) <= SAME_RECORDING_TOLERANCE_S for d in seen):
            continue  # same recording as one we already kept
        seen.append(float(duration))
        out.append(item)
    return out
