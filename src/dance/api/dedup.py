"""Fuzzy-match heuristics for "is this incoming CSV row already in the library?"

Goal: catch obvious duplicates (case / punctuation / artist-order differences)
without flagging different *versions* of the same song as duplicates. A "club
mix" and an "original mix" of "One More Time" are intentionally treated as
distinct tracks — DJs care about which version they have.

Strategy (cheap, stdlib only):

1. Normalize artist + title to a canonical form: lowercase, alphanumeric-only,
   single-spaced. This collapses "Daft Punk – One More Time" and "daft punk
   one more time" but preserves "(Club Mix)" vs "(Extended Mix)" if those
   words are present.
2. Token-set comparison: split into words, intersect — order doesn't matter
   ("RY X;Rhye" matches "Rhye;RY X").
3. SequenceMatcher ratio on the canonical form as a tie-breaker — > 0.92
   ratio with all incoming tokens present = duplicate.

Two guards keep different *versions* apart even when the names fuzz together
(``Title`` vs ``Title (Extended Mix)`` on a long title can clear the 0.92
ratio):

4. Duration: when both sides know their duration, they must agree within
   :data:`SAME_RECORDING_TOLERANCE_S` (5 s — the same guard the recommender's
   same-recording collapse uses; real extended-vs-radio pairs differ by 60 s+).
5. Version words: if the two titles carry different version markers
   (remix / extended / edit / live / ...), they are different recordings.

:func:`match_duplicate` also reports whether the duration actually confirmed
the match. Callers must not treat a name-only ("fuzzy-only") match as proof of
identity — in particular, playlist ingest never copies a Spotify ID onto one.

We never block ingest on a dupe match; we only flag them so the UI can show
"these N look already-loaded, want to skip?".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")

#: Same recording when both durations are known and differ by at most this.
#: Mirrors ``dance.recommender.dedup.SAME_RECORDING_TOLERANCE_S``.
SAME_RECORDING_TOLERANCE_S = 5.0

#: Title words that name a *version*. Two titles whose sets of these differ
#: are different recordings, however similar the rest of the string is.
VERSION_WORDS = frozenset(
    {
        "remix", "rmx", "extended", "edit", "mix", "mixed", "live", "acoustic",
        "instrumental", "acapella", "vip", "dub", "rework", "bootleg", "flip",
        "radio", "club", "version", "remaster", "remastered", "reprise", "demo",
        "unplugged", "slowed", "sped", "nightcore", "cover", "karaoke",
    }
)
# "Original Mix" is the default version, not a different one: strip it before
# comparing version words so ``Title (Original Mix)`` still matches ``Title``.
_ORIGINAL_MIX = re.compile(r"\boriginal mix\b")


def normalize(text: str) -> str:
    """Lowercase, ASCII-fold, strip non-alphanumeric, collapse whitespace."""
    if not text:
        return ""
    # NFKD strips diacritics: "Amélie" -> "Amelie"
    folded = unicodedata.normalize("NFKD", text)
    folded = folded.encode("ascii", "ignore").decode("ascii")
    folded = folded.lower()
    folded = _NON_ALNUM.sub(" ", folded)
    folded = _WS.sub(" ", folded).strip()
    return folded


def version_words(title: str | None) -> frozenset[str]:
    """Version markers present in a title (after dropping "original mix")."""
    norm = _ORIGINAL_MIX.sub(" ", normalize(title or ""))
    return frozenset(w for w in norm.split() if w in VERSION_WORDS)


@dataclass(frozen=True)
class _ExistingEntry:
    track_id: int
    artist_norm: str
    title_norm: str
    combined: str
    duration_s: float | None = None
    versions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DuplicateMatch:
    track_id: int
    #: True when both durations were known and agreed within tolerance —
    #: the only kind of match strong enough to copy identity (spotify_id)
    #: across. False = names matched, durations unknown on one side.
    duration_confirmed: bool


def _positive(value: float | int | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if v > 0 else None


def index_existing(rows: Iterable[Sequence[Any]]) -> list[_ExistingEntry]:
    """Build a searchable index from ``(track_id, artist, title[, duration_s])``.

    The duration is optional so existing three-field callers keep working;
    without it the duration guard simply cannot fire for that entry.
    """
    out: list[_ExistingEntry] = []
    for row in rows:
        track_id, artist, title = int(row[0]), row[1], row[2]
        duration = row[3] if len(row) > 3 else None
        a = normalize(artist or "")
        t = normalize(title or "")
        out.append(
            _ExistingEntry(
                track_id=track_id,
                artist_norm=a,
                title_norm=t,
                combined=f"{a} {t}".strip(),
                duration_s=_positive(duration),
                versions=version_words(title),
            )
        )
    return out


def find_duplicate(
    incoming_artist: str,
    incoming_title: str,
    index: list[_ExistingEntry],
    ratio_threshold: float = 0.92,
    incoming_duration_s: float | None = None,
) -> int | None:
    """Return the track_id of the most likely duplicate, or ``None``.

    See :func:`match_duplicate`, which also says whether the match was
    confirmed by duration.
    """
    m = match_duplicate(
        incoming_artist, incoming_title, index, ratio_threshold, incoming_duration_s
    )
    return m.track_id if m else None


def match_duplicate(
    incoming_artist: str,
    incoming_title: str,
    index: list[_ExistingEntry],
    ratio_threshold: float = 0.92,
    incoming_duration_s: float | None = None,
) -> DuplicateMatch | None:
    """Return the most likely duplicate, or ``None``.

    Stages:

    1. Token-set: incoming tokens (combined artist+title) must all appear in
       the existing combined string. Cheap, catches reorderings.
    2. SequenceMatcher ratio on the combined strings ≥ threshold for
       ambiguous cases where lengths differ slightly.

    Gates applied to every candidate entry first:

    - different version words → not a duplicate;
    - both durations known and more than 5 s apart → not a duplicate.
    """
    inc_combined = f"{normalize(incoming_artist)} {normalize(incoming_title)}".strip()
    if not inc_combined:
        return None
    inc_tokens = set(inc_combined.split())
    inc_versions = version_words(incoming_title)
    inc_duration = _positive(incoming_duration_s)

    def _confirmed(entry: _ExistingEntry) -> bool | None:
        """None = durations disagree (reject); else whether both were known."""
        if inc_duration is None or entry.duration_s is None:
            return False
        if abs(inc_duration - entry.duration_s) > SAME_RECORDING_TOLERANCE_S:
            return None
        return True

    # Ranked by (duration-confirmed, ratio): a confirmed match beats a
    # name-only one, so a same-named row with no duration can't shadow the
    # row that actually proves identity.
    best: tuple[tuple[bool, float], DuplicateMatch] | None = None
    for entry in index:
        if not entry.combined:
            continue
        if entry.versions != inc_versions:
            continue
        confirmed = _confirmed(entry)
        if confirmed is None:
            continue
        ex_tokens = set(entry.combined.split())
        match = DuplicateMatch(entry.track_id, confirmed)
        # Quick win: identical token sets.
        if inc_tokens == ex_tokens:
            if confirmed:
                return match
            if best is None or (False, 1.0) > best[0]:
                best = ((False, 1.0), match)
            continue
        # Otherwise: every incoming token must appear in the existing entry
        # (catches "Daft Punk One More Time" vs "Daft Punk - One More Time")
        # but rejects "One More Time Club Mix" vs "One More Time" since the
        # superset would contain extra words.
        if not inc_tokens.issubset(ex_tokens):
            continue
        # And token counts must be close enough that we're not matching a
        # very long existing entry that happens to contain all the incoming
        # words.
        if len(ex_tokens) - len(inc_tokens) > 2:
            continue
        ratio = SequenceMatcher(None, inc_combined, entry.combined).ratio()
        key = (confirmed, ratio)
        if ratio >= ratio_threshold and (best is None or key > best[0]):
            best = (key, match)

    return best[1] if best else None


__all__ = [
    "DuplicateMatch",
    "find_duplicate",
    "index_existing",
    "match_duplicate",
    "normalize",
    "version_words",
]
