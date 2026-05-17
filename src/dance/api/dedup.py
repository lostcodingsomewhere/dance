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

We never block ingest on a dupe match; we only flag them so the UI can show
"these N look already-loaded, want to skip?".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable


_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


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


@dataclass(frozen=True)
class _ExistingEntry:
    track_id: int
    artist_norm: str
    title_norm: str
    combined: str


def index_existing(
    rows: Iterable[tuple[int, str | None, str | None]],
) -> list[_ExistingEntry]:
    """Build a searchable index from ``(track_id, artist, title)`` tuples."""
    out: list[_ExistingEntry] = []
    for track_id, artist, title in rows:
        a = normalize(artist or "")
        t = normalize(title or "")
        out.append(
            _ExistingEntry(
                track_id=track_id,
                artist_norm=a,
                title_norm=t,
                combined=f"{a} {t}".strip(),
            )
        )
    return out


def find_duplicate(
    incoming_artist: str,
    incoming_title: str,
    index: list[_ExistingEntry],
    ratio_threshold: float = 0.92,
) -> int | None:
    """Return the track_id of the most likely duplicate, or ``None``.

    Two-stage:

    1. Token-set: incoming tokens (combined artist+title) must all appear in
       the existing combined string. Cheap, catches reorderings.
    2. SequenceMatcher ratio on the combined strings ≥ threshold for
       ambiguous cases where lengths differ slightly.
    """
    inc_combined = f"{normalize(incoming_artist)} {normalize(incoming_title)}".strip()
    if not inc_combined:
        return None
    inc_tokens = set(inc_combined.split())

    best: tuple[float, int] | None = None
    for entry in index:
        if not entry.combined:
            continue
        ex_tokens = set(entry.combined.split())
        # Quick win: identical token sets.
        if inc_tokens == ex_tokens:
            return entry.track_id
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
        if ratio >= ratio_threshold and (best is None or ratio > best[0]):
            best = (ratio, entry.track_id)

    return best[1] if best else None


__all__ = ["find_duplicate", "index_existing", "normalize"]
