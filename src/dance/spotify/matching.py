"""Pick the right YouTube upload for a Spotify track. Pure: no I/O.

The old downloader asked yt-dlp for ``ytsearch1:{artist} {title}`` — the
single top hit — and, when the ±30 s ``--match-filter`` rejected it, retried
with no filter at all. That accepted whatever ranked first: live versions,
extended mixes, hour-long sets. The file was then re-tagged with the Spotify
identity, so the wrong audio looked right. Measured on the real library:
22 of 278 files differ from the Spotify duration by more than 10 s, 11 by
more than 60 s (``Cycles``: a 414 s file for a 186 s track).

This module replaces "take the top hit" with "score every candidate, gate
hard on duration, and only accept something that scores well":

1. :func:`search_queries` — the queries to send (all credited artists first,
   the primary artist as a fallback when nothing qualifies).
2. :func:`parse_search_results` — turn ``yt-dlp --flat-playlist -J`` output
   into :class:`Candidate` rows.
3. :func:`rank_candidates` — score, apply the duration gate and the minimum
   score, and order eligible candidates best-first.

The duration gate is **never relaxed** on retry. A track that has no
eligible candidate fails with a message naming the best near-miss; a human
can then decide. Better no file than a confident wrong one.

Scoring weights, version-marker penalties, the minimum score, the duration
gate and the post-download tolerance are ported from djbooth (MIT licence,
``packages/tracks-ts/src/acquisition.ts`` and ``acquire-manifest.ts``).
Deviations from the port are marked ``dance:`` inline.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants — ported from djbooth (MIT), see module docstring.
# ---------------------------------------------------------------------------

#: Identity-score weights (djbooth acquisition.ts ``scoreYoutubeCandidate``).
W_TITLE = 0.48
W_ARTIST = 0.25
W_DURATION = 0.27
#: Bonus for an uploader that looks like the label / artist channel.
OFFICIAL_BONUS = 0.04
#: Minimum identity score to download a candidate (acquire-manifest.ts).
MIN_SCORE = 0.78
#: Results requested per search (``ytsearch10:``).
SEARCH_LIMIT = 10
#: Never download more than this many candidates for one track. Each failed
#: attempt costs YouTube a full media fetch; three is plenty to route around
#: one broken upload without hammering the service.
MAX_DOWNLOAD_ATTEMPTS = 3

#: Version markers that make an upload a *different recording* unless the
#: requested title (or artist credit) contains them. Per-marker penalty,
#: summed and capped at :data:`MAX_VERSION_PENALTY`.
#: djbooth acquisition.ts ``unwantedVersionPenalty``: live/cover/karaoke/
#: slowed/sped up/nightcore .18 each, mixed .12, remix .10, cap .45.
VERSION_PENALTIES: tuple[tuple[str, float], ...] = (
    ("live", 0.18),
    ("cover", 0.18),
    ("karaoke", 0.18),
    ("slowed", 0.18),
    ("sped up", 0.18),
    ("nightcore", 0.18),
    ("mixed", 0.12),
    ("remix", 0.10),
    # dance: extra markers seen in the real library's mis-picks. "extended"
    # is the 186 s -> 414 s ``Cycles`` case; "hour" catches "1 hour loop"
    # uploads; instrumental/acapella/8d are different audio, not the track.
    ("extended", 0.10),
    ("instrumental", 0.18),
    ("acapella", 0.18),
    ("a cappella", 0.18),
    ("8d", 0.18),
    ("hour", 0.18),
)
MAX_VERSION_PENALTY = 0.45

#: yt-dlp ``live_status`` values that can never be the studio recording.
_LIVE_STATUSES = frozenset({"is_live", "is_upcoming", "post_live"})

#: YouTube video ids are 11 chars of [A-Za-z0-9_-]. Anything else is not
#: something we will hand to yt-dlp as a URL.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_OFFICIAL_UPLOADER_RE = re.compile(
    # dance: + "vevo" (label channels like "CamelPhatVEVO").
    r"(?:topic|official|records?|music|vevo)$",
    re.IGNORECASE,
)
_FEAT_RE = re.compile(r"\b(?:feat|featuring|ft)\b\.?")
_NON_WORD_RE = re.compile(r"[^0-9a-z]+")


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """What we want: the Spotify identity of one recording."""

    title: str
    artists: tuple[str, ...]
    duration_s: float | None = None

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else ""


@dataclass(frozen=True)
class Candidate:
    """One search result from ``yt-dlp --flat-playlist -J ytsearchN:...``."""

    video_id: str
    title: str
    uploader: str | None = None
    duration_s: float | None = None
    live_status: str | None = None
    #: Position in the search results (0 = YouTube's top hit). Breaks score
    #: ties in favour of YouTube's own relevance order.
    rank: int = 0

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def describe(self) -> str:
        dur = f"{self.duration_s:.0f}s" if self.duration_s is not None else "?s"
        return f"{self.title} {dur}"


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    duration_ok: bool
    #: Empty when eligible; otherwise why this candidate was rejected.
    rejection: str = ""
    #: Version markers found in the upload title that the target lacks.
    unwanted_markers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def eligible(self) -> bool:
        return not self.rejection

    def describe(self) -> str:
        return f"{self.score:.2f}: {self.candidate.describe()}"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_match_text(value: str | None) -> str:
    """Lowercase, ASCII-fold, ``&`` -> ``and``, drop ``feat.``, words only.

    Ported from djbooth ``normalizeMatchText``.
    """
    if not value:
        return ""
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower().replace("&", " and ")
    folded = _FEAT_RE.sub(" ", folded)
    folded = _NON_WORD_RE.sub(" ", folded)
    return " ".join(folded.split())


def split_artists(artist_field: str | None) -> tuple[str, ...]:
    """Split a credited-artist string into names.

    Spotify ingest joins credits with ``", "``; older Exportify CSVs use
    ``;``. When the field has commas, only commas split it — a name can
    itself contain ``;`` (``TH;EN``, seen live in the real library).
    ``&`` is never a separator — ``Faul & Wad`` is one act.
    """
    if not artist_field:
        return ()
    sep = "," if "," in artist_field else ";"
    parts = artist_field.split(sep)
    return tuple(p.strip() for p in parts if p.strip())


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Word-boundary containment on normalised text ("live" is not in "alive")."""
    if not haystack or not phrase:
        return False
    return f" {phrase} " in f" {haystack} "


# ---------------------------------------------------------------------------
# Scoring components — ported from djbooth acquisition.ts (MIT)
# ---------------------------------------------------------------------------


def text_similarity(expected: str, actual: str) -> float:
    """djbooth ``textSimilarity`` on normalised strings.

    dance: containment is word-bounded, so ``cycles`` is not found inside
    ``recycles``.
    """
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 1.0
    if _contains_phrase(actual, expected):
        return 0.96
    exp_tokens = set(expected.split())
    act_tokens = set(actual.split())
    inter = len(exp_tokens & act_tokens)
    union = len(exp_tokens | act_tokens)
    jaccard = inter / union if union else 0.0
    coverage = inter / len(exp_tokens) if exp_tokens else 0.0
    return min(1.0, jaccard * 0.45 + coverage * 0.55)


def token_coverage(expected: str, actual: str) -> float:
    """Share of expected artist tokens present anywhere in ``actual``."""
    ignored = {"and", "the", "vs"}
    exp_tokens = [t for t in dict.fromkeys(expected.split()) if len(t) > 1 and t not in ignored]
    if not exp_tokens:
        return 0.5
    act_tokens = set(actual.split())
    return sum(1 for t in exp_tokens if t in act_tokens) / len(exp_tokens)


def duration_similarity(expected: float | None, actual: float | None) -> float:
    """1.0 inside max(8 s, 2.5 %), then a linear fall-off (djbooth)."""
    if expected is None or actual is None or expected <= 0 or actual <= 0:
        return 0.45
    diff = abs(expected - actual)
    tolerance = max(8.0, expected * 0.025)
    if diff <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (diff - tolerance) / max(35.0, expected * 0.2))


def unwanted_markers(target: Target, candidate_title: str) -> tuple[str, ...]:
    """Version markers in the upload title that the request does not ask for.

    dance: the request is the title *and* the artist credit, so a band
    called "Live" is not penalised for its own name.
    """
    requested = normalize_match_text(" ".join((target.title, *target.artists)))
    actual = normalize_match_text(candidate_title)
    return tuple(
        marker
        for marker, _ in VERSION_PENALTIES
        if _contains_phrase(actual, marker) and not _contains_phrase(requested, marker)
    )


def version_penalty(markers: Iterable[str]) -> float:
    weights = dict(VERSION_PENALTIES)
    return min(MAX_VERSION_PENALTY, sum(weights.get(m, 0.0) for m in markers))


def score_candidate(target: Target, candidate: Candidate) -> float:
    """Identity score in [0, 1] (djbooth ``scoreYoutubeCandidate``)."""
    target_title = normalize_match_text(target.title)
    candidate_title = normalize_match_text(candidate.title)
    target_artists = normalize_match_text(" ".join(target.artists))
    candidate_artists = normalize_match_text(
        " ".join(x for x in (candidate.uploader, candidate.title) if x)
    )
    title_score = text_similarity(target_title, candidate_title)
    artist_score = token_coverage(target_artists, candidate_artists)
    dur_score = duration_similarity(target.duration_s, candidate.duration_s)
    official = (
        OFFICIAL_BONUS
        if candidate.uploader and _OFFICIAL_UPLOADER_RE.search(candidate.uploader.strip())
        else 0.0
    )
    penalty = version_penalty(unwanted_markers(target, candidate.title))
    raw = (
        title_score * W_TITLE
        + artist_score * W_ARTIST
        + dur_score * W_DURATION
        + official
        - penalty
    )
    return round(max(0.0, min(1.0, raw)), 3)


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def duration_gate_s(expected_s: float) -> float:
    """Allowed |Δ| between Spotify and a search result: max(30 s, 10 %).

    djbooth acquire-manifest.ts ``hasWrongDuration``.
    """
    return max(30.0, expected_s * 0.10)


def passes_duration_gate(expected_s: float | None, actual_s: float | None) -> bool:
    """Hard pre-download gate. Never relaxed.

    No expected duration (a CSV row without one) means there is nothing to
    gate on. dance: an *unknown candidate* duration fails the gate when the
    target is known — djbooth lets it through, but an unmeasurable upload is
    exactly how an hour-long set slips past a filter.
    """
    if expected_s is None or expected_s <= 0:
        return True
    if actual_s is None or actual_s <= 0:
        return False
    return abs(expected_s - actual_s) <= duration_gate_s(expected_s)


def verify_tolerance_s(expected_s: float) -> float:
    """Post-download tolerance: max(10 s, 3.5 %) (djbooth verifyAudioFile)."""
    return max(10.0, expected_s * 0.035)


def passes_verification(expected_s: float | None, measured_s: float) -> bool:
    if expected_s is None or expected_s <= 0:
        return True
    return abs(expected_s - measured_s) <= verify_tolerance_s(expected_s)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_candidates(
    target: Target,
    candidates: Sequence[Candidate],
    min_score: float = MIN_SCORE,
) -> list[ScoredCandidate]:
    """Score every candidate; eligible ones first, each group best-first.

    Duplicate video ids (the same upload returned by two queries) are
    collapsed to one row.
    """
    seen: set[str] = set()
    scored: list[ScoredCandidate] = []
    for cand in candidates:
        if cand.video_id in seen:
            continue
        seen.add(cand.video_id)
        score = score_candidate(target, cand)
        duration_ok = passes_duration_gate(target.duration_s, cand.duration_s)
        markers = unwanted_markers(target, cand.title)
        if cand.live_status in _LIVE_STATUSES:
            rejection = f"live stream ({cand.live_status})"
        elif not duration_ok:
            if cand.duration_s is None:
                rejection = "duration unknown"
            else:
                assert target.duration_s is not None
                rejection = (
                    f"duration {cand.duration_s:.0f}s vs {target.duration_s:.0f}s "
                    f"(gate ±{duration_gate_s(target.duration_s):.0f}s)"
                )
        elif score < min_score:
            rejection = f"score {score:.2f} < {min_score:.2f}"
        else:
            rejection = ""
        scored.append(
            ScoredCandidate(
                candidate=cand,
                score=score,
                duration_ok=duration_ok,
                rejection=rejection,
                unwanted_markers=markers,
            )
        )
    scored.sort(key=lambda s: (not s.eligible, -s.score, s.candidate.rank, s.candidate.video_id))
    return scored


def eligible(ranked: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    return [s for s in ranked if s.eligible]


def no_candidate_message(ranked: Sequence[ScoredCandidate]) -> str:
    """The failure message when nothing qualified — names the best near-miss."""
    if not ranked:
        return "no candidate passed (search returned nothing)"
    best = max(ranked, key=lambda s: s.score)
    return f"no candidate passed (best {best.describe()} — {best.rejection or 'rejected'})"


# ---------------------------------------------------------------------------
# Search I/O shapes (pure — the subprocess lives in acquire.py)
# ---------------------------------------------------------------------------


def search_queries(target: Target) -> list[str]:
    """Queries in the order they should be tried.

    The first credits every artist; uploads often credit only the main one,
    so the primary-artist variant is the fallback when the first query yields
    nothing eligible (djbooth ytdlp-library.ts ``search``). Scoring always
    uses the complete identity regardless of which query found a candidate.
    """
    full = " ".join((*target.artists, target.title)).strip()
    queries = [full]
    if len(target.artists) > 1:
        primary = f"{target.primary_artist} {target.title}".strip()
        if primary != full:
            queries.append(primary)
    return queries


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_search_results(payload: Any, rank_offset: int = 0) -> list[Candidate]:
    """Parse ``yt-dlp --flat-playlist -J ytsearchN:...`` JSON.

    Entries without a valid 11-char video id or a title are dropped — we
    only ever download by id, never by free-text query. ``rank`` is the
    entry's position plus ``rank_offset`` (so a fallback query's results
    rank after the first query's).
    """
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    out: list[Candidate] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        vid = entry.get("id")
        title = entry.get("title")
        if not isinstance(vid, str) or not _VIDEO_ID_RE.match(vid):
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        uploader = entry.get("uploader") or entry.get("channel")
        live_status = entry.get("live_status")
        out.append(
            Candidate(
                video_id=vid,
                title=title.strip(),
                uploader=str(uploader) if uploader else None,
                duration_s=_float_or_none(entry.get("duration")),
                live_status=str(live_status) if live_status else None,
                rank=rank_offset + position,
            )
        )
    return out


__all__ = [
    "MAX_DOWNLOAD_ATTEMPTS",
    "MIN_SCORE",
    "SEARCH_LIMIT",
    "Candidate",
    "ScoredCandidate",
    "Target",
    "duration_gate_s",
    "eligible",
    "no_candidate_message",
    "normalize_match_text",
    "parse_search_results",
    "passes_duration_gate",
    "passes_verification",
    "rank_candidates",
    "score_candidate",
    "search_queries",
    "split_artists",
    "verify_tolerance_s",
]
