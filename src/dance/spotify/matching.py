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
3. :func:`rank_candidates` — score, apply the hard gates and the minimum
   score, and order eligible candidates best-first.

Hard gates, checked before the score and **never relaxed** on retry:

- duration within max(30 s, 10 %) of Spotify's (or at most 12 min when
  Spotify's is unknown);
- no unrequested different-audio marker in the title (live, cover, remix,
  slowed, VIP, ...) — :data:`HARD_VERSION_MARKERS`;
- the requested artists are credited in the title or own the channel —
  :func:`artist_credited`.

A track that has no eligible candidate fails with a message naming the best
near-miss; a human can then decide. Better no file than a confident wrong
one.

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
#: dance: the longest upload accepted when there is no expected duration to
#: gate on (see :func:`passes_duration_gate`). Club and progressive cuts run
#: to ~10 min; past 12 it is a mix, a set or a loop.
UNKNOWN_TARGET_MAX_S = 720.0
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
    # dance: producer variants — different audio under the same title.
    # "VIP Mix" at the studio length otherwise scored 0.98.
    ("vip", 0.18),
    ("bootleg", 0.18),
    ("rework", 0.18),
    ("flip", 0.18),
    ("dub", 0.18),
    # Hard-rejected as a phrase (below); "dub" already carries the penalty.
    ("dub mix", 0.0),
    # dance: partial audio. A "(Preview)" of "Vision Blurred" ran 202 s vs
    # Spotify's 186 s — inside the search gate — and scored 0.92 in the
    # live search results.
    ("preview", 0.18),
    ("snippet", 0.18),
    ("teaser", 0.18),
)
MAX_VERSION_PENALTY = 0.45
_SINGLE_WORD_MARKERS = frozenset(m for m, _ in VERSION_PENALTIES if " " not in m)

#: dance: markers that are *always* different audio — a hard rejection, not
#: just a penalty, when the request does not name them. djbooth only
#: penalises, so a live take or an unrequested remix of about the right
#: length still cleared the bar ("One More Time (Live)" at 322 s scored
#: 0.80 for the 320 s studio cut, a "VIP Mix" 0.98). The softer markers
#: stay score-only: "extended" and "hour" uploads are caught by the duration
#: gate anyway; "mixed" / "hour" also occur in legitimate titles ("Rush Hour
#: Records"), and a bare "dub" in genre tags ("[Dub Techno]") — only the
#: phrase "dub mix" is a hard rejection.
HARD_VERSION_MARKERS = frozenset(
    {
        "live",
        "cover",
        "karaoke",
        "slowed",
        "sped up",
        "nightcore",
        "remix",
        "instrumental",
        "acapella",
        "a cappella",
        "8d",
        "vip",
        "bootleg",
        "rework",
        "flip",
        "dub mix",
        "preview",
        "snippet",
        "teaser",
    }
)
#: Phrases containing a marker word that do not describe the audio. Removed
#: from upload titles before markers are looked for: "(Cover Art)" is how
#: many official audio-only uploads label their still image.
_BENIGN_PHRASES: tuple[str, ...] = ("cover art", "album cover")

#: dance: hard identity gate (djbooth has none). An exact title, a matching
#: length and an official-looking uploader add up to 0.79 — above
#: :data:`MIN_SCORE` with *zero* artist match — so "Surrender" by a
#: different act on a "- Topic" channel was eligible. A candidate must show
#: the credited artists in its title (more than this share of their words),
#: or name only credited artists before the title's " - ", or come from a
#: channel named only for credited artists (``Sian - Topic``).
MIN_TITLE_ARTIST_COVERAGE = 0.5
#: Channel-name words that say what kind of channel it is, not whose.
_CHANNEL_WORDS = frozenset(
    {
        "topic",
        "official",
        "vevo",
        "music",
        "records",
        "record",
        "recordings",
        "tv",
        "channel",
        "youtube",
        "and",
        "the",
        "x",
        "vs",
    }
)
#: The same words glued on without a space: ``CamelPhatVEVO``.
_CHANNEL_SUFFIX_RE = re.compile(r"(?:vevo|official|music|tv|records?)+$")
#: "Artist - Title" separators (hyphen, en dash, em dash, bar) with spaces.
_CREDIT_SEPARATOR_RE = re.compile(r"\s[-–—|]\s")

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
    #: Whether the upload credits the requested artists (title or channel).
    artist_ok: bool = True

    @property
    def eligible(self) -> bool:
        return not self.rejection

    def describe(self) -> str:
        return f"{self.score:.2f}: {self.candidate.describe()}"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_match_text(value: str | None) -> str:
    """Lowercase, strip accents, ``&`` -> ``and``, drop ``feat.``, words only.

    Ported from djbooth ``normalizeMatchText``: every Unicode letter and
    number is kept (``[^\\p{Letter}\\p{Number}]+`` -> space), so Cyrillic,
    CJK, Greek or Arabic titles normalise to themselves rather than to an
    empty string. dance: recomposed with NFC after the accents are dropped,
    so Hangul stays as syllables instead of loose jamo.
    """
    if not value:
        return ""
    folded = unicodedata.normalize("NFKD", value)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = unicodedata.normalize("NFC", folded)
    folded = folded.lower().replace("&", " and ")
    folded = _FEAT_RE.sub(" ", folded)
    folded = "".join(ch if unicodedata.category(ch)[0] in "LN" else " " for ch in folded)
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


_IGNORED_ARTIST_TOKENS = frozenset({"and", "the", "vs"})


def _artist_tokens(normalized: str) -> list[str]:
    """Meaningful words of a normalised artist credit, de-duplicated.

    Single Latin letters and joiners ("and", "the", "vs") carry no identity.
    dance: a single *non-ASCII* character is kept — in CJK one character
    can be a whole name.
    """
    return [
        t
        for t in dict.fromkeys(normalized.split())
        if (len(t) > 1 or not t.isascii()) and t not in _IGNORED_ARTIST_TOKENS
    ]


def token_coverage(expected: str, actual: str) -> float:
    """Share of expected artist tokens present anywhere in ``actual``."""
    exp_tokens = _artist_tokens(expected)
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
    called "Live" is not penalised for its own name. A phrase marker counts
    as requested when a single-word marker inside it is (a "... - Dub"
    request accepts a "(Dub Mix)" upload).
    """
    requested = normalize_match_text(" ".join((target.title, *target.artists)))
    actual = f" {normalize_match_text(candidate_title)} "
    for phrase in _BENIGN_PHRASES:
        actual = actual.replace(f" {phrase} ", " ")
    actual = actual.strip()

    def is_requested(marker: str) -> bool:
        return _contains_phrase(requested, marker) or any(
            word in _SINGLE_WORD_MARKERS and _contains_phrase(requested, word)
            for word in marker.split()
        )

    return tuple(
        marker
        for marker, _ in VERSION_PENALTIES
        if _contains_phrase(actual, marker) and not is_requested(marker)
    )


def version_penalty(markers: Iterable[str]) -> float:
    weights = dict(VERSION_PENALTIES)
    return min(MAX_VERSION_PENALTY, sum(weights.get(m, 0.0) for m in markers))


def _names_only_artists(text: str | None, artist_words: set[str], ignore: frozenset[str]) -> bool:
    """True when every meaningful word of ``text`` is a credited-artist word.

    ``ignore`` words (channel decoration, joiners) are skipped; a word with
    a channel suffix glued on (``camelphatvevo``) counts by its stem. At
    least one artist word must be present.
    """
    found = False
    for word in normalize_match_text(text).split():
        if word in ignore or (len(word) == 1 and word.isascii()):
            continue
        stem = _CHANNEL_SUFFIX_RE.sub("", word)
        if word in artist_words or (stem and stem in artist_words):
            found = True
            continue
        return False
    return found


def artist_credited(target: Target, candidate: Candidate) -> bool:
    """Hard identity gate: does this upload credit the requested artists?

    dance: not in djbooth (see :data:`MIN_TITLE_ARTIST_COVERAGE`). Passes
    when any of these holds:

    1. more than half of the credited artists' words appear in the title
       (``Sian, KASIA & Burko - Surrender``);
    2. the part of the title before the first `` - `` names only credited
       artists (``Kaskade - Vision Blurred`` for a three-artist credit);
    3. the channel is named only for credited artists, ignoring channel
       words (``Sian - Topic``, ``CamelPhatVEVO``).

    A channel that merely *contains* an artist's word does not count:
    ``Dr Tammy Sian`` is not ``Sian``.
    """
    words = _artist_tokens(normalize_match_text(" ".join(target.artists)))
    if not words:
        return True  # nothing identifying to check (e.g. a one-letter act)
    artist_words = set(words)
    title_words = set(normalize_match_text(candidate.title).split())
    coverage = sum(1 for w in words if w in title_words) / len(words)
    if coverage > MIN_TITLE_ARTIST_COVERAGE:
        return True
    parts = _CREDIT_SEPARATOR_RE.split(candidate.title, maxsplit=1)
    if len(parts) == 2 and _names_only_artists(
        parts[0], artist_words, _IGNORED_ARTIST_TOKENS | {"x"}
    ):
        return True
    return _names_only_artists(candidate.uploader, artist_words, _CHANNEL_WORDS)


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


def _known(seconds: float | None) -> bool:
    return seconds is not None and seconds > 0


def passes_duration_gate(expected_s: float | None, actual_s: float | None) -> bool:
    """Hard pre-download gate. Never relaxed.

    dance: an *unknown candidate* duration always fails — djbooth lets it
    through, but an unmeasurable upload is exactly how an hour-long set
    slips past a filter. With no expected duration (a CSV row without one,
    a Spotify ``duration_ms`` of 0) the candidate must still be at most
    :data:`UNKNOWN_TARGET_MAX_S` long: otherwise "Artist - Title (Full Set)"
    scores 0.85 on name alone.
    """
    if not _known(actual_s):
        return False
    assert actual_s is not None
    if not _known(expected_s):
        return actual_s <= UNKNOWN_TARGET_MAX_S
    assert expected_s is not None
    return abs(expected_s - actual_s) <= duration_gate_s(expected_s)


def verify_tolerance_s(expected_s: float) -> float:
    """Post-download tolerance: max(10 s, 3.5 %) (djbooth verifyAudioFile)."""
    return max(10.0, expected_s * 0.035)


def passes_verification(expected_s: float | None, measured_s: float) -> bool:
    """Post-download length check. Unknown expected → the length cap."""
    if not _known(expected_s):
        return measured_s <= UNKNOWN_TARGET_MAX_S + verify_tolerance_s(UNKNOWN_TARGET_MAX_S)
    assert expected_s is not None
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
        hard_markers = [m for m in markers if m in HARD_VERSION_MARKERS]
        artist_ok = artist_credited(target, cand)
        if cand.live_status in _LIVE_STATUSES:
            rejection = f"live stream ({cand.live_status})"
        elif not duration_ok:
            if not _known(cand.duration_s):
                rejection = "duration unknown"
            elif not _known(target.duration_s):
                rejection = (
                    f"duration {cand.duration_s:.0f}s > {UNKNOWN_TARGET_MAX_S:.0f}s cap "
                    "(no expected duration)"
                )
            else:
                assert target.duration_s is not None
                rejection = (
                    f"duration {cand.duration_s:.0f}s vs {target.duration_s:.0f}s "
                    f"(gate ±{duration_gate_s(target.duration_s):.0f}s)"
                )
        elif hard_markers:
            rejection = f"different version ({', '.join(hard_markers)})"
        elif not artist_ok:
            rejection = "artist not credited in title or channel"
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
                artist_ok=artist_ok,
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
    "HARD_VERSION_MARKERS",
    "MAX_DOWNLOAD_ATTEMPTS",
    "MIN_SCORE",
    "SEARCH_LIMIT",
    "UNKNOWN_TARGET_MAX_S",
    "Candidate",
    "ScoredCandidate",
    "Target",
    "artist_credited",
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
