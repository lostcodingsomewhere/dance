"""Structural mix-compatibility scorer.

Answers: "how cleanly can we bring candidate track B in over track A?"

Phrase-aligned DJ transitions work by riding A's outro under B's intro.
The longer and cleaner both windows are, the more room the DJ has to work
with.  This module turns that structural data into a single ``[0, 1]`` signal
(``transition_fit``) that :func:`dance.recommender.scoring.combine` can weight
alongside harmonic/BPM/embedding features.

Architecture
------------
* :class:`TrackStructure` — pure dataclass capturing the mix-relevant
  shape of a track (intro/outro length, usability flags).
* :func:`load_structure` — the single DB-touching function; reads full-track
  SECTION regions labelled *intro* or *outro*.
* :func:`transition_fit` — pure scorer, ``[0, 1]`` or ``None``.

All module-level thresholds are named constants so callers can read and
override them without magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from dance.core.database import Region, RegionType, SectionLabel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum intro/outro length (in bars) we consider "clean" for a mix-in/mix-out.
#: 16 bars = standard 32-beat phrase at most tempos; below this there is not
#: enough room to blend or EQ-ride without risk.
MIN_MIX_BARS: int = 16

#: The "ideal" mutual window: an intro AND outro both at or above this many bars
#: gives maximum headroom for a long, comfortable blend.  Scores are normalized
#: against this cap so that ludicrously long intros (64+ bars) do not push past 1.
MAX_USEFUL_BARS: int = 32

#: Score returned when *only* the candidate is available (current track unknown
#: / nothing is playing).  We reward a candidate that has *both* a clean intro
#: (mixable in) and a clean outro (mixable out again) because that makes it safe
#: to place anywhere in a set.
_CANDIDATE_ONLY_CLEAN_BOTH: float = 0.8

#: Score when only the candidate's intro is clean (mixable in but harder to
#: transition out of).
_CANDIDATE_ONLY_CLEAN_INTRO: float = 0.5

#: Baseline score when the candidate has neither a clean intro nor outro —
#: it may still be playable but offers little structural headroom.
_CANDIDATE_ONLY_FALLBACK: float = 0.15

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrackStructure:
    """Mix-relevant structural summary of a single track.

    Populated from full-track SECTION regions (``stem_file_id IS NULL``)
    labelled *intro* or *outro*.  Any field can be ``None`` when the track
    has not been region-detected yet, or when the relevant section is absent.

    Attributes
    ----------
    intro_bars:
        Length of the intro section in bars, or ``None`` if absent / unknown.
    outro_bars:
        Length of the outro section in bars, or ``None`` if absent / unknown.
    has_clean_intro:
        ``True`` when ``intro_bars >= MIN_MIX_BARS`` — i.e. there is enough
        room to mix *into* this track.
    has_clean_outro:
        ``True`` when ``outro_bars >= MIN_MIX_BARS`` — i.e. there is enough
        room to mix *out of* this track.
    """

    intro_bars: int | None
    outro_bars: int | None
    has_clean_intro: bool
    has_clean_outro: bool


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------


def load_structure(session: Session, track_id: int) -> TrackStructure:
    """Load mix-relevant structure for *track_id* from the database.

    Reads full-track (``stem_file_id IS NULL``) SECTION regions whose
    ``section_label`` is *intro* or *outro*.  When multiple intro/outro
    regions exist (unusual but possible with auto-detectors), the **longest**
    is used because it represents the most headroom available.

    Prefers ``length_bars`` on the Region row; bar-from-ms derivation is
    intentionally omitted — BPM lives on the audio_analysis table, not here,
    and a cross-join would couple this loader to more of the schema than
    justified for a heuristic score.

    Parameters
    ----------
    session:
        Active SQLAlchemy session.
    track_id:
        Primary key of the track to inspect.

    Returns
    -------
    TrackStructure
        All fields may be ``None`` / ``False`` when data is absent.
    """
    rows = (
        session.query(Region)
        .filter(
            Region.track_id == track_id,
            Region.stem_file_id.is_(None),
            Region.region_type == RegionType.SECTION.value,
            Region.section_label.in_(
                [SectionLabel.INTRO.value, SectionLabel.OUTRO.value]
            ),
        )
        .all()
    )

    intro_bars: int | None = None
    outro_bars: int | None = None

    for row in rows:
        bars: int | None = row.length_bars
        if bars is None:
            continue
        if row.section_label == SectionLabel.INTRO.value:
            intro_bars = bars if intro_bars is None else max(intro_bars, bars)
        elif row.section_label == SectionLabel.OUTRO.value:
            outro_bars = bars if outro_bars is None else max(outro_bars, bars)

    return TrackStructure(
        intro_bars=intro_bars,
        outro_bars=outro_bars,
        has_clean_intro=intro_bars is not None and intro_bars >= MIN_MIX_BARS,
        has_clean_outro=outro_bars is not None and outro_bars >= MIN_MIX_BARS,
    )


def load_structures(
    session: Session, track_ids: list[int]
) -> dict[int, TrackStructure]:
    """Batch variant of :func:`load_structure` — one query for many tracks.

    Avoids the N+1 pattern when scoring a whole candidate pool (the planner
    scores every track in the library). Tracks with no intro/outro section
    rows simply get an all-``None`` :class:`TrackStructure`.
    """
    if not track_ids:
        return {}
    rows = (
        session.query(Region)
        .filter(
            Region.track_id.in_(track_ids),
            Region.stem_file_id.is_(None),
            Region.region_type == RegionType.SECTION.value,
            Region.section_label.in_(
                [SectionLabel.INTRO.value, SectionLabel.OUTRO.value]
            ),
        )
        .all()
    )
    intro: dict[int, int] = {}
    outro: dict[int, int] = {}
    for row in rows:
        bars: int | None = row.length_bars
        if bars is None:
            continue
        tid = int(row.track_id)
        if row.section_label == SectionLabel.INTRO.value:
            intro[tid] = max(intro.get(tid, 0), bars)
        elif row.section_label == SectionLabel.OUTRO.value:
            outro[tid] = max(outro.get(tid, 0), bars)

    out: dict[int, TrackStructure] = {}
    for tid in track_ids:
        i = intro.get(tid)
        o = outro.get(tid)
        out[int(tid)] = TrackStructure(
            intro_bars=i,
            outro_bars=o,
            has_clean_intro=i is not None and i >= MIN_MIX_BARS,
            has_clean_outro=o is not None and o >= MIN_MIX_BARS,
        )
    return out


# ---------------------------------------------------------------------------
# Pure scorer
# ---------------------------------------------------------------------------


def transition_fit(
    current: TrackStructure | None,
    candidate: TrackStructure | None,
) -> float | None:
    """Score the structural suitability of a transition from *current* to *candidate*.

    Returns a value in ``[0, 1]`` or ``None`` when there is genuinely nothing
    to judge (both inputs are ``None``).

    Metric rationale
    ----------------
    A DJ mixes *out of* the current track's outro and *into* the candidate's
    intro.  Two factors matter:

    1. **Usability flags** — does the current have a clean outro, does the
       candidate have a clean intro?  Missing either halves the base score.
    2. **Mutual headroom** — the actual bar count of the narrower window
       (``min(outro_bars, intro_bars)``) normalized against :data:`MAX_USEFUL_BARS`.
       More bars = smoother blend possible.

    The combination is:

    .. code-block:: text

        flag_score   = 0.5 * has_clean_outro + 0.5 * has_clean_intro  → [0, 1]
        mutual_bars  = min(current.outro_bars, candidate.intro_bars), capped at MAX_USEFUL_BARS
        bar_score    = mutual_bars / MAX_USEFUL_BARS                   → [0, 1]
        result       = 0.5 * flag_score + 0.5 * bar_score

    When bar counts are unavailable (``None``) the bar component is skipped and
    the score is derived from flags alone, renormalized to ``[0, 1]``.

    When *current* is ``None`` (e.g. first track in a set, or planner has no
    context), fall back to a "standalone mixability" score based solely on the
    candidate's own clean intro *and* outro — a track that is easy to both mix
    into and out of is universally safe.

    Parameters
    ----------
    current:
        Structure of the track currently playing / outgoing.  ``None`` when
        there is no context (first pick, planner mode).
    candidate:
        Structure of the track being evaluated for the next slot.

    Returns
    -------
    float | None
        ``[0, 1]`` when at least one side has structural data, ``None`` when
        both are absent.
    """
    # ---- total absence -------------------------------------------------------
    if candidate is None and current is None:
        return None

    # ---- candidate-only fallback (no "current" context) ---------------------
    if current is None:
        # candidate is not None here
        assert candidate is not None
        if candidate.has_clean_intro and candidate.has_clean_outro:
            return _CANDIDATE_ONLY_CLEAN_BOTH
        if candidate.has_clean_intro:
            return _CANDIDATE_ONLY_CLEAN_INTRO
        return _CANDIDATE_ONLY_FALLBACK

    # ---- current is present; candidate may still be None --------------------
    if candidate is None:
        # We know the outgoing track but nothing about the candidate.
        # Not enough to score — return None so combine skips the feature.
        return None

    # ---- both sides present -------------------------------------------------
    # Flag component: does each side contribute its half of a clean transition?
    flag_score = 0.5 * float(current.has_clean_outro) + 0.5 * float(
        candidate.has_clean_intro
    )

    # Bar-count component: mutual headroom (narrower window, capped).
    outro_b = current.outro_bars
    intro_b = candidate.intro_bars

    if outro_b is not None and intro_b is not None:
        mutual = min(outro_b, intro_b)
        bar_score = min(1.0, mutual / MAX_USEFUL_BARS)
        return 0.5 * flag_score + 0.5 * bar_score
    else:
        # No bar data on at least one side — fall back to flags alone.
        return flag_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MAX_USEFUL_BARS",
    "MIN_MIX_BARS",
    "TrackStructure",
    "load_structure",
    "load_structures",
    "transition_fit",
]
