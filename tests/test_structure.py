"""Tests for :mod:`dance.recommender.structure`.

Covers:
* Pure-function behaviour of :func:`transition_fit` (no DB).
* :func:`load_structure` against a real in-memory SQLite session (using the
  ``session`` + ``make_track`` fixtures from conftest).
* Composition sanity: the output of ``transition_fit`` slots into
  :func:`dance.recommender.scoring.combine` under the ``transition_fit`` feature
  name without blowing up.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from dance.core.database import Region, RegionSource, RegionType, SectionLabel, now_utc
from dance.recommender import scoring as sc
from dance.recommender.structure import (
    MAX_USEFUL_BARS,
    MIN_MIX_BARS,
    TrackStructure,
    load_structure,
    transition_fit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _struct(
    intro: int | None = None,
    outro: int | None = None,
) -> TrackStructure:
    """Build a TrackStructure with explicit bar counts; flags computed from MIN_MIX_BARS."""
    return TrackStructure(
        intro_bars=intro,
        outro_bars=outro,
        has_clean_intro=intro is not None and intro >= MIN_MIX_BARS,
        has_clean_outro=outro is not None and outro >= MIN_MIX_BARS,
    )


# ---------------------------------------------------------------------------
# transition_fit — None / missing data cases
# ---------------------------------------------------------------------------


def test_both_none_returns_none():
    assert transition_fit(None, None) is None


def test_candidate_none_with_current_returns_none():
    """We know the outgoing track but not what comes next — not enough to score."""
    current = _struct(outro=32)
    assert transition_fit(current, None) is None


# ---------------------------------------------------------------------------
# transition_fit — candidate-only fallback (current is None)
# ---------------------------------------------------------------------------


def test_candidate_only_clean_both():
    cand = _struct(intro=32, outro=32)
    score = transition_fit(None, cand)
    assert score == pytest.approx(0.8)


def test_candidate_only_clean_intro_only():
    cand = _struct(intro=32, outro=4)
    score = transition_fit(None, cand)
    assert score == pytest.approx(0.5)


def test_candidate_only_no_clean_sections():
    cand = _struct(intro=8, outro=4)
    score = transition_fit(None, cand)
    assert score == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# transition_fit — perfect: long intro + long outro
# ---------------------------------------------------------------------------


def test_perfect_transition_scores_high():
    """Both sides have MAX_USEFUL_BARS — should be near 1.0."""
    current = _struct(outro=MAX_USEFUL_BARS)
    cand = _struct(intro=MAX_USEFUL_BARS)
    score = transition_fit(current, cand)
    assert score is not None
    assert score >= 0.9


def test_perfect_transition_exactly_one():
    """Exact formula check: flag_score=1.0, bar_score=1.0 → 1.0."""
    current = _struct(outro=MAX_USEFUL_BARS)
    cand = _struct(intro=MAX_USEFUL_BARS)
    score = transition_fit(current, cand)
    assert score == pytest.approx(1.0)


def test_over_max_useful_bars_capped_at_one():
    """Bars beyond MAX_USEFUL_BARS should not push the score above 1.0."""
    current = _struct(outro=64)
    cand = _struct(intro=64)
    score = transition_fit(current, cand)
    assert score is not None
    assert score <= 1.0


# ---------------------------------------------------------------------------
# transition_fit — penalized for missing clean intro/outro
# ---------------------------------------------------------------------------


def test_missing_clean_intro_penalized():
    """Candidate has a dirty intro (< MIN_MIX_BARS) — score should be low."""
    current = _struct(outro=32)
    cand = _struct(intro=4)  # too short, has_clean_intro=False
    score = transition_fit(current, cand)
    assert score is not None
    assert score < 0.5


def test_missing_clean_outro_penalized():
    """Current track has no clean outro — score should be low."""
    current = _struct(outro=4)  # too short
    cand = _struct(intro=32)
    score = transition_fit(current, cand)
    assert score is not None
    assert score < 0.5


def test_both_short_scores_near_zero():
    """No clean sections at all → score well below mid-range."""
    current = _struct(outro=4)
    cand = _struct(intro=4)
    score = transition_fit(current, cand)
    assert score is not None
    assert score < 0.2


# ---------------------------------------------------------------------------
# transition_fit — flags-only fallback when bars absent
# ---------------------------------------------------------------------------


def test_flag_only_fallback_both_clean():
    """No bar counts, but both flags set → 1.0 (flags alone)."""
    current = TrackStructure(intro_bars=None, outro_bars=None, has_clean_intro=False, has_clean_outro=True)
    cand = TrackStructure(intro_bars=None, outro_bars=None, has_clean_intro=True, has_clean_outro=False)
    score = transition_fit(current, cand)
    assert score == pytest.approx(1.0)


def test_flag_only_fallback_half_clean():
    """Only current's outro is clean, no bar counts → 0.5."""
    current = TrackStructure(intro_bars=None, outro_bars=None, has_clean_intro=False, has_clean_outro=True)
    cand = TrackStructure(intro_bars=None, outro_bars=None, has_clean_intro=False, has_clean_outro=False)
    score = transition_fit(current, cand)
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# transition_fit — result is always in [0, 1] for valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current_outro,cand_intro",
    [
        (0, 0),
        (1, 1),
        (16, 16),
        (32, 32),
        (64, 64),
        (0, 32),
        (32, 0),
    ],
)
def test_score_always_in_unit_interval(current_outro: int, cand_intro: int) -> None:
    current = _struct(outro=current_outro)
    cand = _struct(intro=cand_intro)
    score = transition_fit(current, cand)
    assert score is not None
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compose with scoring.combine
# ---------------------------------------------------------------------------


def test_transition_fit_composes_with_combine():
    """transition_fit output slots into scoring.combine without error."""
    cand = _struct(intro=32)
    current = _struct(outro=32)
    fit = transition_fit(current, cand)
    assert fit is not None

    prof = sc.Profile("test", {"embed": 0.5, "transition_fit": 0.5})
    score, breakdown = sc.combine({"embed": 0.8, "transition_fit": fit}, prof)
    assert 0.0 <= score <= 1.0
    assert "transition_fit" in breakdown


def test_transition_fit_none_dropped_by_combine():
    """None transition_fit is gracefully skipped by combine."""
    prof = sc.Profile("test", {"embed": 0.5, "transition_fit": 0.5})
    score, breakdown = sc.combine({"embed": 0.8, "transition_fit": None}, prof)
    # Only embed contributed — score should be 0.8 (renormalized over weight 0.5).
    assert score == pytest.approx(0.8)
    assert "transition_fit" not in breakdown


# ---------------------------------------------------------------------------
# load_structure — DB integration tests
# ---------------------------------------------------------------------------


def _add_region(
    session: Session,
    track_id: int,
    label: SectionLabel,
    length_bars: int | None = None,
    length_ms: int | None = None,
) -> Region:
    """Insert a full-track SECTION region and flush."""
    r = Region(
        track_id=track_id,
        stem_file_id=None,
        position_ms=0,
        length_ms=length_ms,
        region_type=RegionType.SECTION.value,
        section_label=label.value,
        length_bars=length_bars,
        source=RegionSource.AUTO.value,
        created_at=now_utc(),
    )
    session.add(r)
    session.flush()
    return r


def test_load_structure_no_regions(session, make_track):
    """Track with no regions → all None / False."""
    track = make_track()
    struct = load_structure(session, track.id)
    assert struct.intro_bars is None
    assert struct.outro_bars is None
    assert struct.has_clean_intro is False
    assert struct.has_clean_outro is False


def test_load_structure_reads_intro_and_outro(session, make_track):
    """Intro + outro regions with length_bars are read correctly."""
    track = make_track()
    _add_region(session, track.id, SectionLabel.INTRO, length_bars=32)
    _add_region(session, track.id, SectionLabel.OUTRO, length_bars=16)

    struct = load_structure(session, track.id)
    assert struct.intro_bars == 32
    assert struct.outro_bars == 16
    assert struct.has_clean_intro is True   # 32 >= MIN_MIX_BARS
    assert struct.has_clean_outro is True   # 16 == MIN_MIX_BARS


def test_load_structure_short_intro_not_clean(session, make_track):
    track = make_track()
    _add_region(session, track.id, SectionLabel.INTRO, length_bars=8)
    struct = load_structure(session, track.id)
    assert struct.intro_bars == 8
    assert struct.has_clean_intro is False


def test_load_structure_ignores_stem_regions(session, make_track):
    """Regions with a stem_file_id set must not be counted."""
    from dance.core.database import StemFile, StemKind

    track = make_track()
    # Create a fake stem file so the FK resolves.
    stem = StemFile(
        track_id=track.id,
        kind=StemKind.VOCALS.value,
        path="/tmp/vocals.wav",
        created_at=now_utc(),
    )
    session.add(stem)
    session.flush()

    # Stem-scoped intro — should be ignored by load_structure.
    r = Region(
        track_id=track.id,
        stem_file_id=stem.id,
        position_ms=0,
        length_ms=32000,
        region_type=RegionType.SECTION.value,
        section_label=SectionLabel.INTRO.value,
        length_bars=32,
        source=RegionSource.AUTO.value,
        created_at=now_utc(),
    )
    session.add(r)
    session.flush()

    struct = load_structure(session, track.id)
    assert struct.intro_bars is None
    assert struct.has_clean_intro is False


def test_load_structure_takes_longest_when_duplicates(session, make_track):
    """If multiple intro/outro regions exist, the longest bars wins."""
    track = make_track()
    _add_region(session, track.id, SectionLabel.INTRO, length_bars=8)
    _add_region(session, track.id, SectionLabel.INTRO, length_bars=32)
    _add_region(session, track.id, SectionLabel.INTRO, length_bars=16)

    struct = load_structure(session, track.id)
    assert struct.intro_bars == 32


def test_load_structure_null_length_bars_skipped(session, make_track):
    """A region whose length_bars is NULL contributes nothing to bar counts."""
    track = make_track()
    _add_region(session, track.id, SectionLabel.INTRO, length_bars=None, length_ms=32000)
    struct = load_structure(session, track.id)
    assert struct.intro_bars is None


def test_load_structure_feeds_transition_fit(session, make_track):
    """End-to-end: insert regions for two tracks, score the transition."""
    t_current = make_track()
    t_candidate = make_track()

    _add_region(session, t_current.id, SectionLabel.OUTRO, length_bars=32)
    _add_region(session, t_candidate.id, SectionLabel.INTRO, length_bars=32)

    current_s = load_structure(session, t_current.id)
    cand_s = load_structure(session, t_candidate.id)

    score = transition_fit(current_s, cand_s)
    assert score is not None
    assert score == pytest.approx(1.0)
