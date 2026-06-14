"""Tail-recommendations for a curated Set.

"What should come next?" given the ordered sequence of tracks the DJ has
already planned. The intuition is **arc-fit**, not just "similar to last
track":

* trend-aware vibe — :meth:`~dance.recommender.journey.JourneyState.vibe_score`
  projects the trajectory forward so the next track feels like *continuation*,
  not repetition
* key trajectory — same Camelot key, then harmonic-compatible, falls off
* BPM band — close to the recent median (±8 BPM tol from the shared core)
* energy slope — projects where the arc is *heading* and prefers candidates
  that land near that projection

All scoring delegates to :mod:`dance.recommender.scoring` and
:mod:`dance.recommender.journey` — no duplicate helpers here.
Stateless public entry: :func:`tail_recs_for_set`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from dance.core.database import (
    AudioAnalysis,
    Set,
    SetTrack,
    Track,
    TrackEmbedding,
)
from dance.core.serialization import decode_embedding
from dance.recommender.journey import JourneyState, journey_from_tracks
from dance.recommender.scoring import (
    PROFILES,
    bpm_score,
    combine,
    energy_score,
    key_score,
    presence_score,
)
from dance.recommender.structure import TrackStructure, load_structures, transition_fit

# Trailing-window depth — how many recent tracks shape the "current arc".
# 5 captures the last ~25 min of a set without letting an opener dominate.
DEFAULT_WINDOW = 5


@dataclass
class TailRecResult:
    """One tail-rec candidate for a Set."""

    track_id: int
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal candidate container
# ---------------------------------------------------------------------------


@dataclass
class _Cand:
    track_id: int
    embedding: np.ndarray | None
    bpm: float | None
    key: str | None
    energy: int | None
    presence_ratio: float | None


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _candidates(session: Session, excluded: set[int]) -> list[_Cand]:
    """Load every track with at least a BPM or embedding, minus *excluded*."""
    rows = (
        session.query(
            Track.id,
            TrackEmbedding.embedding,
            TrackEmbedding.dim,
            AudioAnalysis.bpm,
            AudioAnalysis.key_camelot,
            AudioAnalysis.floor_energy,
            AudioAnalysis.presence_ratio,
        )
        .outerjoin(
            TrackEmbedding,
            (TrackEmbedding.track_id == Track.id) & (TrackEmbedding.stem_file_id.is_(None)),
        )
        .outerjoin(
            AudioAnalysis,
            (AudioAnalysis.track_id == Track.id) & (AudioAnalysis.stem_file_id.is_(None)),
        )
        .all()
    )
    out: list[_Cand] = []
    for track_id, embed_blob, dim, bpm, key, energy, presence_ratio in rows:
        tid = int(track_id)
        if tid in excluded:
            continue
        vec: np.ndarray | None = None
        if embed_blob and dim:
            raw = decode_embedding(embed_blob, int(dim)).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(raw)) or 1.0
            vec = raw / norm
        out.append(
            _Cand(
                track_id=tid,
                embedding=vec,
                bpm=float(bpm) if bpm is not None else None,
                key=str(key) if key else None,
                energy=int(energy) if energy is not None else None,
                presence_ratio=float(presence_ratio) if presence_ratio is not None else None,
            )
        )
    return out


def _score_candidate(
    cand: _Cand,
    journey: JourneyState,
    cand_structure: TrackStructure | None,
    current_structure: TrackStructure | None,
) -> TailRecResult:
    """Score one candidate against *journey*, returning a :class:`TailRecResult`."""
    # Build a target presence from the journey's trailing window average if needed;
    # since full-mix presence_ratio is rarely populated, we pass None → combine
    # drops the feature gracefully.
    target_presence: float | None = None

    features: dict[str, float | None] = {
        "embedding": journey.vibe_score(cand.embedding),
        "key": key_score(cand.key, journey.target_keys),
        "bpm": bpm_score(cand.bpm, journey.target_bpm),
        "energy": energy_score(cand.energy, journey.projected_energy),
        "presence": presence_score(cand.presence_ratio, target_presence),
        # Structural mixability: ride the last set track's outro under this
        # candidate's intro (phrase-aligned transition).
        "transition_fit": transition_fit(current_structure, cand_structure),
    }

    score, breakdown = combine(features, PROFILES["tail"])

    # Human reasons — thresholds chosen to match historical tail.py feel.
    reasons: list[str] = []
    embed_s = breakdown.get("embedding", 0.0)
    key_s = breakdown.get("key", 0.0)
    bpm_s = breakdown.get("bpm", 0.0)
    energy_s = breakdown.get("energy", 0.0)

    if embed_s >= 0.8:
        reasons.append("vibe match")
    if key_s >= 1.0:
        reasons.append("same key")
    elif key_s >= 0.6:
        reasons.append("harmonic-compat")
    if journey.target_bpm is not None:
        if bpm_s >= 0.95:
            reasons.append(f"~{journey.target_bpm:.0f} BPM")
        elif bpm_s >= 0.7:
            reasons.append(f"close to {journey.target_bpm:.0f} BPM")
    if journey.projected_energy is not None and cand.energy is not None and energy_s >= 0.7:
        reasons.append(f"arc → energy {journey.projected_energy:.0f}")

    return TailRecResult(
        track_id=cand.track_id,
        score=score,
        score_breakdown=breakdown,
        reasons=reasons,
    )


def recs_for_journey(
    session: Session,
    journey: JourneyState,
    *,
    k: int,
    exclude_track_ids: set[int],
    current_track_id: int | None,
    empty: bool = False,
) -> list[TailRecResult]:
    """Score every candidate track against a prebuilt *journey*.

    The shared engine behind :func:`tail_recs_for_set` (and reusable by any
    caller that builds its own ``JourneyState``). ``current_track_id`` is the
    outgoing track whose outro we mix out of (for ``transition_fit``);
    ``empty`` forces all-zero scores when there is no context yet.
    """
    cands = _candidates(session, exclude_track_ids)
    if not cands:
        return []

    structures = load_structures(session, [c.track_id for c in cands])
    current_structure = (
        load_structures(session, [current_track_id]).get(current_track_id)
        if current_track_id is not None
        else None
    )

    results: list[TailRecResult] = []
    for cand in cands:
        if empty:
            results.append(TailRecResult(track_id=cand.track_id, score=0.0))
        else:
            results.append(
                _score_candidate(
                    cand, journey, structures.get(cand.track_id), current_structure
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


def tail_recs_for_set(
    session: Session,
    set_id: int,
    *,
    k: int = 10,
    window: int = DEFAULT_WINDOW,
    exclude_track_ids: list[int] | None = None,
) -> list[TailRecResult]:
    """Top-K candidates to append to a set, scored by arc-fit.

    Returns at most ``k`` :class:`TailRecResult`. Stateless: loads once per
    call. If the set does not exist or no candidates qualify, returns ``[]``.
    When the set is empty the journey is empty and all scores are ``0.0`` —
    the rail can show "add tracks first" rather than render this list, but we
    still return candidates so the endpoint doesn't 404.
    """
    s = session.get(Set, set_id)
    if s is None:
        return []

    set_rows = (
        session.query(SetTrack)
        .filter(SetTrack.set_id == set_id)
        .order_by(SetTrack.position)
        .all()
    )
    set_track_ids = [int(r.track_id) for r in set_rows]

    journey = journey_from_tracks(session, set_track_ids, window=max(1, window))
    return recs_for_journey(
        session,
        journey,
        k=k,
        exclude_track_ids=set(set_track_ids) | set(exclude_track_ids or []),
        current_track_id=set_track_ids[-1] if set_track_ids else None,
        empty=not set_track_ids,
    )


__all__ = [
    "DEFAULT_WINDOW",
    "TailRecResult",
    "recs_for_journey",
    "tail_recs_for_set",
]
