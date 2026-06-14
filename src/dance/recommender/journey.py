"""The journey: the time-shaped context every recommender scores against.

A set isn't a bag of similar tracks — it's a trajectory. :class:`JourneyState`
captures that trajectory across the axes defined in the proposal (vibe, key,
BPM, energy, repetition) at a point in time, and both surfaces build one:

* the **planner** (tail recs) builds it from the ordered set so far —
  :func:`journey_from_tracks`;
* the **Booth** builds it from the active stem combo plus the trailing tracks
  of the live set — :func:`journey_from_combo`.

The vibe axis is **trend-aware**: rather than rewarding mere proximity to the
recent average, we project the target *forward* along the direction the set has
been moving (``normalize(target + β·trend)``) and reward candidates near that
projection. That is what makes a recommendation feel like it continues the
journey instead of repeating it.

DB loading lives here (once) so the recommenders don't each re-implement the
combo/trailing joins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from dance.core.database import AudioAnalysis, TrackEmbedding
from dance.core.serialization import decode_embedding
from dance.recommender.scoring import embed_score, project_energy

# How far to push the vibe target along the established trend, in trend-vector
# units. 0 = pure proximity (no journey); larger = more aggressively "moving".
TREND_BETA = 0.5


@dataclass
class JourneyState:
    """The arc context at one moment. All fields optional — a fresh/empty set
    yields a mostly-``None`` state and scorers degrade gracefully."""

    trailing_track_ids: list[int] = field(default_factory=list)
    target_vibe_vec: np.ndarray | None = None
    vibe_trend_vec: np.ndarray | None = None
    target_keys: list[str] = field(default_factory=list)
    target_bpm: float | None = None
    projected_energy: float | None = None
    used_source_ids: set[int] = field(default_factory=set)
    used_keys: list[str] = field(default_factory=list)
    set_fraction: float | None = None
    # Booth-only (micro scale): the stems currently armed in Live.
    current_combo_stem_ids: list[int] = field(default_factory=list)

    def vibe_target(self, *, beta: float = TREND_BETA) -> np.ndarray | None:
        """The trend-projected vibe target. Falls back to the plain target when
        there is no trend (window too short / no movement)."""
        if self.target_vibe_vec is None:
            return None
        if self.vibe_trend_vec is None:
            return self.target_vibe_vec
        return _unit(self.target_vibe_vec + beta * self.vibe_trend_vec)

    def vibe_score(self, cand_vec: np.ndarray | None, *, beta: float = TREND_BETA) -> float | None:
        """Embedding fit against the trend-projected target."""
        return embed_score(cand_vec, self.vibe_target(beta=beta))


# ---------------------------------------------------------------------------
# Pure trajectory helpers
# ---------------------------------------------------------------------------


def recency_weighted_vibe(vecs_in_order: list[np.ndarray]) -> np.ndarray | None:
    """Recency-weighted mean of unit vectors (oldest→newest), ramp 1.0→2.0."""
    if not vecs_in_order:
        return None
    n = len(vecs_in_order)
    weights = np.array([1.0 + i / max(1, n - 1) for i in range(n)], dtype=np.float32)
    stacked = np.stack([_unit(v) for v in vecs_in_order], axis=0)
    avg = (stacked * weights.reshape(-1, 1)).sum(axis=0) / weights.sum()
    return _unit(avg)


def vibe_trend(vecs_in_order: list[np.ndarray]) -> np.ndarray | None:
    """Direction the vibe is heading: ``unit(mean(recent half) - mean(older half))``.

    Halving (rather than newest-minus-oldest) makes the trend robust to a single
    outlier track. ``None`` if fewer than two vectors.
    """
    if len(vecs_in_order) < 2:
        return None
    units = [_unit(v) for v in vecs_in_order]
    mid = len(units) // 2
    older = np.mean(np.stack(units[:mid] or units[:1], axis=0), axis=0)
    recent = np.mean(np.stack(units[mid:], axis=0), axis=0)
    diff = recent - older
    if float(np.linalg.norm(diff)) == 0.0:
        return None
    return _unit(diff)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def journey_from_tracks(
    session: Session,
    ordered_track_ids: list[int],
    *,
    window: int,
    set_fraction: float | None = None,
) -> JourneyState:
    """Build the macro journey from an ordered track sequence (the planner case).

    ``ordered_track_ids`` is the whole set so far (used for anti-repetition);
    only the trailing ``window`` shapes the vibe/key/BPM/energy targets.
    """
    if not ordered_track_ids:
        return JourneyState(set_fraction=set_fraction)
    trailing = ordered_track_ids[-window:]
    vecs = _ordered_fullmix_vecs(session, trailing)
    keys, bpms, energies = _ordered_signals(session, trailing)
    return JourneyState(
        trailing_track_ids=trailing,
        target_vibe_vec=recency_weighted_vibe(vecs),
        vibe_trend_vec=vibe_trend(vecs),
        target_keys=keys,
        target_bpm=float(np.median(bpms)) if bpms else None,
        projected_energy=project_energy(energies),
        used_source_ids=set(ordered_track_ids),
        used_keys=keys,
        set_fraction=set_fraction,
    )


def journey_from_combo(
    session: Session,
    combo_stem_ids: list[int],
    *,
    master_bpm: float | None = None,
    trailing_track_ids: list[int] | None = None,
    window: int = 5,
    set_fraction: float | None = None,
) -> JourneyState:
    """Build the Booth journey: vibe/key/BPM target from the active stem combo,
    trend + projected energy + anti-repetition from the trailing live set.
    """
    combo_vec = _combo_embedding(session, combo_stem_ids)
    combo_keys, combo_bpms = _combo_keys_and_bpms(session, combo_stem_ids)
    target_bpm = master_bpm
    if target_bpm is None and combo_bpms:
        target_bpm = sum(combo_bpms) / len(combo_bpms)

    trailing = (trailing_track_ids or [])[-window:]
    trend_vecs = _ordered_fullmix_vecs(session, trailing) if trailing else []
    _, _, energies = _ordered_signals(session, trailing) if trailing else ([], [], [])

    return JourneyState(
        trailing_track_ids=trailing,
        target_vibe_vec=combo_vec,
        vibe_trend_vec=vibe_trend(trend_vecs),
        target_keys=combo_keys,
        target_bpm=target_bpm,
        projected_energy=project_energy(energies),
        used_source_ids=set(trailing),
        used_keys=combo_keys,
        set_fraction=set_fraction,
        current_combo_stem_ids=list(combo_stem_ids),
    )


# ---------------------------------------------------------------------------
# DB loaders (shared — the recommenders import these instead of re-rolling them)
# ---------------------------------------------------------------------------


def _ordered_fullmix_vecs(session: Session, track_ids: list[int]) -> list[np.ndarray]:
    """Full-mix embeddings for ``track_ids``, returned in the given order."""
    if not track_ids:
        return []
    rows = (
        session.query(TrackEmbedding)
        .filter(
            TrackEmbedding.track_id.in_(track_ids),
            TrackEmbedding.stem_file_id.is_(None),
        )
        .all()
    )
    by_track = {int(r.track_id): decode_embedding(r.embedding, int(r.dim)) for r in rows}
    return [by_track[t] for t in track_ids if t in by_track]


def _ordered_signals(
    session: Session, track_ids: list[int]
) -> tuple[list[str], list[float], list[int]]:
    """Full-mix (key, bpm, floor_energy) for ``track_ids`` in order."""
    if not track_ids:
        return [], [], []
    rows = (
        session.query(AudioAnalysis)
        .filter(
            AudioAnalysis.track_id.in_(track_ids),
            AudioAnalysis.stem_file_id.is_(None),
        )
        .all()
    )
    by_track: dict[int, AudioAnalysis] = {int(r.track_id): r for r in rows}
    keys: list[str] = []
    bpms: list[float] = []
    energies: list[int] = []
    for tid in track_ids:
        r = by_track.get(tid)
        if r is None:
            continue
        if r.key_camelot:
            keys.append(str(r.key_camelot))
        if r.bpm is not None:
            bpms.append(float(r.bpm))
        if r.floor_energy is not None:
            energies.append(int(r.floor_energy))
    return keys, bpms, energies


def _combo_embedding(session: Session, combo_stem_ids: list[int]) -> np.ndarray | None:
    """Averaged + normalized embedding of the active combo stems."""
    if not combo_stem_ids:
        return None
    rows = (
        session.query(TrackEmbedding)
        .filter(TrackEmbedding.stem_file_id.in_(combo_stem_ids))
        .all()
    )
    if not rows:
        return None
    vecs = [_unit(decode_embedding(r.embedding, int(r.dim))) for r in rows]
    return _unit(np.mean(np.stack(vecs, axis=0), axis=0))


def _combo_keys_and_bpms(
    session: Session, combo_stem_ids: list[int]
) -> tuple[list[str], list[float]]:
    if not combo_stem_ids:
        return [], []
    rows = (
        session.query(AudioAnalysis)
        .filter(AudioAnalysis.stem_file_id.in_(combo_stem_ids))
        .all()
    )
    keys: list[str] = []
    bpms: list[float] = []
    for r in rows:
        # Stem analyses store dominant_pitch_camelot, not key_camelot.
        pitch = getattr(r, "dominant_pitch_camelot", None) or getattr(r, "key_camelot", None)
        if pitch:
            keys.append(str(pitch))
        if r.bpm is not None:
            bpms.append(float(r.bpm))
    return keys, bpms


def _unit(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(v)) or 1.0
    return v / norm


__all__ = [
    "TREND_BETA",
    "JourneyState",
    "journey_from_combo",
    "journey_from_tracks",
    "recency_weighted_vibe",
    "vibe_trend",
]
