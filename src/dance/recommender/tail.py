"""Tail-recommendations for a curated Set.

"What should come next?" given the ordered sequence of tracks the DJ has
already planned. The intuition is **arc-fit**, not just "similar to last
track":

* trailing-window embedding aggregate — sonically near the recent run
* key trajectory — same key, then Camelot-compat, falls off
* BPM band — close to the recent median, falls off at ±20 BPM
* energy slope — projects where the arc is *heading* and prefers candidates
  that land near that projection (not necessarily the last value)

Designed to mirror :mod:`dance.recommender.recommender` patterns so the same
``Track`` / ``TrackEmbedding`` / ``AudioAnalysis`` joins are reused. Stateless
public entry: :func:`tail_recs_for_set`.
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
from dance.pipeline.utils.camelot import get_compatible_keys

# Trailing-window depth — how many recent tracks shape the "current arc".
# 5 captures the last ~25 min of a set without letting an opener dominate.
DEFAULT_WINDOW = 5

# Scoring weights — sum to 1.0 so a perfect candidate scores ~1.0.
_W_EMBED = 0.40
_W_KEY = 0.25
_W_BPM = 0.20
_W_ENERGY = 0.15


@dataclass
class TailRecResult:
    """One tail-rec candidate for a Set."""

    track_id: int
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm


def _bpm_score(candidate_bpm: float | None, target_bpm: float | None) -> float:
    """Linear falloff: 1.0 at exact match, 0.0 at ±20 BPM. Matches the
    per-column rec scorer so the rail and the column banners feel consistent.
    """
    if candidate_bpm is None or target_bpm is None:
        return 0.0
    delta = abs(float(candidate_bpm) - float(target_bpm))
    return max(0.0, 1.0 - delta / 20.0)


def _key_score(candidate_key: str | None, target_keys: list[str]) -> float:
    """1.0 if candidate matches any target key exactly, 0.6 if Camelot-compat
    (±1, relative major/minor), 0.0 otherwise. Same scale as column recs."""
    if not candidate_key or not target_keys:
        return 0.0
    candidate = candidate_key.upper()
    if candidate in {k.upper() for k in target_keys if k}:
        return 1.0
    compat: set[str] = set()
    for t in target_keys:
        if t:
            compat.update(k.upper() for k in get_compatible_keys(t))
    return 0.6 if candidate in compat else 0.0


def _energy_score(candidate_energy: int | None, target_energy: float | None) -> float:
    """Falloff around the *projected* arc energy.

    ``floor_energy`` is a 1–10 integer (see Krukowski's "energy floors"). 1.0 at
    exact projection, 0.0 at ±3 floors. We don't penalize "too low" any
    differently from "too high" — the DJ may want either side, the arc is the
    intent."""
    if candidate_energy is None or target_energy is None:
        return 0.5  # neutral when we can't compare
    delta = abs(float(candidate_energy) - float(target_energy))
    return max(0.0, 1.0 - delta / 3.0)


def _project_energy(trailing_energies: list[int]) -> float | None:
    """Linear extrapolation: fit a line over the trailing energies, return
    the next value. Falls back to last value if we only have one point, None
    if we have none."""
    pts = [float(e) for e in trailing_energies if e is not None]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    xs = np.arange(len(pts), dtype=np.float32)
    ys = np.array(pts, dtype=np.float32)
    # Numpy 1.x supports `polyfit`, 2.x recommends the Polynomial API but
    # keeps polyfit for compat. Linear fit is fine and avoids an import.
    slope, intercept = np.polyfit(xs, ys, deg=1)
    projected = float(slope * len(pts) + intercept)
    # Clamp to the floor_energy range so projections stay meaningful.
    return max(1.0, min(10.0, projected))


class _TailRecsImpl:
    """Stateful helper — loads the data once per request, then scores."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def run(
        self,
        set_id: int,
        *,
        k: int,
        window: int,
        exclude_track_ids: list[int],
    ) -> list[TailRecResult]:
        s = self.session.get(Set, set_id)
        if s is None:
            return []

        set_rows = (
            self.session.query(SetTrack)
            .filter(SetTrack.set_id == set_id)
            .order_by(SetTrack.position)
            .all()
        )
        set_track_ids = [int(r.track_id) for r in set_rows]
        trailing = set_track_ids[-window:]

        # Everything in the set is excluded (don't re-recommend what's
        # already planned); plus whatever the caller passed in.
        excluded: set[int] = set(set_track_ids) | set(exclude_track_ids)

        # Aggregate signals over the trailing window.
        combo_vec = self._trailing_embedding(trailing)
        keys, bpms, energies = self._trailing_signals(trailing)

        target_bpm = float(np.median(bpms)) if bpms else None
        target_energy = _project_energy(energies)

        # Candidate pool: every track with at least bpm or embedding;
        # excludes the set itself.
        candidates = self._candidates(excluded)
        if not candidates:
            return []

        results: list[TailRecResult] = []
        for cand in candidates:
            breakdown: dict[str, float] = {}
            reasons: list[str] = []

            embed_s = 0.0
            if combo_vec is not None and cand.embedding is not None:
                cos = float(np.dot(cand.embedding, combo_vec))
                # Map cos [-1, 1] → [0, 1] so it composes with other [0, 1] scores.
                embed_s = max(0.0, (cos + 1.0) / 2.0)
            breakdown["embedding"] = embed_s
            if embed_s >= 0.8:
                reasons.append("vibe match")

            key_s = _key_score(cand.key, keys)
            breakdown["key"] = key_s
            if key_s >= 1.0:
                reasons.append("same key")
            elif key_s >= 0.6:
                reasons.append("harmonic-compat")

            bpm_s = _bpm_score(cand.bpm, target_bpm)
            breakdown["bpm"] = bpm_s
            if bpm_s >= 0.95 and target_bpm is not None:
                reasons.append(f"~{target_bpm:.0f} BPM")
            elif bpm_s >= 0.7 and target_bpm is not None:
                reasons.append(f"close to {target_bpm:.0f} BPM")

            energy_s = _energy_score(cand.energy, target_energy)
            breakdown["energy"] = energy_s
            if target_energy is not None and cand.energy is not None and energy_s >= 0.7:
                reasons.append(f"arc → energy {target_energy:.0f}")

            # If we have no trailing window (empty set), all scores are
            # zero/neutral; rank by embedding-only-ish (still ~0) and BPM
            # falls to 0. Return candidates ordered arbitrarily — the rail
            # should show "set is empty, add tracks first" rather than render
            # this list, but we still return something so the endpoint
            # doesn't 404.
            if combo_vec is None and not keys and not bpms:
                score = 0.0
            else:
                score = embed_s * _W_EMBED + key_s * _W_KEY + bpm_s * _W_BPM + energy_s * _W_ENERGY

            results.append(
                TailRecResult(
                    track_id=cand.track_id,
                    score=score,
                    score_breakdown=breakdown,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _trailing_embedding(self, track_ids: list[int]) -> np.ndarray | None:
        if not track_ids:
            return None
        rows = (
            self.session.query(TrackEmbedding)
            .filter(
                TrackEmbedding.track_id.in_(track_ids),
                TrackEmbedding.stem_file_id.is_(None),
            )
            .all()
        )
        if not rows:
            return None
        # Weight recent tracks more — linear ramp from 1.0 (oldest in window)
        # to 2.0 (newest). Crude but matches "what's playing now matters most".
        weights_by_id: dict[int, float] = {}
        n = len(track_ids)
        for i, tid in enumerate(track_ids):
            weights_by_id[tid] = 1.0 + i / max(1, n - 1)

        vecs: list[np.ndarray] = []
        weights: list[float] = []
        for row in rows:
            v = decode_embedding(row.embedding, int(row.dim))
            vecs.append(_normalize(v.astype(np.float32, copy=False)))
            weights.append(weights_by_id.get(int(row.track_id), 1.0))
        if not vecs:
            return None
        w = np.array(weights, dtype=np.float32).reshape(-1, 1)
        stacked = np.stack(vecs, axis=0)
        weighted_avg = (stacked * w).sum(axis=0) / w.sum()
        return _normalize(weighted_avg)

    def _trailing_signals(self, track_ids: list[int]) -> tuple[list[str], list[float], list[int]]:
        if not track_ids:
            return [], [], []
        rows = (
            self.session.query(AudioAnalysis)
            .filter(
                AudioAnalysis.track_id.in_(track_ids),
                AudioAnalysis.stem_file_id.is_(None),
            )
            .all()
        )
        # Preserve set order — important for energy slope.
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

    def _candidates(self, excluded: set[int]) -> list[_Cand]:
        rows = (
            self.session.query(
                Track.id,
                TrackEmbedding.embedding,
                TrackEmbedding.dim,
                AudioAnalysis.bpm,
                AudioAnalysis.key_camelot,
                AudioAnalysis.floor_energy,
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
        for track_id, embed_blob, dim, bpm, key, energy in rows:
            tid = int(track_id)
            if tid in excluded:
                continue
            vec: np.ndarray | None = None
            if embed_blob and dim:
                vec = _normalize(
                    decode_embedding(embed_blob, int(dim)).astype(np.float32, copy=False)
                )
            out.append(
                _Cand(
                    track_id=tid,
                    embedding=vec,
                    bpm=float(bpm) if bpm is not None else None,
                    key=str(key) if key else None,
                    energy=int(energy) if energy is not None else None,
                )
            )
        return out


@dataclass
class _Cand:
    track_id: int
    embedding: np.ndarray | None
    bpm: float | None
    key: str | None
    energy: int | None


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
    call. If the set is empty or no candidates qualify, returns ``[]``.
    """
    impl = _TailRecsImpl(session)
    return impl.run(
        set_id=set_id,
        k=k,
        window=max(1, window),
        exclude_track_ids=list(exclude_track_ids or []),
    )


__all__ = [
    "DEFAULT_WINDOW",
    "TailRecResult",
    "tail_recs_for_set",
]
