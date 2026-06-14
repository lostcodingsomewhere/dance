"""Shared scoring core for every recommender.

One implementation of each compatibility signal, one place that combines them.
The Booth (per-column), the planner (tail), and the seed graph all import from
here so they can never drift apart again (they used to: the graph handled
half/double-time BPM matches the live scorers silently missed).

Two layers:

* **Feature scorers** — pure functions, each maps one signal to ``[0, 1]`` and
  is ``None``-safe (missing data → ``None``, never a crash). Scale-free ratio
  scorers are used where the underlying unit is unknown/arbitrary.
* **Profiles + :func:`combine`** — a profile is a weight vector over the
  feature set. ``combine`` does the weighted blend, renormalizing over only the
  features that were actually computable, so a missing signal down-weights
  gracefully instead of tanking the score.

Trend-aware vibe scoring lives in :mod:`dance.recommender.journey` because it
needs the journey's direction vector; everything context-free lives here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from dance.pipeline.utils.camelot import get_compatible_keys

# Default BPM tolerance (half-width of the linear falloff window). Tighter than
# the old ±20 — for electronic, 124 vs 140 is not a match.
DEFAULT_BPM_TOL = 8.0
# Half/double-time matches are real but slightly less clean than a direct match.
_HALFTIME_PENALTY = 0.9


# ---------------------------------------------------------------------------
# Feature scorers — each → [0, 1], None-safe
# ---------------------------------------------------------------------------


def embed_score(cand_vec: np.ndarray | None, target_vec: np.ndarray | None) -> float | None:
    """Cosine similarity of two embeddings, mapped from ``[-1, 1]`` to ``[0, 1]``.

    Both vectors are L2-normalized defensively. Returns ``None`` if either side
    is absent so :func:`combine` can drop the feature.
    """
    if cand_vec is None or target_vec is None:
        return None
    a = _unit(cand_vec)
    b = _unit(target_vec)
    cos = float(np.dot(a, b))
    return max(0.0, (cos + 1.0) / 2.0)


def key_score(cand_key: str | None, target_keys: list[str]) -> float | None:
    """``1.0`` exact Camelot match, ``0.6`` harmonic-compatible, ``0.0`` else.

    ``target_keys`` is the bag of keys in the current context (one per active
    stem, or the recent set keys). ``None`` when we have nothing to compare.
    """
    if not cand_key or not target_keys:
        return None
    cand = cand_key.upper()
    exact = {k.upper() for k in target_keys if k}
    if cand in exact:
        return 1.0
    compat: set[str] = set()
    for t in target_keys:
        if t:
            compat.update(k.upper() for k in get_compatible_keys(t))
    return 0.6 if cand in compat else 0.0


def bpm_score(
    cand_bpm: float | None,
    target_bpm: float | None,
    *,
    tol: float = DEFAULT_BPM_TOL,
    halftime: bool = True,
) -> float | None:
    """Linear falloff to 0 at ``±tol``, with optional half/double-time folding.

    A 70 BPM candidate against a 140 target scores via the doubled match
    (×:py:data:`_HALFTIME_PENALTY`). This is the single BPM scorer — the Booth
    used to lack the fold and miss these.
    """
    if cand_bpm is None or target_bpm is None:
        return None
    cand = float(cand_bpm)
    target = float(target_bpm)
    options: list[tuple[float, float]] = [(cand, 1.0)]
    if halftime:
        options += [(cand * 2.0, _HALFTIME_PENALTY), (cand / 2.0, _HALFTIME_PENALTY)]
    best = 0.0
    for value, penalty in options:
        s = max(0.0, 1.0 - abs(value - target) / tol) * penalty
        best = max(best, s)
    return best


def energy_score(cand_energy: int | None, target_energy: float | None) -> float | None:
    """Falloff around the *projected* arc energy: ``1.0`` at projection,
    ``0.0`` at ±3 floors. Symmetric — over and under are equally off-arc."""
    if cand_energy is None or target_energy is None:
        return None
    delta = abs(float(cand_energy) - float(target_energy))
    return max(0.0, 1.0 - delta / 3.0)


def ratio_score(a: float | None, b: float | None) -> float | None:
    """Scale-free similarity of two positive quantities: ``min/max`` → ``[0, 1]``.

    Used where the unit is arbitrary (e.g. ``kick_density``). ``1.0`` when
    equal, → ``0`` as they diverge. Non-positive or missing inputs → ``None``.
    """
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    if a <= 0.0 and b <= 0.0:
        return 1.0
    hi = max(a, b)
    if hi <= 0.0:
        return None
    return max(0.0, min(a, b) / hi)


def delta_score(a: float | None, b: float | None, *, scale: float = 1.0) -> float | None:
    """Linear similarity for values already on a known scale (e.g. ``presence_ratio``
    in ``[0, 1]``): ``1 - |a-b|/scale`` clamped to ``[0, 1]``."""
    if a is None or b is None or scale <= 0.0:
        return None
    return max(0.0, 1.0 - abs(float(a) - float(b)) / scale)


def kick_density_score(cand: float | None, target: float | None) -> float | None:
    """Similarity of two drum stems' kick densities (scale-free ratio)."""
    return ratio_score(cand, target)


def presence_score(cand: float | None, target: float | None) -> float | None:
    """Similarity of two stems' presence ratios (both already in ``[0, 1]``)."""
    return delta_score(cand, target, scale=1.0)


def timbre_score(
    cand: tuple[float | None, float | None],
    target: tuple[float | None, float | None],
) -> float | None:
    """Timbral similarity from ``(brightness, warmth)`` pairs (scale-free).

    Averages whichever components are present on both sides; ``None`` if neither
    component is comparable.
    """
    parts = [
        ratio_score(cand[0], target[0]),
        ratio_score(cand[1], target[1]),
    ]
    present = [p for p in parts if p is not None]
    return sum(present) / len(present) if present else None


def confidence_gate(raw: float | None, confidence: float | None) -> float | None:
    """Discount a raw score by the analysis confidence that produced its inputs.

    ``confidence is None`` → pass through unchanged (neutral). Otherwise multiply
    by the clamped confidence so shaky key/BPM detections contribute less.
    """
    if raw is None:
        return None
    if confidence is None:
        return raw
    return raw * max(0.0, min(1.0, float(confidence)))


def project_energy(trailing_energies: list[int]) -> float | None:
    """Linear extrapolation of the energy arc: fit a line over the trailing
    floor-energies and return the next value, clamped to ``[1, 10]``.

    Falls back to the single value when given one point, ``None`` when empty.
    (Moved here from ``tail.py`` so the Booth can project too.)
    """
    pts = [float(e) for e in trailing_energies if e is not None]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    xs = np.arange(len(pts), dtype=np.float32)
    ys = np.array(pts, dtype=np.float32)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    projected = float(slope * len(pts) + intercept)
    return max(1.0, min(10.0, projected))


# ---------------------------------------------------------------------------
# Profiles + combine
# ---------------------------------------------------------------------------

# Canonical feature names — the keys of a feature dict and of profile weights.
FEATURES = (
    "embedding",
    "key",
    "bpm",
    "energy",
    "kick_density",
    "presence",
    "timbre",
    "transition_fit",
)


@dataclass(frozen=True)
class Profile:
    """A named weight vector over :data:`FEATURES`. Weights need not sum to 1 —
    :func:`combine` renormalizes over the features actually present."""

    name: str
    weights: Mapping[str, float]


# Per-context profiles. Key is dropped for drums/other (pitch on a kit is noise);
# kick_density carries the drums column instead. See the proposal's §3.2 table.
PROFILES: dict[str, Profile] = {
    "column:drums": Profile("column:drums", {"embedding": 0.45, "bpm": 0.30, "kick_density": 0.25}),
    "column:bass": Profile(
        "column:bass", {"embedding": 0.40, "key": 0.25, "bpm": 0.25, "presence": 0.10}
    ),
    "column:vocals": Profile(
        "column:vocals", {"embedding": 0.45, "key": 0.35, "bpm": 0.10, "timbre": 0.10}
    ),
    "column:other": Profile(
        "column:other", {"embedding": 0.45, "key": 0.20, "bpm": 0.10, "timbre": 0.25}
    ),
    "column:mix": Profile(
        "column:mix", {"embedding": 0.40, "key": 0.25, "bpm": 0.20, "timbre": 0.15}
    ),
    "tail": Profile(
        "tail",
        {
            "embedding": 0.35,
            "key": 0.20,
            "bpm": 0.15,
            "energy": 0.15,
            "transition_fit": 0.10,
            "presence": 0.05,
        },
    ),
}


def profile_for_column(column: str) -> Profile:
    """Resolve the per-column Booth profile, defaulting to ``column:mix``."""
    return PROFILES.get(f"column:{column}", PROFILES["column:mix"])


def combine(features: Mapping[str, float | None], profile: Profile) -> tuple[float, dict[str, float]]:
    """Weighted blend of ``features`` under ``profile``.

    Only features that are (a) weighted by the profile and (b) present (not
    ``None``) contribute; the score is renormalized over their weights so a
    missing signal neither helps nor unfairly hurts. Returns
    ``(score, breakdown)`` where breakdown holds the raw per-feature values that
    contributed (for UI explanations).
    """
    total = 0.0
    active_weight = 0.0
    breakdown: dict[str, float] = {}
    for name, weight in profile.weights.items():
        if weight <= 0.0:
            continue
        value = features.get(name)
        if value is None:
            continue
        total += float(value) * float(weight)
        active_weight += float(weight)
        breakdown[name] = float(value)
    score = total / active_weight if active_weight > 0.0 else 0.0
    return score, breakdown


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(v)) or 1.0
    return v / norm


__all__ = [
    "DEFAULT_BPM_TOL",
    "FEATURES",
    "PROFILES",
    "Profile",
    "bpm_score",
    "combine",
    "confidence_gate",
    "delta_score",
    "embed_score",
    "energy_score",
    "key_score",
    "kick_density_score",
    "presence_score",
    "profile_for_column",
    "project_energy",
    "ratio_score",
    "timbre_score",
]
