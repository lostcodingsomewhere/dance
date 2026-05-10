"""
Query the recommendation graph.

Given seed track IDs and a configurable mix of edge kinds, return ranked
candidate tracks. Each result carries a list of ``reasons`` (one per edge
that contributed) so callers can explain the suggestion in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from dance.core.database import EdgeKind, TrackEdge


DEFAULT_WEIGHTS: dict[EdgeKind, float] = {
    EdgeKind.HARMONIC_COMPAT: 1.0,
    EdgeKind.TEMPO_COMPAT: 1.0,
    EdgeKind.EMBEDDING_NEIGHBOR: 1.0,
    EdgeKind.TAG_OVERLAP: 1.0,
    EdgeKind.MANUALLY_PAIRED: 1.0,
    EdgeKind.PLAYLIST_NEIGHBOR: 1.0,
}


@dataclass
class RecommendationResult:
    track_id: int
    score: float
    reasons: list[dict] = field(default_factory=list)


class Recommender:
    """Query the ``track_edges`` graph to produce ranked suggestions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def recommend(
        self,
        seeds: list[int],
        k: int = 10,
        kinds: list[EdgeKind] | None = None,
        weights: dict[EdgeKind, float] | None = None,
        exclude: list[int] | None = None,
    ) -> list[RecommendationResult]:
        if not seeds:
            return []

        active_kinds: list[EdgeKind] = (
            list(kinds) if kinds is not None else list(DEFAULT_WEIGHTS)
        )
        kind_values = [k_.value for k_ in active_kinds]
        weight_map: dict[str, float] = {
            k_.value: (weights or {}).get(k_, DEFAULT_WEIGHTS.get(k_, 1.0))
            for k_ in active_kinds
        }

        excluded: set[int] = set(exclude or [])
        seed_set: set[int] = set(seeds)

        # Aggregate per candidate target.
        scores: dict[int, float] = {}
        reasons: dict[int, list[dict]] = {}

        edges = (
            self.session.query(TrackEdge)
            .filter(
                TrackEdge.from_track_id.in_(list(seed_set)),
                TrackEdge.kind.in_(kind_values),
            )
            .all()
        )

        for edge in edges:
            target = int(edge.to_track_id)
            if target in seed_set or target in excluded:
                continue
            kind_weight = weight_map.get(edge.kind, 0.0)
            if kind_weight == 0.0:
                continue
            contribution = float(edge.weight) * kind_weight
            scores[target] = scores.get(target, 0.0) + contribution
            reasons.setdefault(target, []).append(
                {
                    "kind": edge.kind,
                    "from_seed": int(edge.from_track_id),
                    "weight": float(edge.weight),
                }
            )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            RecommendationResult(track_id=tid, score=score, reasons=reasons[tid])
            for tid, score in ranked
        ]


def recommend(
    session: Session,
    seeds: list[int],
    **kwargs,
) -> list[RecommendationResult]:
    """Convenience wrapper: instantiate ``Recommender`` and delegate."""
    return Recommender(session).recommend(seeds=seeds, **kwargs)


__all__ = [
    "DEFAULT_WEIGHTS",
    "RecommendationResult",
    "Recommender",
    "recommend",
]
