"""
Query the recommendation graph.

Given seed track IDs and a configurable mix of edge kinds, return ranked
candidate tracks. Each result carries a list of ``reasons`` (one per edge
that contributed) so callers can explain the suggestion in the UI.

Also exposes :meth:`Recommender.recommend_by_text` — CLAP is a joint
audio/text model so an arbitrary natural-language query ("punchy techy with
vocals") can rank tracks directly without going through tags.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import EdgeKind, Track, TrackEdge, TrackEmbedding
from dance.core.serialization import decode_embedding


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

    # ------------------------------------------------------------------

    def recommend_by_text(
        self,
        query: str,
        text_encoder: Callable[[str], np.ndarray],
        *,
        k: int = 10,
        model_name: str | None = None,
        exclude: list[int] | None = None,
    ) -> list[RecommendationResult]:
        """Rank tracks by CLAP cosine similarity to a text query.

        Args:
            query: Free-form text ("punchy techy with vocals").
            text_encoder: Callable that returns a 1-D numpy embedding for the
                query — typically ``EmbeddingStage.encode_text`` after the
                stage is loaded.
            k: Top-K results.
            model_name: Restrict to embeddings produced by this model
                (matches ``track_embeddings.model``). Defaults to
                full-mix embeddings produced by ANY model when None.
            exclude: Track IDs to omit.

        Returns:
            Ranked results with a single ``reasons`` entry of kind
            ``"text_query"``.
        """
        if not query.strip():
            return []

        query_vec = text_encoder(query).astype(np.float32, copy=False)
        query_norm = float(np.linalg.norm(query_vec)) or 1.0
        query_vec = query_vec / query_norm

        # Pull all full-mix embeddings (one per track).
        embed_q = (
            self.session.query(TrackEmbedding)
            .filter(TrackEmbedding.stem_file_id.is_(None))
        )
        if model_name:
            embed_q = embed_q.filter(TrackEmbedding.model == model_name)
        rows = embed_q.all()

        if not rows:
            return []

        excluded: set[int] = set(exclude or [])

        # Batch cosine: stack track embeddings and dot against the query.
        track_ids: list[int] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            if int(row.track_id) in excluded:
                continue
            v = decode_embedding(row.embedding, int(row.dim))
            norm = float(np.linalg.norm(v)) or 1.0
            vectors.append(v / norm)
            track_ids.append(int(row.track_id))

        if not vectors:
            return []

        matrix = np.stack(vectors, axis=0)  # (N, dim)
        # Both sides L2-normalized → dot = cosine in [-1, 1].
        cosines = matrix @ query_vec

        # Rank by cosine descending.
        order = np.argsort(-cosines)[:k]
        return [
            RecommendationResult(
                track_id=track_ids[i],
                score=float(cosines[i]),
                reasons=[
                    {"kind": "text_query", "query": query, "cosine": float(cosines[i])}
                ],
            )
            for i in order
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
