"""
Recommendation graph builder.

Materializes only the embedding kNN edges (``EMBEDDING_NEIGHBOR``) into
``track_edges``.  Harmonic, tempo, and tag-overlap compatibility is now
scored live via the shared core in :mod:`dance.recommender.scoring` at
query time, so there is nothing to materialize for those signals.

Operates on the whole library at once, or on a ``track_ids`` subset for
incremental rebuilds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

import numpy as np
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    EdgeKind,
    TrackEdge,
    TrackEmbedding,
    now_utc,
)
from dance.core.serialization import decode_embedding

logger = logging.getLogger(__name__)

ALL_KINDS: tuple[EdgeKind, ...] = (EdgeKind.EMBEDDING_NEIGHBOR,)


class GraphBuilder:
    """Populate ``track_edges`` with embedding-neighbor edges."""

    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    def build(
        self,
        track_ids: list[int] | None = None,
        kinds: list[EdgeKind] | None = None,
    ) -> dict[str, int]:
        """Build (or rebuild) embedding-neighbor edges.

        ``track_ids=None`` rebuilds globally; otherwise only edges touching
        the listed tracks are deleted-and-rewritten.  ``kinds`` is accepted
        for API compatibility but only :data:`EdgeKind.EMBEDDING_NEIGHBOR`
        is supported — any other kind is logged and skipped.
        """
        requested = list(kinds) if kinds is not None else list(ALL_KINDS)
        results: dict[str, int] = {}
        for kind in requested:
            if kind != EdgeKind.EMBEDDING_NEIGHBOR:
                logger.warning("Unsupported edge kind (no longer built): %s", kind)
                continue
            self._clear_existing(kind, track_ids)
            edges = self._build_embedding(track_ids)
            count = self._upsert_edges(edges)
            results[kind.value] = count
            self.session.commit()
            logger.info("Built %d %s edges", count, kind.value)
        return results

    # ------------------------------------------------------------------
    # Edge-kind builder
    # ------------------------------------------------------------------

    def _build_embedding(self, track_ids: list[int] | None) -> list[TrackEdge]:
        model = self.settings.clap_model
        rows = (
            self.session.query(TrackEmbedding)
            .filter(
                TrackEmbedding.stem_file_id.is_(None),
                TrackEmbedding.model == model,
            )
            .all()
        )
        if len(rows) < 2:
            return []

        ids: list[int] = []
        vectors: list[np.ndarray] = []
        for r in rows:
            try:
                vec = decode_embedding(r.embedding, r.dim)
            except ValueError as exc:
                logger.warning("Skipping bad embedding for track %s: %s", r.track_id, exc)
                continue
            ids.append(int(r.track_id))
            vectors.append(vec.astype(np.float32, copy=False))
        if len(ids) < 2:
            return []

        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1e-12
        sim = matrix @ matrix.T / np.outer(norms, norms)
        np.fill_diagonal(sim, -np.inf)

        k = max(1, int(self.settings.recommender_top_k))
        k = min(k, len(ids) - 1)
        now = now_utc()
        touched = set(track_ids) if track_ids else None

        # For each A, take A's top-K. Materialize both (A,B) and (B,A) with
        # the same cosine so the recommender's outbound lookup just works.
        emitted: dict[tuple[int, int], float] = {}
        for i, a_id in enumerate(ids):
            row = sim[i]
            top_idx = np.argpartition(-row, k - 1)[:k]
            for j in top_idx:
                if i == j:
                    continue
                b_id = ids[int(j)]
                cosine = float(row[int(j)])
                emitted[(a_id, b_id)] = cosine
                emitted.setdefault((b_id, a_id), cosine)

        edges: list[TrackEdge] = []
        for (a_id, b_id), cosine in emitted.items():
            if touched is not None and a_id not in touched and b_id not in touched:
                continue
            weight = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
            edges.append(
                TrackEdge(
                    from_track_id=a_id,
                    to_track_id=b_id,
                    kind=EdgeKind.EMBEDDING_NEIGHBOR.value,
                    weight=weight,
                    meta=json.dumps({"cosine": cosine}),
                    computed_at=now,
                )
            )
        return edges

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _clear_existing(
        self, kind: EdgeKind, track_ids: list[int] | None
    ) -> None:
        """Delete the rows of ``kind`` we're about to rewrite."""
        q = self.session.query(TrackEdge).filter(TrackEdge.kind == kind.value)
        if track_ids:
            ids = list(track_ids)
            q = q.filter(
                or_(
                    TrackEdge.from_track_id.in_(ids),
                    TrackEdge.to_track_id.in_(ids),
                )
            )
        q.delete(synchronize_session=False)

    def _upsert_edges(self, edges: Iterable[TrackEdge]) -> int:
        """Upsert ``edges`` on ``(from_track_id, to_track_id, kind)``."""
        latest: dict[tuple[int, int, str], TrackEdge] = {}
        for e in edges:
            latest[(e.from_track_id, e.to_track_id, e.kind)] = e
        for (from_id, to_id, kind), new in latest.items():
            existing = (
                self.session.query(TrackEdge)
                .filter(
                    and_(
                        TrackEdge.from_track_id == from_id,
                        TrackEdge.to_track_id == to_id,
                        TrackEdge.kind == kind,
                    )
                )
                .first()
            )
            if existing is None:
                self.session.add(new)
            else:
                existing.weight = new.weight
                existing.meta = new.meta
                existing.computed_at = new.computed_at
        self.session.flush()
        return len(latest)
