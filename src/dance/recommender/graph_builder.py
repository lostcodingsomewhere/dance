"""
Recommendation graph builder.

Reads per-track analysis (key, BPM, tags, embeddings) and writes the
``track_edges`` table. Operates on the whole library at once, or on a
``track_ids`` subset for incremental rebuilds.

Each edge kind has its own private method so they can be reasoned about
independently. All symmetric kinds materialize BOTH directions so downstream
queries can use a simple ``from_track_id = X`` lookup.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import numpy as np
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    now_utc,
    AudioAnalysis,
    EdgeKind,
    TrackEdge,
    TrackEmbedding,
    TrackTag,
)
from dance.core.serialization import decode_embedding
from dance.pipeline.utils.camelot import get_compatible_keys

logger = logging.getLogger(__name__)

ALL_KINDS: tuple[EdgeKind, ...] = (
    EdgeKind.HARMONIC_COMPAT,
    EdgeKind.TEMPO_COMPAT,
    EdgeKind.EMBEDDING_NEIGHBOR,
    EdgeKind.TAG_OVERLAP,
)


class GraphBuilder:
    """Populate ``track_edges`` from per-track analysis."""

    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    def build(
        self,
        track_ids: list[int] | None = None,
        kinds: list[EdgeKind] | None = None,
    ) -> dict[str, int]:
        """Build (or rebuild) edges, returning per-kind counts of rows written.

        ``track_ids=None`` rebuilds globally; otherwise only edges touching
        the listed tracks are deleted-and-rewritten. ``kinds=None`` builds
        every kind.
        """
        kinds = list(kinds) if kinds is not None else list(ALL_KINDS)
        dispatch = {
            EdgeKind.HARMONIC_COMPAT: self._build_harmonic,
            EdgeKind.TEMPO_COMPAT: self._build_tempo,
            EdgeKind.EMBEDDING_NEIGHBOR: self._build_embedding,
            EdgeKind.TAG_OVERLAP: self._build_tag_overlap,
        }
        results: dict[str, int] = {}
        for kind in kinds:
            fn = dispatch.get(kind)
            if fn is None:
                logger.warning("Unsupported edge kind: %s", kind)
                continue
            self._clear_existing(kind, track_ids)
            edges = fn(track_ids)
            count = self._upsert_edges(edges)
            results[kind.value] = count
            self.session.commit()
            logger.info("Built %d %s edges", count, kind.value)
        return results

    # ------------------------------------------------------------------
    # Edge-kind builders
    # ------------------------------------------------------------------

    def _build_harmonic(self, track_ids: list[int] | None) -> list[TrackEdge]:
        rows = (
            self.session.query(AudioAnalysis.track_id, AudioAnalysis.key_camelot)
            .filter(
                AudioAnalysis.stem_file_id.is_(None),
                AudioAnalysis.key_camelot.isnot(None),
            )
            .all()
        )
        track_keys: list[tuple[int, str]] = [
            (tid, key.upper()) for tid, key in rows if key
        ]
        now = now_utc()
        touched = set(track_ids) if track_ids else None
        edges: list[TrackEdge] = []
        for a_id, a_key in track_keys:
            compatible = set(get_compatible_keys(a_key))
            for b_id, b_key in track_keys:
                if a_id == b_id or b_key not in compatible:
                    continue
                if touched is not None and a_id not in touched and b_id not in touched:
                    continue
                weight, distance = _harmonic_weight(a_key, b_key)
                if weight is None:
                    continue
                edges.append(
                    TrackEdge(
                        from_track_id=a_id,
                        to_track_id=b_id,
                        kind=EdgeKind.HARMONIC_COMPAT.value,
                        weight=weight,
                        meta=json.dumps({"distance": distance}),
                        computed_at=now,
                    )
                )
        return edges

    def _build_tempo(self, track_ids: list[int] | None) -> list[TrackEdge]:
        rows = (
            self.session.query(AudioAnalysis.track_id, AudioAnalysis.bpm)
            .filter(
                AudioAnalysis.stem_file_id.is_(None),
                AudioAnalysis.bpm.isnot(None),
            )
            .all()
        )
        track_bpms: list[tuple[int, float]] = [(tid, float(bpm)) for tid, bpm in rows]
        now = now_utc()
        touched = set(track_ids) if track_ids else None
        edges: list[TrackEdge] = []
        for a_id, a_bpm in track_bpms:
            for b_id, b_bpm in track_bpms:
                if a_id == b_id:
                    continue
                if touched is not None and a_id not in touched and b_id not in touched:
                    continue
                weight, delta, halftime = _tempo_weight(a_bpm, b_bpm)
                if weight is None:
                    continue
                edges.append(
                    TrackEdge(
                        from_track_id=a_id,
                        to_track_id=b_id,
                        kind=EdgeKind.TEMPO_COMPAT.value,
                        weight=weight,
                        meta=json.dumps({"bpm_delta": delta, "halftime": halftime}),
                        computed_at=now,
                    )
                )
        return edges

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

    def _build_tag_overlap(self, track_ids: list[int] | None) -> list[TrackEdge]:
        rows = self.session.query(TrackTag.track_id, TrackTag.tag_id).all()
        by_track: dict[int, set[int]] = defaultdict(set)
        for tid, tag_id in rows:
            by_track[int(tid)].add(int(tag_id))
        items = [(tid, tags) for tid, tags in by_track.items() if tags]
        now = now_utc()
        touched = set(track_ids) if track_ids else None
        edges: list[TrackEdge] = []
        for i in range(len(items)):
            a_id, a_tags = items[i]
            for j in range(i + 1, len(items)):
                b_id, b_tags = items[j]
                if touched is not None and a_id not in touched and b_id not in touched:
                    continue
                shared = a_tags & b_tags
                if not shared:
                    continue
                union = a_tags | b_tags
                jaccard = len(shared) / len(union)
                if jaccard <= 0.1:
                    continue
                meta = json.dumps(
                    {"shared_count": len(shared), "total_count": len(union)}
                )
                for src, dst in ((a_id, b_id), (b_id, a_id)):
                    edges.append(
                        TrackEdge(
                            from_track_id=src,
                            to_track_id=dst,
                            kind=EdgeKind.TAG_OVERLAP.value,
                            weight=float(jaccard),
                            meta=meta,
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


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _harmonic_weight(a_key: str, b_key: str) -> tuple[float | None, str | None]:
    """Return (weight, distance) for a pair of compatible Camelot keys.

    Caller must have already confirmed ``b_key in get_compatible_keys(a_key)``.
    """
    if a_key == b_key:
        return 1.0, "exact"
    try:
        a_num, a_letter = int(a_key[:-1]), a_key[-1]
        b_num, b_letter = int(b_key[:-1]), b_key[-1]
    except (ValueError, IndexError):
        return None, None
    if a_num == b_num and a_letter != b_letter:
        return 0.8, "relative"
    if a_letter == b_letter:
        diff = abs(a_num - b_num)
        if diff == 1 or diff == 11:  # 11 handles the 12<->1 wrap
            return 0.7, "adjacent"
    return None, None


def _tempo_weight(a_bpm: float, b_bpm: float) -> tuple[float | None, float, bool]:
    """Return (weight, delta, halftime_flag) for a BPM pair, or (None, ...)."""
    delta = abs(a_bpm - b_bpm)
    if delta <= 3:
        return 1.0, delta, False
    if delta <= 6:
        return 0.7, delta, False
    if abs(a_bpm - b_bpm * 2) <= 3 or abs(a_bpm * 2 - b_bpm) <= 3:
        return 0.5, delta, True
    return None, delta, False
