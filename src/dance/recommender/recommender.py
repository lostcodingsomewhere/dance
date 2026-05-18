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
from dance.core.database import (
    AudioAnalysis,
    EdgeKind,
    StemFile,
    Track,
    TrackEdge,
    TrackEmbedding,
)
from dance.core.serialization import decode_embedding
from dance.pipeline.utils.camelot import get_compatible_keys


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


@dataclass
class ColumnRecResult:
    """One candidate for a per-column rec stream.

    ``stem_file_id`` is set for stem columns (drums/bass/vocals/other) and
    ``None`` for the mix column (where the candidate is a whole track).
    """

    track_id: int
    stem_file_id: int | None
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


# Columns that map to extracted stems vs the full-mix column.
_STEM_KINDS = ("drums", "bass", "vocals", "other")
_MIX_COLUMN = "mix"
VALID_COLUMNS = (*_STEM_KINDS, _MIX_COLUMN)


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


# ---------------------------------------------------------------------------
# Per-column recommender — the live-remixing rec stream backbone.
# ---------------------------------------------------------------------------

# Scoring weights for combining the three signals. Tuned by feel; keep the
# sum at 1.0 so a perfect candidate scores ~1.0.
_W_EMBED = 0.5
_W_KEY = 0.3
_W_BPM = 0.2


def _bpm_score(candidate_bpm: float | None, target_bpm: float | None) -> float:
    """Linear falloff: 1.0 at exact match, 0.0 at ±20 BPM or more.

    Both stems and full tracks store their *original* recording BPM. Live
    warps to master tempo, but warps farther than ~6% start sounding
    artifacted, so closer-original-BPM ≈ better-sounding swap.
    """
    if candidate_bpm is None or target_bpm is None:
        return 0.0
    delta = abs(float(candidate_bpm) - float(target_bpm))
    return max(0.0, 1.0 - delta / 20.0)


def _key_score(
    candidate_key: str | None,
    target_keys: list[str],
) -> float:
    """1.0 if candidate matches any target exactly, 0.6 if Camelot-compat,
    0.0 otherwise. ``target_keys`` is the bag of keys currently in the combo
    (one per stem, or one overall for the mix column)."""
    if not candidate_key or not target_keys:
        return 0.0
    candidate = candidate_key.upper()
    compat: set[str] = set()
    for t in target_keys:
        if t:
            compat.update(k.upper() for k in get_compatible_keys(t))
    if candidate in {k.upper() for k in target_keys if k}:
        return 1.0
    if candidate in compat:
        return 0.6
    return 0.0


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm


class _ColumnRecommenderImpl:
    """Stateful helper that loads the data once per request and scores.

    Pulled out of Recommender.recommend_by_column to keep that method short.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def run(
        self,
        column: str,
        combo_stem_ids: list[int],
        *,
        master_bpm: float | None,
        k: int,
        exclude_track_ids: list[int],
    ) -> list[ColumnRecResult]:
        if column not in VALID_COLUMNS:
            raise ValueError(f"unknown column {column!r}; valid: {VALID_COLUMNS}")

        excluded = set(exclude_track_ids)

        # 1. Aggregate combo embedding (averaged + normalized).
        combo_vec = self._combo_embedding(combo_stem_ids)
        combo_keys, combo_bpms = self._combo_keys_and_bpms(combo_stem_ids)
        target_bpm = master_bpm
        if target_bpm is None and combo_bpms:
            target_bpm = sum(combo_bpms) / len(combo_bpms)

        # 2. Pull candidates for this column.
        if column == _MIX_COLUMN:
            candidates = self._mix_candidates(excluded)
        else:
            candidates = self._stem_candidates(column, excluded)

        if not candidates:
            return []

        # 3. Score each candidate.
        results: list[ColumnRecResult] = []
        for cand in candidates:
            breakdown: dict[str, float] = {}
            reasons: list[str] = []

            # Embedding similarity (if we have a combo to compare against).
            embed_s = 0.0
            if combo_vec is not None and cand.embedding is not None:
                embed_s = float(np.dot(cand.embedding, combo_vec))
                # Map cosine [-1, 1] into [0, 1] so it composes with the
                # other scores cleanly; negatives mean "dissimilar".
                embed_s = max(0.0, (embed_s + 1.0) / 2.0)
                breakdown["embedding"] = embed_s

            # Key compat.
            key_s = _key_score(cand.key, combo_keys)
            breakdown["key"] = key_s
            if key_s >= 1.0:
                reasons.append("same key")
            elif key_s >= 0.6:
                reasons.append("harmonic-compat key")

            # BPM compat.
            bpm_s = _bpm_score(cand.bpm, target_bpm)
            breakdown["bpm"] = bpm_s
            if bpm_s >= 0.95 and target_bpm is not None:
                reasons.append("matched BPM")
            elif bpm_s >= 0.7 and target_bpm is not None:
                reasons.append("close BPM")

            # If no combo (empty), down-weight embedding to 0 and renormalize.
            if combo_vec is None:
                score = (key_s * _W_KEY + bpm_s * _W_BPM) / (_W_KEY + _W_BPM)
            else:
                score = embed_s * _W_EMBED + key_s * _W_KEY + bpm_s * _W_BPM

            results.append(
                ColumnRecResult(
                    track_id=cand.track_id,
                    stem_file_id=cand.stem_file_id,
                    score=score,
                    score_breakdown=breakdown,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _combo_embedding(self, combo_stem_ids: list[int]) -> np.ndarray | None:
        """Average + normalize the embeddings of the active combo stems.

        Returns None if no embeddings are available (empty combo or none of
        the active stems have embeddings yet).
        """
        if not combo_stem_ids:
            return None
        rows = (
            self.session.query(TrackEmbedding)
            .filter(TrackEmbedding.stem_file_id.in_(combo_stem_ids))
            .all()
        )
        if not rows:
            return None
        vecs = []
        for row in rows:
            v = decode_embedding(row.embedding, int(row.dim))
            vecs.append(_normalize(v.astype(np.float32, copy=False)))
        avg = np.mean(np.stack(vecs, axis=0), axis=0)
        return _normalize(avg)

    def _combo_keys_and_bpms(
        self, combo_stem_ids: list[int]
    ) -> tuple[list[str], list[float]]:
        if not combo_stem_ids:
            return [], []
        rows = (
            self.session.query(AudioAnalysis)
            .filter(AudioAnalysis.stem_file_id.in_(combo_stem_ids))
            .all()
        )
        keys: list[str] = []
        bpms: list[float] = []
        for r in rows:
            # Stem analyses store dominant_pitch_camelot, not key_camelot.
            pitch = getattr(r, "dominant_pitch_camelot", None) or getattr(
                r, "key_camelot", None
            )
            if pitch:
                keys.append(str(pitch))
            if r.bpm is not None:
                bpms.append(float(r.bpm))
        return keys, bpms

    def _stem_candidates(
        self, column: str, excluded: set[int]
    ) -> list["_Cand"]:
        rows = (
            self.session.query(
                StemFile.id,
                StemFile.track_id,
                TrackEmbedding.embedding,
                TrackEmbedding.dim,
                AudioAnalysis.bpm,
                AudioAnalysis.dominant_pitch_camelot,
            )
            .outerjoin(
                TrackEmbedding, TrackEmbedding.stem_file_id == StemFile.id
            )
            .outerjoin(
                AudioAnalysis, AudioAnalysis.stem_file_id == StemFile.id
            )
            .filter(StemFile.kind == column)
            .all()
        )
        out: list[_Cand] = []
        for stem_id, track_id, embed_blob, dim, bpm, key in rows:
            if int(track_id) in excluded:
                continue
            vec: np.ndarray | None = None
            if embed_blob and dim:
                vec = _normalize(
                    decode_embedding(embed_blob, int(dim)).astype(np.float32, copy=False)
                )
            out.append(
                _Cand(
                    stem_file_id=int(stem_id),
                    track_id=int(track_id),
                    embedding=vec,
                    bpm=float(bpm) if bpm is not None else None,
                    key=str(key) if key else None,
                )
            )
        return out

    def _mix_candidates(self, excluded: set[int]) -> list["_Cand"]:
        rows = (
            self.session.query(
                Track.id,
                TrackEmbedding.embedding,
                TrackEmbedding.dim,
                AudioAnalysis.bpm,
                AudioAnalysis.key_camelot,
            )
            .outerjoin(
                TrackEmbedding,
                (TrackEmbedding.track_id == Track.id)
                & (TrackEmbedding.stem_file_id.is_(None)),
            )
            .outerjoin(
                AudioAnalysis,
                (AudioAnalysis.track_id == Track.id)
                & (AudioAnalysis.stem_file_id.is_(None)),
            )
            .all()
        )
        out: list[_Cand] = []
        for track_id, embed_blob, dim, bpm, key in rows:
            if int(track_id) in excluded:
                continue
            vec: np.ndarray | None = None
            if embed_blob and dim:
                vec = _normalize(
                    decode_embedding(embed_blob, int(dim)).astype(np.float32, copy=False)
                )
            out.append(
                _Cand(
                    stem_file_id=None,
                    track_id=int(track_id),
                    embedding=vec,
                    bpm=float(bpm) if bpm is not None else None,
                    key=str(key) if key else None,
                )
            )
        return out


@dataclass
class _Cand:
    stem_file_id: int | None
    track_id: int
    embedding: np.ndarray | None
    bpm: float | None
    key: str | None


def recommend_by_column(
    session: Session,
    column: str,
    combo_stem_ids: list[int],
    *,
    master_bpm: float | None = None,
    k: int = 5,
    exclude_track_ids: list[int] | None = None,
) -> list[ColumnRecResult]:
    """Top-K candidate stems (or tracks for ``column='mix'``) for this column,
    re-scored against the currently-active combo. Stateless — load + score
    + sort in one call.
    """
    impl = _ColumnRecommenderImpl(session)
    return impl.run(
        column=column,
        combo_stem_ids=list(combo_stem_ids),
        master_bpm=master_bpm,
        k=k,
        exclude_track_ids=list(exclude_track_ids or []),
    )


__all__ = [
    "DEFAULT_WEIGHTS",
    "ColumnRecResult",
    "RecommendationResult",
    "Recommender",
    "VALID_COLUMNS",
    "recommend",
    "recommend_by_column",
]
