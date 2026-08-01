"""
Query the recommendation graph.

Given seed track IDs, return ranked candidate tracks using live scoring via
the shared core in :mod:`dance.recommender.scoring`.  The candidate pool is
the ``EMBEDDING_NEIGHBOR`` edges materialised by :mod:`dance.recommender.graph_builder`;
key/BPM compatibility is computed at query time from the seed journey so no
harmonic/tempo edges need to be stored.

Also exposes :meth:`Recommender.recommend_by_text` — CLAP is a joint
audio/text model so an arbitrary natural-language query ("punchy techy with
vocals") can rank tracks directly without going through tags.

The per-column live recommender (the Booth backbone) scores entirely through
the shared core in :mod:`dance.recommender.scoring` and the journey context in
:mod:`dance.recommender.journey` — no compatibility maths is reimplemented here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from dance.core.database import (
    AudioAnalysis,
    EdgeKind,
    StemFile,
    Track,
    TrackEdge,
    TrackEmbedding,
)
from dance.core.serialization import decode_embedding
from dance.recommender import dedup, scoring
from dance.recommender.journey import JourneyState, journey_from_combo, journey_from_tracks


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
        exclude: list[int] | None = None,
    ) -> list[RecommendationResult]:
        """Return the top-``k`` candidates for ``seeds`` using live scoring.

        Candidate pool: tracks reachable via ``EMBEDDING_NEIGHBOR`` edges from
        any seed.  Each candidate is scored with the shared
        :func:`~dance.recommender.scoring.combine` function, using a journey
        built from the seeds (embedding + key + BPM).  Seeds and ``exclude``
        tracks are omitted from results.
        """
        if not seeds:
            return []

        excluded: set[int] = set(exclude or [])
        seed_set: set[int] = set(seeds)

        # 1. Candidate pool from the materialized embedding-neighbor edges.
        edges = (
            self.session.query(TrackEdge)
            .filter(
                TrackEdge.from_track_id.in_(list(seed_set)),
                TrackEdge.kind == EdgeKind.EMBEDDING_NEIGHBOR.value,
            )
            .all()
        )
        cand_ids: set[int] = set()
        for edge in edges:
            target = int(edge.to_track_id)
            if target not in seed_set and target not in excluded:
                cand_ids.add(target)

        if not cand_ids:
            return []

        # 2. Build the journey context from the ordered seeds.
        journey = journey_from_tracks(self.session, list(seeds), window=len(seeds))

        # 3. Load embeddings + analysis for every candidate in one query.
        embed_rows = (
            self.session.query(TrackEmbedding)
            .filter(
                TrackEmbedding.track_id.in_(cand_ids),
                TrackEmbedding.stem_file_id.is_(None),
            )
            .all()
        )
        analysis_rows = (
            self.session.query(AudioAnalysis)
            .filter(
                AudioAnalysis.track_id.in_(cand_ids),
                AudioAnalysis.stem_file_id.is_(None),
            )
            .all()
        )

        embeds: dict[int, np.ndarray | None] = {
            int(r.track_id): _decode(r.embedding, r.dim) for r in embed_rows
        }
        analysis_by_tid: dict[int, AudioAnalysis] = {
            int(r.track_id): r for r in analysis_rows
        }

        # 4. Score each candidate through the shared core.
        _profile = scoring.Profile("seed", {"embedding": 0.5, "key": 0.25, "bpm": 0.25})
        results: list[RecommendationResult] = []
        for tid in cand_ids:
            vec = embeds.get(tid)
            row = analysis_by_tid.get(tid)
            key = str(row.key_camelot) if row and row.key_camelot else None
            bpm = float(row.bpm) if row and row.bpm is not None else None
            features: dict[str, float | None] = {
                "embedding": journey.vibe_score(vec),
                "key": scoring.key_score(key, journey.target_keys),
                "bpm": scoring.bpm_score(bpm, journey.target_bpm),
            }
            score, breakdown = scoring.combine(features, _profile)
            reasons = [
                {"kind": name, "value": round(v, 3)} for name, v in breakdown.items()
            ]
            results.append(RecommendationResult(track_id=tid, score=score, reasons=reasons))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

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

# Soft penalty applied to a candidate whose source track was played recently
# (it's in the journey's trailing window). Discourages re-introducing the same
# song without hard-excluding it.
_REPEAT_PENALTY = 0.5


@dataclass
class _Cand:
    stem_file_id: int | None
    track_id: int
    embedding: np.ndarray | None
    bpm: float | None
    key: str | None
    bpm_confidence: float | None = None
    key_confidence: float | None = None
    kick_density: float | None = None
    presence: float | None = None
    brightness: float | None = None
    warmth: float | None = None


@dataclass
class _ComboTargets:
    """Timbral aggregates of the active combo — the micro-scale targets the
    journey object doesn't carry (it owns vibe/key/BPM/energy)."""

    kick_density: float | None = None
    presence: float | None = None
    brightness: float | None = None
    warmth: float | None = None


class _ColumnRecommenderImpl:
    """Stateful helper that loads the data once per request and scores via the
    shared core. Pulled out of :meth:`Recommender.recommend_by_column` to keep
    that wrapper short."""

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
        trailing_track_ids: list[int] | None,
    ) -> list[ColumnRecResult]:
        if column not in VALID_COLUMNS:
            raise ValueError(f"unknown column {column!r}; valid: {VALID_COLUMNS}")

        excluded = set(exclude_track_ids)

        # 1. Build the journey context (vibe/key/BPM + trend + anti-repetition)
        #    from the active combo plus the trailing live set.
        journey = journey_from_combo(
            self.session,
            combo_stem_ids,
            master_bpm=master_bpm,
            trailing_track_ids=trailing_track_ids,
        )
        targets = self._combo_targets(combo_stem_ids)
        profile = scoring.profile_for_column(column)

        # 2. Pull candidates for this column.
        if column == _MIX_COLUMN:
            candidates = self._mix_candidates(excluded)
        else:
            candidates = self._stem_candidates(column, excluded)
        if not candidates:
            return []

        # 3. Score each candidate through the shared core.
        results: list[ColumnRecResult] = []
        for cand in candidates:
            features = self._features(cand, journey, targets)
            score, breakdown = scoring.combine(features, profile)
            if cand.track_id in journey.used_source_ids:
                score *= _REPEAT_PENALTY
            results.append(
                ColumnRecResult(
                    track_id=cand.track_id,
                    stem_file_id=cand.stem_file_id,
                    score=score,
                    score_breakdown=breakdown,
                    reasons=_reasons(breakdown, column),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        # Collapse duplicate RECORDINGS before truncating, so a rec list never
        # spends two of its k slots on the same song ingested twice. Done after
        # the sort so the copy the scorer preferred is the one kept, and before
        # [:k] so the freed slots go to real alternatives. Extended-vs-radio
        # variants survive — see dedup.SAME_RECORDING_TOLERANCE_S.
        results = dedup.dedupe_by_recording(self.session, results)
        return results[:k]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _features(
        self, cand: _Cand, journey: JourneyState, targets: _ComboTargets
    ) -> dict[str, float | None]:
        """Per-candidate feature dict consumed by :func:`scoring.combine`.

        Key/BPM are confidence-gated so shaky analysis contributes less.
        ``combine`` ignores any feature the column profile doesn't weight, so we
        can compute the full set unconditionally.
        """
        return {
            "embedding": journey.vibe_score(cand.embedding),
            "key": scoring.confidence_gate(
                scoring.key_score(cand.key, journey.target_keys), cand.key_confidence
            ),
            "bpm": scoring.confidence_gate(
                scoring.bpm_score(cand.bpm, journey.target_bpm), cand.bpm_confidence
            ),
            "kick_density": scoring.kick_density_score(cand.kick_density, targets.kick_density),
            "presence": scoring.presence_score(cand.presence, targets.presence),
            "timbre": scoring.timbre_score(
                (cand.brightness, cand.warmth), (targets.brightness, targets.warmth)
            ),
        }

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _combo_targets(self, combo_stem_ids: list[int]) -> _ComboTargets:
        """Mean timbral signals across the active combo stems."""
        if not combo_stem_ids:
            return _ComboTargets()
        rows = (
            self.session.query(
                AudioAnalysis.kick_density,
                AudioAnalysis.presence_ratio,
                AudioAnalysis.brightness,
                AudioAnalysis.warmth,
            )
            .filter(AudioAnalysis.stem_file_id.in_(combo_stem_ids))
            .all()
        )
        return _ComboTargets(
            kick_density=_mean([r[0] for r in rows]),
            presence=_mean([r[1] for r in rows]),
            brightness=_mean([r[2] for r in rows]),
            warmth=_mean([r[3] for r in rows]),
        )

    def _stem_candidates(self, column: str, excluded: set[int]) -> list[_Cand]:
        rows = (
            self.session.query(
                StemFile.id,
                StemFile.track_id,
                TrackEmbedding.embedding,
                TrackEmbedding.dim,
                AudioAnalysis.bpm,
                AudioAnalysis.bpm_confidence,
                AudioAnalysis.dominant_pitch_camelot,
                AudioAnalysis.dominant_pitch_confidence,
                AudioAnalysis.kick_density,
                AudioAnalysis.presence_ratio,
                AudioAnalysis.brightness,
                AudioAnalysis.warmth,
            )
            .outerjoin(TrackEmbedding, TrackEmbedding.stem_file_id == StemFile.id)
            .outerjoin(AudioAnalysis, AudioAnalysis.stem_file_id == StemFile.id)
            .join(Track, Track.id == StemFile.track_id)
            .filter(StemFile.kind == column)
            # Redundant copies never enter the candidate pool. 125 of 357
            # rows in this library are the same recording ingested more than
            # once (docs/proposals/library-duplicates.md); without this a rec
            # list spends its slots showing you the same song twice.
            .filter(Track.duplicate_of.is_(None))
            .all()
        )
        out: list[_Cand] = []
        for row in rows:
            (
                stem_id,
                track_id,
                embed_blob,
                dim,
                bpm,
                bpm_conf,
                key,
                key_conf,
                kick_density,
                presence,
                brightness,
                warmth,
            ) = row
            if int(track_id) in excluded:
                continue
            out.append(
                _Cand(
                    stem_file_id=int(stem_id),
                    track_id=int(track_id),
                    embedding=_decode(embed_blob, dim),
                    bpm=_f(bpm),
                    key=str(key) if key else None,
                    bpm_confidence=_f(bpm_conf),
                    key_confidence=_f(key_conf),
                    kick_density=_f(kick_density),
                    presence=_f(presence),
                    brightness=_f(brightness),
                    warmth=_f(warmth),
                )
            )
        return out

    def _mix_candidates(self, excluded: set[int]) -> list[_Cand]:
        rows = (
            self.session.query(
                Track.id,
                TrackEmbedding.embedding,
                TrackEmbedding.dim,
                AudioAnalysis.bpm,
                AudioAnalysis.bpm_confidence,
                AudioAnalysis.key_camelot,
                AudioAnalysis.key_confidence,
                AudioAnalysis.presence_ratio,
                AudioAnalysis.brightness,
                AudioAnalysis.warmth,
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
            # Same reason as _stem_candidates — see there.
            .filter(Track.duplicate_of.is_(None))
            .all()
        )
        out: list[_Cand] = []
        for row in rows:
            (
                track_id,
                embed_blob,
                dim,
                bpm,
                bpm_conf,
                key,
                key_conf,
                presence,
                brightness,
                warmth,
            ) = row
            if int(track_id) in excluded:
                continue
            out.append(
                _Cand(
                    stem_file_id=None,
                    track_id=int(track_id),
                    embedding=_decode(embed_blob, dim),
                    bpm=_f(bpm),
                    key=str(key) if key else None,
                    bpm_confidence=_f(bpm_conf),
                    key_confidence=_f(key_conf),
                    presence=_f(presence),
                    brightness=_f(brightness),
                    warmth=_f(warmth),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _reasons(breakdown: dict[str, float], column: str) -> list[str]:
    """Human-readable explanations derived from the score breakdown."""
    out: list[str] = []
    embed = breakdown.get("embedding")
    if embed is not None and embed >= 0.8:
        out.append("vibe match")
    key = breakdown.get("key")
    if key is not None and key >= 1.0:
        out.append("same key")
    elif key is not None and key >= 0.6:
        out.append("harmonic-compat key")
    bpm = breakdown.get("bpm")
    if bpm is not None and bpm >= 0.95:
        out.append("matched BPM")
    elif bpm is not None and bpm >= 0.7:
        out.append("close BPM")
    kick = breakdown.get("kick_density")
    if kick is not None and kick >= 0.8:
        out.append("tight drums")
    timbre = breakdown.get("timbre")
    if timbre is not None and timbre >= 0.8:
        out.append("similar texture")
    return out


def _decode(blob, dim) -> np.ndarray | None:
    if not blob or not dim:
        return None
    v = decode_embedding(blob, int(dim)).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(v)) or 1.0
    return v / norm


def _f(value) -> float | None:
    return float(value) if value is not None else None


def _mean(values: list) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def recommend_by_column(
    session: Session,
    column: str,
    combo_stem_ids: list[int],
    *,
    master_bpm: float | None = None,
    k: int = 5,
    exclude_track_ids: list[int] | None = None,
    trailing_track_ids: list[int] | None = None,
) -> list[ColumnRecResult]:
    """Top-K candidate stems (or tracks for ``column='mix'``) for this column,
    re-scored against the active combo through the shared scoring core.

    ``trailing_track_ids`` is the recent live-set sequence; when supplied the
    recommender gains trend-aware vibe and a soft anti-repetition penalty.
    Stateless — load + score + sort in one call.
    """
    impl = _ColumnRecommenderImpl(session)
    return impl.run(
        column=column,
        combo_stem_ids=list(combo_stem_ids),
        master_bpm=master_bpm,
        k=k,
        exclude_track_ids=list(exclude_track_ids or []),
        trailing_track_ids=list(trailing_track_ids) if trailing_track_ids else None,
    )


__all__ = [
    "ColumnRecResult",
    "RecommendationResult",
    "Recommender",
    "VALID_COLUMNS",
    "recommend",
    "recommend_by_column",
]
