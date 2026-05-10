"""CLAP zero-shot tagger.

CLAP is a joint audio↔text embedding model. We exploit that by:

1. Computing CLAP text embeddings for a fixed vocabulary of candidate labels
   (subgenres, moods, elements, dj_notes). Done once, cached.
2. For each track, taking the stored full-mix audio embedding and ranking
   candidate labels by cosine similarity.
3. Writing the top-K labels per category to ``track_tags`` with
   ``source = inferred``.

No new model weights. No API key. ~50 ms / track.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    Tag,
    TagKind,
    TagSource,
    Track,
    TrackEmbedding,
    TrackTag,
    normalize_tag_value,
    now_utc,
)
from dance.core.serialization import decode_embedding
from dance.pipeline.utils.db import upsert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------
#
# The vocabulary is small and curated rather than auto-generated — we want
# tags that are useful for DJ set building, not the long tail of every
# possible descriptor. Add to these lists carefully; each new label costs
# one CLAP text encoding (negligible) but dilutes the relative scores.


SUBGENRE_LABELS: list[str] = [
    "tech house",
    "deep house",
    "minimal house",
    "afro house",
    "progressive house",
    "melodic techno",
    "industrial techno",
    "minimal techno",
    "trance",
    "drum and bass",
    "dubstep",
    "electro",
    "breakbeat",
    "garage",
    "hip-hop",
    "ambient",
    "downtempo",
    "disco",
]

MOOD_LABELS: list[str] = [
    "dark",
    "uplifting",
    "driving",
    "hypnotic",
    "groovy",
    "aggressive",
    "euphoric",
    "melancholic",
    "minimal",
    "atmospheric",
    "industrial",
    "warm",
    "cold",
    "trippy",
    "punchy",
    "rolling",
    "spacey",
    "raw",
]

ELEMENT_LABELS: list[str] = [
    "acid line",
    "vocal chops",
    "rolling bassline",
    "sub bass",
    "tribal percussion",
    "filter sweep",
    "riser",
    "big synth lead",
    "arpeggio",
    "lush pads",
    "strings",
    "guitar",
    "piano",
    "saxophone",
    "vinyl crackle",
    "tape hiss",
    "spoken word",
    "female vocal",
    "male vocal",
]

DJ_NOTE_LABELS: list[str] = [
    "peak-time banger",
    "warm-up groove",
    "opener",
    "closer",
    "breakdown-heavy",
    "long intro",
    "long outro",
    "vocal-led",
    "instrumental",
    "tool",
    "transition track",
]


_VOCAB_BY_KIND: dict[TagKind, list[str]] = {
    TagKind.SUBGENRE: SUBGENRE_LABELS,
    TagKind.MOOD: MOOD_LABELS,
    TagKind.ELEMENT: ELEMENT_LABELS,
    TagKind.DJ_NOTE: DJ_NOTE_LABELS,
}


# ---------------------------------------------------------------------------


@dataclass
class TaggerResponse:
    subgenre: str | None = None
    mood_tags: list[str] = field(default_factory=list)
    element_tags: list[str] = field(default_factory=list)
    dj_notes: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    model: str | None = None

    def all_tags(self) -> list[tuple[TagKind, str]]:
        out: list[tuple[TagKind, str]] = []
        if self.subgenre:
            out.append((TagKind.SUBGENRE, self.subgenre))
        for v in self.mood_tags:
            out.append((TagKind.MOOD, v))
        for v in self.element_tags:
            out.append((TagKind.ELEMENT, v))
        for v in self.dj_notes:
            out.append((TagKind.DJ_NOTE, v))
        return out


# ---------------------------------------------------------------------------


class ClapZeroShotTagger:
    """Rank a controlled vocabulary of labels by CLAP cosine vs the track's audio."""

    source = TagSource.INFERRED

    def __init__(self, settings: Settings, *, text_encoder=None) -> None:
        """Args:
            settings: source of thresholds + top_k config + CLAP model name.
            text_encoder: callable ``(str) -> np.ndarray`` returning a
                normalized CLAP text embedding. If None, an
                :class:`~dance.pipeline.stages.embed.EmbeddingStage` is loaded
                lazily on first call.
        """
        self.settings = settings
        self._text_encoder = text_encoder
        self._embedding_stage = None
        self._label_cache: dict[TagKind, tuple[list[str], np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------

    def _ensure_encoder(self):
        if self._text_encoder is not None:
            return self._text_encoder
        from dance.pipeline.stages.embed import EmbeddingStage

        if self._embedding_stage is None:
            stage = EmbeddingStage()
            stage._ensure_model(self.settings)
            self._embedding_stage = stage
        self._text_encoder = self._embedding_stage.encode_text
        return self._text_encoder

    def _label_matrix(self, kind: TagKind) -> tuple[list[str], np.ndarray]:
        """Return (labels, normalized embedding matrix) for one tag kind, cached."""
        if kind in self._label_cache:
            return self._label_cache[kind]

        with self._cache_lock:
            if kind in self._label_cache:
                return self._label_cache[kind]

            encoder = self._ensure_encoder()
            labels = list(_VOCAB_BY_KIND[kind])
            vectors: list[np.ndarray] = []
            for label in labels:
                v = encoder(label).astype(np.float32, copy=False)
                norm = float(np.linalg.norm(v)) or 1.0
                vectors.append(v / norm)
            matrix = np.stack(vectors, axis=0)
            self._label_cache[kind] = (labels, matrix)
            return labels, matrix

    # ------------------------------------------------------------------

    def _track_embedding(self, session: Session, track: Track) -> np.ndarray | None:
        """Load the full-mix CLAP embedding for a track, L2-normalized."""
        row = (
            session.query(TrackEmbedding)
            .filter(
                TrackEmbedding.track_id == track.id,
                TrackEmbedding.stem_file_id.is_(None),
                TrackEmbedding.model == self.settings.clap_model,
            )
            .first()
        )
        if row is None:
            return None
        v = decode_embedding(row.embedding, int(row.dim)).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    # ------------------------------------------------------------------

    def tag_track(self, session: Session, track: Track) -> TaggerResponse:
        if not self.settings.tagger_enabled:
            raise RuntimeError("CLAP zero-shot tagger disabled in settings")

        audio_vec = self._track_embedding(session, track)
        if audio_vec is None:
            raise RuntimeError(
                f"track {track.id} has no full-mix CLAP embedding — "
                "run the embed stage first"
            )

        result = TaggerResponse(model=self.settings.clap_model)
        threshold = float(self.settings.tagger_zeroshot_threshold)
        top_k_map = dict(self.settings.tagger_zeroshot_top_k)

        for kind in (TagKind.SUBGENRE, TagKind.MOOD, TagKind.ELEMENT, TagKind.DJ_NOTE):
            labels, matrix = self._label_matrix(kind)
            scores = matrix @ audio_vec  # cosine, since both normalized
            # Top-k by score, then drop anything below threshold.
            top_k = top_k_map.get(kind.value, 3)
            order = np.argsort(-scores)
            picks: list[tuple[str, float]] = []
            for i in order[:top_k]:
                s = float(scores[i])
                if s < threshold:
                    break  # sorted descending — rest are worse
                picks.append((labels[i], s))

            # Stash scores for debugging / UI.
            for label, score in picks:
                result.scores[f"{kind.value}:{label}"] = score

            if kind is TagKind.SUBGENRE:
                result.subgenre = picks[0][0] if picks else None
            elif kind is TagKind.MOOD:
                result.mood_tags = [label for label, _ in picks]
            elif kind is TagKind.ELEMENT:
                result.element_tags = [label for label, _ in picks]
            elif kind is TagKind.DJ_NOTE:
                result.dj_notes = [label for label, _ in picks]

        self._write_tags(session, track, result)
        return result

    # ------------------------------------------------------------------

    def _write_tags(
        self,
        session: Session,
        track: Track,
        parsed: TaggerResponse,
    ) -> None:
        """Replace this track's tags from this source; keep other sources."""
        (
            session.query(TrackTag)
            .filter(
                TrackTag.track_id == track.id,
                TrackTag.source == self.source.value,
            )
            .delete(synchronize_session=False)
        )

        now = now_utc()
        seen: set[tuple[str, str]] = set()
        for kind, value in parsed.all_tags():
            value = value.strip()
            if not value:
                continue
            normalized = normalize_tag_value(value)
            key = (kind.value, normalized)
            if key in seen:
                continue
            seen.add(key)

            tag = upsert(
                session,
                Tag,
                where={"kind": kind.value, "normalized_value": normalized},
                value=value,
                created_at=now,
            )
            session.flush()

            score = parsed.scores.get(f"{kind.value}:{value}")
            upsert(
                session,
                TrackTag,
                where={
                    "track_id": track.id,
                    "tag_id": tag.id,
                    "source": self.source.value,
                },
                confidence=score,
                created_at=now,
            )
        session.commit()

    # ------------------------------------------------------------------

    def vocabulary(self) -> dict[str, Iterable[str]]:
        """Return the controlled vocabulary by kind (for docs / UI listings)."""
        return {k.value: list(v) for k, v in _VOCAB_BY_KIND.items()}
