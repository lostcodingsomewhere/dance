"""Which stored CLAP embeddings are on the current scheme.

``track_embeddings.model_version`` records *how* a vector was produced, not
just which weights produced it. Vectors made different ways do not live in a
comparable space — the old scheme stored one random 10 s crop per audio, the
current one a mean over evenly spread non-silent windows — so every reader
must compare like with like. Readers filter with :func:`current_scheme`; the
embed stage writes :data:`CLAP_EMBED_VERSION` and deletes the rows of any
other scheme for the same (track, stem, model).

Kept free of torch/transformers so the recommender can import it cheaply.
"""

from __future__ import annotations

from sqlalchemy.sql.elements import ColumnElement

from dance.core.database import TrackEmbedding

# Scheme id stored in ``track_embeddings.model_version`` (String(32)).
# win = the model's native window length, never randomly cropped;
# nonsilent = near-silent windows skipped; mean = per-window L2-normalised,
# mean-pooled, re-normalised. Bump the suffix whenever the vectors change.
CLAP_EMBED_VERSION = "win-nonsilent-mean-v1"


def current_scheme() -> ColumnElement[bool]:
    """SQL condition selecting only embeddings written by the current scheme."""
    return TrackEmbedding.model_version == CLAP_EMBED_VERSION
