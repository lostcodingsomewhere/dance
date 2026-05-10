"""
Recommendation graph and query layer for Dance.

The recommender is a **library-level** concern, distinct from per-track
pipeline stages. ``GraphBuilder`` walks the entire track table (or an
incremental subset) and writes ``track_edges`` rows describing pairwise
compatibility/similarity. ``Recommender`` queries that graph to produce
ordered suggestions seeded by a list of track IDs.
"""

from dance.recommender.graph_builder import GraphBuilder
from dance.recommender.recommender import (
    RecommendationResult,
    Recommender,
    recommend,
)

__all__ = [
    "GraphBuilder",
    "Recommender",
    "RecommendationResult",
    "recommend",
]
