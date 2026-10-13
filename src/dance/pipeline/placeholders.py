"""
Placeholder Track rows from the optimistic Spotify ingest.

``POST /pipeline/ingest/track`` and ``/ingest/playlist`` create the Track
row *before* the audio exists so the UI can show it right away. The row
gets ``file_hash = "pending:<spotify_id>"`` (to satisfy the unique
constraint) and ``state = pending``. The real SHA256 replaces the
placeholder once the download lands, either via the download worker's
finalize step or the library scanner adopting the row by path.

Until then the row must not be claimed by any stage. Its audio file is
still being written (or doesn't exist yet), so analysis would mark an
in-flight download ERROR with "Track file not found".
"""

from __future__ import annotations

from typing import Any

from dance.core.database import Track

PLACEHOLDER_HASH_PREFIX = "pending:"


def is_placeholder_hash(file_hash: str | None) -> bool:
    return bool(file_hash) and str(file_hash).startswith(PLACEHOLDER_HASH_PREFIX)


def not_placeholder() -> Any:
    """SQLAlchemy criterion: rows whose download has landed (claimable).

    ``file_hash`` is NOT NULL, so a plain ``NOT LIKE`` is exact.
    """
    return ~Track.file_hash.like(f"{PLACEHOLDER_HASH_PREFIX}%")
