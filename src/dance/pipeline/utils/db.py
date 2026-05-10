"""Tiny DB helpers shared across stages.

Keep this module very small — only patterns that are used in 3+ places belong
here.
"""

from __future__ import annotations

from typing import Any, TypeVar

from sqlalchemy.orm import Session

from dance.core.database import StemFile

T = TypeVar("T")


def upsert(session: Session, model: type[T], where: dict[str, Any], **values) -> T:
    """Find a row matching ``where`` or create a new one; then set ``values``.

    The row is added to the session if new but **not committed** — the caller
    decides commit boundaries. Returns the (possibly new) row.
    """
    row = session.query(model).filter_by(**where).first()
    if row is None:
        row = model(**where)
        session.add(row)
    for key, value in values.items():
        setattr(row, key, value)
    return row


def get_stems_for_track(session: Session, track_id: int) -> list[StemFile]:
    """Return all ``StemFile`` rows for a track (deterministic order by id)."""
    return (
        session.query(StemFile)
        .filter(StemFile.track_id == track_id)
        .order_by(StemFile.id)
        .all()
    )
