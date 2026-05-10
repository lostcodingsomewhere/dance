"""FastAPI dependency providers.

These wire request handlers to the singletons created in :func:`create_app`
(session factory, bridge, settings). Tests override the providers by name to
inject fakes.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Tag,
    Track,
    TrackTag,
)
from dance.osc.bridge import AbletonBridge


# These functions are placeholders — ``create_app`` overrides them via
# ``app.dependency_overrides`` so they are reachable for tests too.


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_bridge(request: Request) -> AbletonBridge:
    return request.app.state.bridge


def get_session(request: Request) -> Iterator[Session]:
    session_factory = request.app.state.session_factory
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Shared serialization helpers
# ---------------------------------------------------------------------------


def fullmix_analysis(session: Session, track_id: int) -> AudioAnalysis | None:
    """Return the full-mix AudioAnalysis row for ``track_id``, or ``None``."""
    return (
        session.query(AudioAnalysis)
        .filter(
            AudioAnalysis.track_id == track_id,
            AudioAnalysis.stem_file_id.is_(None),
        )
        .one_or_none()
    )


def track_tag_values(session: Session, track_id: int) -> list[str]:
    rows = (
        session.query(Tag.value)
        .join(TrackTag, TrackTag.tag_id == Tag.id)
        .filter(TrackTag.track_id == track_id)
        .distinct()
        .all()
    )
    return [v for (v,) in rows]


def track_to_out(session: Session, track: Track) -> dict:
    """Build the dict that maps to ``TrackOut``.

    Returns a plain dict so callers can let pydantic v2 validate the response.
    """
    analysis = fullmix_analysis(session, int(track.id))
    return {
        "id": int(track.id),
        "file_path": track.file_path,
        "title": track.title,
        "artist": track.artist,
        "duration_seconds": track.duration_seconds,
        "state": track.state,
        "analysis": analysis,
        "tags": track_tag_values(session, int(track.id)),
    }


__all__ = [
    "fullmix_analysis",
    "get_bridge",
    "get_session",
    "get_settings",
    "track_tag_values",
    "track_to_out",
    "Depends",
]
