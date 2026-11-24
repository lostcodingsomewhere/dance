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
from dance.pipeline.utils.pads import MIX_ROLE
from dance.recommender.recommender import ColumnRecResult
from dance.recommender.versions import LeadingVersion, leading_versions, version_index


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
    versions = version_index(session)
    return {
        "id": int(track.id),
        "file_path": track.file_path,
        "title": track.title,
        "artist": track.artist,
        "duration_seconds": track.duration_seconds,
        "state": track.state,
        "analysis": analysis,
        "tags": track_tag_values(session, int(track.id)),
        "version_count": len(versions.versions_of(int(track.id))),
        "version_family": versions.family_id(int(track.id)),
    }


def column_rec_out(
    session: Session, rec: ColumnRecResult, leading: dict[int, LeadingVersion]
) -> dict:
    track = session.get(Track, rec.track_id)
    analysis = fullmix_analysis(session, rec.track_id)
    version = leading.get(rec.track_id)
    original = session.get(Track, version.of_track_id) if version else None
    return {
        "track_id": rec.track_id,
        "stem_file_id": rec.stem_file_id,
        "track_title": track.title if track else None,
        "track_artist": track.artist if track else None,
        "duration_seconds": track.duration_seconds if track else None,
        "bpm": float(analysis.bpm) if analysis and analysis.bpm is not None else None,
        "key_camelot": analysis.key_camelot if analysis else None,
        "floor_energy": (
            int(analysis.floor_energy)
            if analysis and analysis.floor_energy is not None
            else None
        ),
        "score": rec.score,
        "score_breakdown": rec.score_breakdown,
        "reasons": rec.reasons,
        "version_of": version.of_track_id if version else None,
        "version_of_title": original.title if original else None,
        "version_label": version.label if version else None,
    }


def versions_leading_a_column(
    session: Session, column: str, of_track_ids: list[int], *, skip: list[int], k: int
) -> dict[int, LeadingVersion]:
    if column != MIX_ROLE:
        return {}
    return leading_versions(
        version_index(session), of_track_ids, skip=skip, limit=max(1, k // 2)
    )


__all__ = [
    "column_rec_out",
    "fullmix_analysis",
    "get_bridge",
    "get_session",
    "get_settings",
    "track_tag_values",
    "track_to_out",
    "versions_leading_a_column",
    "Depends",
]
