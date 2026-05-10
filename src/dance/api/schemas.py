"""Pydantic response models for the Dance API.

These are the STABLE contract between backend and frontend. Each shape has
exactly one model; endpoint-specific variants are avoided unless they truly
differ. ``from_attributes=True`` lets us pass SQLAlchemy rows directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class AnalysisOut(_Base):
    """Full-mix analysis fields surfaced to the UI."""

    bpm: float | None = None
    key_camelot: str | None = None
    key_standard: str | None = None
    floor_energy: int | None = None
    energy_overall: float | None = None
    brightness: float | None = None
    warmth: float | None = None
    danceability: float | None = None


class StemAnalysisOut(_Base):
    """Per-stem analysis subset relevant to the UI."""

    bpm: float | None = None
    energy_overall: float | None = None
    floor_energy: int | None = None
    presence_ratio: float | None = None
    vocal_present: bool | None = None
    kick_density: float | None = None
    dominant_pitch_camelot: str | None = None


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------


class StemFileOut(_Base):
    id: int
    kind: str
    path: str
    analysis: StemAnalysisOut | None = None


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


class RegionOut(_Base):
    id: int
    position_ms: int
    length_ms: int | None = None
    region_type: str
    section_label: str | None = None
    length_bars: int | None = None
    name: str | None = None
    color: str | None = None
    confidence: float | None = None
    source: str
    stem_file_id: int | None = None


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class TrackOut(_Base):
    id: int
    file_path: str
    title: str | None = None
    artist: str | None = None
    duration_seconds: float | None = None
    state: str
    analysis: AnalysisOut | None = None
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class RecommendationOut(_Base):
    track_id: int
    score: float
    reasons: list[dict[str, Any]] = Field(default_factory=list)
    title: str | None = None
    artist: str | None = None
    bpm: float | None = None
    key_camelot: str | None = None
    floor_energy: int | None = None


class RecommendRequest(BaseModel):
    seeds: list[int]
    k: int = 10
    kinds: list[str] | None = None
    weights: dict[str, float] | None = None
    exclude: list[int] | None = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class SessionPlayOut(_Base):
    track_id: int
    played_at: datetime
    position_in_set: int
    energy_at_play: int | None = None
    transition_type: str | None = None
    title: str | None = None
    artist: str | None = None


class SessionOut(_Base):
    id: int
    name: str | None = None
    notes: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    plays: list[SessionPlayOut] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    name: str | None = None
    notes: str | None = None


class SessionPlayCreateRequest(BaseModel):
    track_id: int
    transition_type: str | None = None
    duration_played_ms: int | None = None


# ---------------------------------------------------------------------------
# Ableton
# ---------------------------------------------------------------------------


class AbletonStateOut(BaseModel):
    tempo: float | None = None
    is_playing: bool | None = None
    beat: float | None = None
    playing_clips: dict[int, int] = Field(default_factory=dict)
    track_volumes: dict[int, float] = Field(default_factory=dict)


class TempoRequest(BaseModel):
    bpm: float


class FireClipRequest(BaseModel):
    track: int
    scene: int


class VolumeRequest(BaseModel):
    track: int
    volume: float


__all__ = [
    "AbletonStateOut",
    "AnalysisOut",
    "FireClipRequest",
    "RecommendRequest",
    "RecommendationOut",
    "RegionOut",
    "SessionCreateRequest",
    "SessionOut",
    "SessionPlayCreateRequest",
    "SessionPlayOut",
    "StemAnalysisOut",
    "StemFileOut",
    "TempoRequest",
    "TrackOut",
    "VolumeRequest",
]
