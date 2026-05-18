"""Pydantic response models for the Dance API.

These are the STABLE contract between backend and frontend. Each shape has
exactly one model; endpoint-specific variants are avoided unless they truly
differ. ``from_attributes=True`` lets us pass SQLAlchemy rows directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def _as_utc_iso(dt: datetime | None) -> str | None:
    """Tag naive datetimes as UTC before serialization.

    The DB stores ``now_utc()`` values without a tzinfo (SQLAlchemy + SQLite
    drops the offset). Without a marker the JS ``new Date()`` parser treats
    them as *local time* and "Xs ago" calculations get a 7-hour skew on the
    West Coast. Stamping ``Z`` makes the contract unambiguous."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


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
    file_path: str | None = None
    bpm: float | None = None
    key_camelot: str | None = None
    floor_energy: int | None = None


class RecommendRequest(BaseModel):
    seeds: list[int]
    k: int = 10
    kinds: list[str] | None = None
    weights: dict[str, float] | None = None
    exclude: list[int] | None = None


class TextRecommendRequest(BaseModel):
    """Free-text query, ranked via CLAP audio↔text joint embedding."""

    query: str
    k: int = 10
    exclude: list[int] | None = None


class ColumnRecOut(_Base):
    """A single candidate in a per-column rec stream."""

    track_id: int
    # None when the column is "mix" (candidate is a whole track, not a stem).
    stem_file_id: int | None = None
    track_title: str | None = None
    track_artist: str | None = None
    bpm: float | None = None
    key_camelot: str | None = None
    floor_energy: int | None = None
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class ColumnRecsResponse(_Base):
    """Top-K candidates for a single stem column, re-scored against combo."""

    column: str
    combo_size: int
    recs: list[ColumnRecOut] = Field(default_factory=list)


class ColumnRecsRequest(BaseModel):
    """Combo + column to score against."""

    column: str
    combo_stem_ids: list[int] = Field(default_factory=list)
    master_bpm: float | None = None
    k: int = 5
    exclude_track_ids: list[int] = Field(default_factory=list)


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


class LoadTrackRequest(BaseModel):
    """Request body for ``POST /api/v1/ableton/load-track``."""

    track_id: int
    include_stems: bool = True
    # Which scene (row in Live's session view, 0-indexed) to stage the track
    # on. The bridge keeps 5 reusable "Deck" track columns; each song claims
    # one scene below them. None lets the bridge default to scene 0.
    scene_index: int | None = None


class LoadTrackResult(BaseModel):
    """Response from ``POST /api/v1/ableton/load-track``.

    AbletonOSC can't actually *load* the audio file into the clip slot
    (Live's Python API doesn't expose it), so this returns the indices of
    the empty audio tracks we created in Live, plus the scene the user
    should drop the stems onto. The backend also reveals the stems folder
    in Finder to make the drag one motion.
    """

    ok: bool
    scene_index: int
    track_indices: dict[str, int] = Field(default_factory=dict)
    # How many of the 4 stems we managed to auto-load into Live (Live 12.0.5+
    # with our patched AbletonOSC). 0 means nothing landed; 4 is a full load
    # and means the user only has to fire the scene to play the track.
    stems_loaded: int = 0
    message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DeckSceneOut(BaseModel):
    """One staged song in Live's session view, with the track metadata the
    companion needs to render a Scene Map row without a second fetch."""

    scene_index: int
    track_id: int
    title: str | None = None
    artist: str | None = None
    bpm: float | None = None
    key_camelot: str | None = None
    floor_energy: int | None = None


class DeckMapOut(BaseModel):
    """Reply for ``GET /api/v1/ableton/decks``.

    ``columns`` is null until the user has hit Load at least once (no decks
    created in Live yet). ``scenes`` lists every scene the bridge has
    staged with one of our songs.
    """

    columns: dict[str, int] | None = None
    scenes: list[DeckSceneOut] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Live Set export (.als)
# ---------------------------------------------------------------------------


class AlsExportRequest(BaseModel):
    """Body for ``POST /api/v1/tracks/{id}/als``.

    ``out_path``: optional override. If absent, the generator picks a name
    under ``settings.als_output_dir`` based on the track title. If
    provided, the path **must** be inside ``als_output_dir`` — otherwise
    the API returns 403.
    """

    out_path: str | None = None


class AlsExportResult(BaseModel):
    """Response for ``POST /api/v1/tracks/{id}/als``."""

    ok: bool
    out_path: str
    size_bytes: int
    track_count: int
    locator_count: int


# ---------------------------------------------------------------------------
# Pipeline ops
# ---------------------------------------------------------------------------


class PipelineStatusOut(BaseModel):
    """Snapshot of the pipeline state. Mirrors ``dance status`` at the CLI.

    Returned by ``GET /api/v1/pipeline/status``. The UI polls this on a
    short interval (e.g. every 3 s) for a glanceable view of what's left.
    """

    counts: dict[str, int]
    """Number of tracks in each ``TrackState`` (every state always present,
    zero-padded so the UI can render a stable grid)."""

    total: int
    """Sum of counts across all states."""

    in_progress: int | bool
    """True iff any track is in an active stage (analyzing / separating /
    analyzing_stems / detecting_regions / embedding)."""

    errors: int
    """Convenience: tracks currently in the ``error`` state."""

    complete: int
    """Convenience: tracks currently in the ``complete`` state."""

    weighted_progress: float
    """0-100 stage-weighted progress. Each track contributes 0-6 stages of
    progress (analyze, separate, analyze_stems, detect_regions, embed,
    complete). Summed across all tracks and normalized. Always > 0 once
    any track has hit an "*ed" state, even if zero have reached ``complete``."""


class PipelineRecentTrackOut(BaseModel):
    """One row in the "recently changed" pipeline feed.

    Returned by ``GET /api/v1/pipeline/recent``. The UI renders this as a
    scrolling activity list so it's obvious which tracks the dispatcher is
    working on.
    """

    id: int
    title: str | None
    artist: str | None
    state: str
    updated_at: datetime | None
    error_message: str | None

    @field_serializer("updated_at")
    def _ser_updated_at(self, v: datetime | None) -> str | None:
        return _as_utc_iso(v)


# ---------------------------------------------------------------------------
# Pipeline ingest (CSV → yt-dlp)
# ---------------------------------------------------------------------------


class IngestPreviewRequest(BaseModel):
    """Body for ``POST /api/v1/pipeline/ingest/preview``.

    ``csv_text``: full text content of an Exportify CSV export. The server
    parses it, runs dedupe against the existing library, and returns a plan
    the UI can render. **No downloads happen** at this stage — preview only.
    """

    csv_text: str


class IngestPreviewRow(BaseModel):
    artist: str
    title: str
    album: str = ""
    duration_s: int = 0
    target_path: str
    """Where the file *would* land on disk if downloaded."""
    target_exists: bool
    """True iff a file already exists at ``target_path`` (size > 100 KB)."""
    duplicate_of: int | None = None
    """``track_id`` of an existing library track that looks like a duplicate,
    if any. Determined by normalized artist+title fuzzy match."""


class IngestPreviewResponse(BaseModel):
    total_rows: int
    new_rows: list[IngestPreviewRow]
    duplicates: list[IngestPreviewRow]
    """Already in the library (either same file on disk or a fuzzy match in
    the DB). UI shows these grayed-out / checked-off by default."""
    parse_errors: list[str]


class IngestCommitRequest(BaseModel):
    """Body for ``POST /api/v1/pipeline/ingest/commit``.

    Two selection modes:

    1. **Bulk** (legacy): pass ``include_duplicates=True/False`` and the
       server downloads either all new rows, or new + duplicate rows.
       Used by callers that don't want a row-by-row UI.

    2. **Explicit per-row** (UI uses this): pass ``selected_keys`` as a
       list of ``"artist|title"`` strings. The server downloads exactly
       those rows from the CSV. ``include_duplicates`` is ignored when
       ``selected_keys`` is non-empty.

    The ``|`` separator is chosen because it never appears in real
    artist/title fields (they'd be stripped by the filesystem-sanitizer).
    """

    csv_text: str
    include_duplicates: bool = False
    selected_keys: list[str] | None = None


class JobItemOut(BaseModel):
    label: str
    status: str
    message: str = ""


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    revision: int
    counts: dict[str, int]
    total: int
    items: list[JobItemOut]


__all__ = [
    "AbletonStateOut",
    "AlsExportRequest",
    "AlsExportResult",
    "AnalysisOut",
    "ColumnRecOut",
    "ColumnRecsRequest",
    "ColumnRecsResponse",
    "FireClipRequest",
    "IngestCommitRequest",
    "IngestPreviewRequest",
    "IngestPreviewResponse",
    "IngestPreviewRow",
    "JobItemOut",
    "JobOut",
    "LoadTrackRequest",
    "LoadTrackResult",
    "DeckSceneOut",
    "DeckMapOut",
    "PipelineRecentTrackOut",
    "PipelineStatusOut",
    "RecommendRequest",
    "RecommendationOut",
    "TextRecommendRequest",
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
