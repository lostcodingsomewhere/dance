"""
SQLAlchemy database models for Dance DJ Pipeline.

Schema overview:
- ``tracks`` is the source-of-truth for audio files (identified by SHA256 hash).
- ``audio_analysis`` is the unified analysis table for both the full mix
  (``stem_file_id IS NULL``) and per-stem analyses (``stem_file_id`` set).
- ``stem_files`` stores per-kind separated stem audio.
- ``regions`` replaces the old ``cue_points`` table and represents both points
  (cues) and ranges (loops, fades, sections, stem-solo windows).
- ``tags`` + ``track_tags`` is the extensible tagging M:N system.
- ``track_embeddings`` holds CLAP (or other) embeddings for the full track or
  for individual stems.
- ``track_edges`` is the recommendation graph between tracks.
- ``sessions`` + ``session_plays`` records DJ set history.
- ``beats`` and ``phrases`` are kept for downstream beat-grid utilities.
"""

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


def now_utc() -> datetime:
    """Return current UTC time as a timezone-aware datetime.

    Centralized so we can swap to ``datetime.UTC`` once we drop Python <3.11.
    """
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrackState(str, Enum):
    """Processing state of a track."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    SEPARATING = "separating"
    SEPARATED = "separated"
    ANALYZING_STEMS = "analyzing_stems"
    STEMS_ANALYZED = "stems_analyzed"
    DETECTING_REGIONS = "detecting_regions"
    REGIONS_DETECTED = "regions_detected"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    COMPLETE = "complete"
    ERROR = "error"


class StemKind(str, Enum):
    """Kind of stem produced by source separation."""

    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    OTHER = "other"


class TagKind(str, Enum):
    """Kind/namespace of a tag."""

    SUBGENRE = "subgenre"
    MOOD = "mood"
    ELEMENT = "element"
    DJ_NOTE = "dj_note"
    CUSTOM = "custom"


class TagSource(str, Enum):
    """Where a (track, tag) association came from."""

    LLM = "llm"
    MANUAL = "manual"
    INFERRED = "inferred"


class RegionType(str, Enum):
    """Type of marked region within a track or stem."""

    CUE = "cue"
    LOOP = "loop"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SECTION = "section"
    STEM_SOLO = "stem_solo"


class SectionLabel(str, Enum):
    """Musical section label for SECTION regions."""

    INTRO = "intro"
    BUILDUP = "buildup"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    BRIDGE = "bridge"
    OUTRO = "outro"
    VERSE = "verse"
    CHORUS = "chorus"
    OTHER = "other"


class RegionSource(str, Enum):
    """Where a region annotation came from."""

    AUTO = "auto"
    MANUAL = "manual"
    LLM = "llm"
    IMPORTED = "imported"


class EdgeKind(str, Enum):
    """Kind of recommendation edge between tracks."""

    HARMONIC_COMPAT = "harmonic_compat"
    TEMPO_COMPAT = "tempo_compat"
    TAG_OVERLAP = "tag_overlap"
    EMBEDDING_NEIGHBOR = "embedding_neighbor"
    MANUALLY_PAIRED = "manually_paired"
    PLAYLIST_NEIGHBOR = "playlist_neighbor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_tag_value(value: str) -> str:
    """Normalize a tag display value for uniqueness checks.

    Lower-cases, strips, and collapses internal whitespace runs into a single
    space. The result is what should be stored in ``Tag.normalized_value``.
    """

    return _WHITESPACE_RE.sub(" ", value.strip().lower())


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class Track(Base):
    """Core track table — identified by content hash (SHA256)."""

    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    spotify_id = Column(String(64), unique=True, nullable=True, index=True)

    file_path = Column(Text, nullable=False)
    file_name = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=True)

    title = Column(String(512), nullable=True)
    artist = Column(String(512), nullable=True)
    album = Column(String(512), nullable=True)
    year = Column(Integer, nullable=True)

    state = Column(
        String(32), default=TrackState.PENDING.value, nullable=False, index=True
    )
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=now_utc, nullable=False)
    updated_at = Column(
        DateTime, default=now_utc, onupdate=now_utc, nullable=False
    )
    analyzed_at = Column(DateTime, nullable=True)

    # Relationships
    analysis = relationship(
        "AudioAnalysis",
        back_populates="track",
        cascade="all, delete-orphan",
    )
    stem_files = relationship(
        "StemFile", back_populates="track", cascade="all, delete-orphan"
    )
    regions = relationship(
        "Region", back_populates="track", cascade="all, delete-orphan"
    )
    embeddings = relationship(
        "TrackEmbedding", back_populates="track", cascade="all, delete-orphan"
    )
    tags = relationship(
        "TrackTag", back_populates="track", cascade="all, delete-orphan"
    )
    beats = relationship("Beat", cascade="all, delete-orphan")
    phrases = relationship("Phrase", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<Track(id={self.id}, artist={self.artist!r}, "
            f"title={self.title!r}, state={self.state!r})>"
        )


# ---------------------------------------------------------------------------
# Stem files
# ---------------------------------------------------------------------------


class StemFile(Base):
    """One audio stem file (drums/bass/vocals/other) for a track."""

    __tablename__ = "stem_files"
    __table_args__ = (
        UniqueConstraint("track_id", "kind", name="uq_stem_files_track_kind"),
        Index("ix_stem_files_track_id", "track_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(
        Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    kind = Column(String(16), nullable=False)
    path = Column(Text, nullable=False)
    model_used = Column(String(64), default="htdemucs_ft")
    separation_quality = Column(Float, nullable=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    track = relationship("Track", back_populates="stem_files")
    analysis = relationship(
        "AudioAnalysis",
        back_populates="stem_file",
        cascade="all, delete-orphan",
    )
    regions = relationship(
        "Region", back_populates="stem_file", cascade="all, delete-orphan"
    )
    embeddings = relationship(
        "TrackEmbedding", back_populates="stem_file", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<StemFile(id={self.id}, track_id={self.track_id}, kind={self.kind!r})>"


# ---------------------------------------------------------------------------
# Audio analysis (unified: full mix + per-stem)
# ---------------------------------------------------------------------------


class AudioAnalysis(Base):
    """Audio analysis row, one per (track, stem).

    ``stem_file_id IS NULL`` indicates analysis of the full mix.
    Per-(track, stem) uniqueness is enforced by two partial unique indexes
    (created in :func:`init_db`) since SQLite treats NULL as distinct in a
    plain UNIQUE constraint.
    """

    __tablename__ = "audio_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(
        Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    stem_file_id = Column(
        Integer,
        ForeignKey("stem_files.id", ondelete="CASCADE"),
        nullable=True,
    )

    # Tempo & rhythm
    bpm = Column(Float, nullable=True)
    bpm_confidence = Column(Float, nullable=True)

    # Key
    key_camelot = Column(String(4), nullable=True, index=True)
    key_standard = Column(String(8), nullable=True)
    key_confidence = Column(Float, nullable=True)

    # Energy / spectral
    energy_overall = Column(Float, nullable=True)
    energy_peak = Column(Float, nullable=True)
    floor_energy = Column(Integer, nullable=True)
    brightness = Column(Float, nullable=True)
    warmth = Column(Float, nullable=True)
    danceability = Column(Float, nullable=True)

    # Stem-meaningful metrics
    presence_ratio = Column(Float, nullable=True)
    rms_curve = Column(LargeBinary, nullable=True)
    rms_curve_hop_ms = Column(Integer, nullable=True)
    rms_curve_length = Column(Integer, nullable=True)
    presence_intervals = Column(Text, nullable=True)
    dominant_pitch_camelot = Column(String(4), nullable=True)
    dominant_pitch_confidence = Column(Float, nullable=True)
    vocal_present = Column(Boolean, nullable=True)
    kick_density = Column(Float, nullable=True)

    analyzed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    track = relationship("Track", back_populates="analysis")
    stem_file = relationship("StemFile", back_populates="analysis")

    def __repr__(self) -> str:
        scope = "fullmix" if self.stem_file_id is None else f"stem={self.stem_file_id}"
        return (
            f"<AudioAnalysis(id={self.id}, track_id={self.track_id}, "
            f"{scope}, bpm={self.bpm}, key={self.key_camelot})>"
        )


# Backwards-compat alias for code paths that still import ``Analysis``.
Analysis = AudioAnalysis


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class Tag(Base):
    """A single tag value within a kind/namespace."""

    __tablename__ = "tags"
    __table_args__ = (
        UniqueConstraint("kind", "normalized_value", name="uq_tags_kind_norm"),
        Index("ix_tags_normalized_value", "normalized_value"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(32), nullable=False)
    value = Column(Text, nullable=False)
    normalized_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, kind={self.kind!r}, value={self.value!r})>"


class TrackTag(Base):
    """M:N join row between tracks and tags."""

    __tablename__ = "track_tags"
    __table_args__ = (Index("ix_track_tags_tag_id", "tag_id"),)

    track_id = Column(
        Integer,
        ForeignKey("tracks.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    tag_id = Column(
        Integer,
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    source = Column(String(16), primary_key=True, nullable=False)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    track = relationship("Track", back_populates="tags")
    tag = relationship("Tag")

    def __repr__(self) -> str:
        return (
            f"<TrackTag(track_id={self.track_id}, tag_id={self.tag_id}, "
            f"source={self.source!r})>"
        )


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


class Region(Base):
    """Cues, loops, fades, sections, and stem-solo windows.

    A Region with ``stem_file_id IS NULL`` applies to the whole track; when
    set, it applies to that specific stem (e.g. a stem-solo or per-stem cue).
    """

    __tablename__ = "regions"
    __table_args__ = (
        Index("ix_regions_track_id_region_type", "track_id", "region_type"),
        Index(
            "ix_regions_stem_file_id_region_type",
            "stem_file_id",
            "region_type",
            sqlite_where=text("stem_file_id IS NOT NULL"),
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(
        Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    stem_file_id = Column(
        Integer,
        ForeignKey("stem_files.id", ondelete="CASCADE"),
        nullable=True,
    )

    position_ms = Column(Integer, nullable=False)
    length_ms = Column(Integer, nullable=True)
    region_type = Column(String(16), nullable=False)
    section_label = Column(String(16), nullable=True)
    # Musical length in bars (e.g., 4/8/16 for loops). Named ``length_bars`` to
    # avoid collision with ``Phrase.bar_count`` which has different semantics
    # (count of bars *in* the phrase, not its musical-loop length).
    length_bars = Column(Integer, nullable=True)
    name = Column(String(64), nullable=True)
    color = Column(String(8), nullable=True)
    confidence = Column(Float, nullable=True)
    source = Column(String(16), nullable=False, default=RegionSource.AUTO.value)
    snapped_to = Column(String(8), nullable=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    track = relationship("Track", back_populates="regions")
    stem_file = relationship("StemFile", back_populates="regions")

    def __repr__(self) -> str:
        return (
            f"<Region(id={self.id}, track_id={self.track_id}, "
            f"type={self.region_type!r}, pos={self.position_ms})>"
        )


# ---------------------------------------------------------------------------
# Track embeddings
# ---------------------------------------------------------------------------


class TrackEmbedding(Base):
    """A single CLAP-style embedding for a track or stem."""

    __tablename__ = "track_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "track_id",
            "stem_file_id",
            "model",
            "model_version",
            name="uq_track_embeddings_track_stem_model_version",
        ),
        Index("ix_track_embeddings_model", "model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(
        Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    stem_file_id = Column(
        Integer,
        ForeignKey("stem_files.id", ondelete="CASCADE"),
        nullable=True,
    )
    model = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=True)
    dim = Column(Integer, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    track = relationship("Track", back_populates="embeddings")
    stem_file = relationship("StemFile", back_populates="embeddings")

    def __repr__(self) -> str:
        return (
            f"<TrackEmbedding(id={self.id}, track_id={self.track_id}, "
            f"model={self.model!r}, dim={self.dim})>"
        )


# ---------------------------------------------------------------------------
# Track edges (recommendation graph)
# ---------------------------------------------------------------------------


class TrackEdge(Base):
    """A directed edge between two tracks for recommendation purposes."""

    __tablename__ = "track_edges"
    __table_args__ = (
        CheckConstraint(
            "from_track_id != to_track_id", name="ck_track_edges_no_self_loop"
        ),
        UniqueConstraint(
            "from_track_id",
            "to_track_id",
            "kind",
            name="uq_track_edges_from_to_kind",
        ),
        Index(
            "ix_track_edges_from_kind_weight", "from_track_id", "kind", "weight"
        ),
        Index("ix_track_edges_to_kind_weight", "to_track_id", "kind", "weight"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_track_id = Column(
        Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    to_track_id = Column(
        Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False
    )
    kind = Column(String(32), nullable=False)
    weight = Column(Float, nullable=False)
    # JSON blob for debugging/diagnostic context only (e.g., ``{"bpm_delta": 2.5}``).
    # NOT for filtering or querying — if a field needs to be filtered on,
    # promote it to a real column instead.
    meta = Column(Text, nullable=True)
    computed_at = Column(DateTime, nullable=False, default=now_utc)

    def __repr__(self) -> str:
        return (
            f"<TrackEdge(from={self.from_track_id}, to={self.to_track_id}, "
            f"kind={self.kind!r}, weight={self.weight})>"
        )


# ---------------------------------------------------------------------------
# Sessions / set history
# ---------------------------------------------------------------------------


class DjSession(Base):
    """A DJ set / practice session."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    plays = relationship(
        "SessionPlay", back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DjSession(id={self.id}, name={self.name!r})>"


class SessionPlay(Base):
    """A single track playback within a session."""

    __tablename__ = "session_plays"
    __table_args__ = (
        Index(
            "ix_session_plays_session_position",
            "session_id",
            "position_in_set",
        ),
        Index("ix_session_plays_track_played", "track_id", "played_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    played_at = Column(DateTime, nullable=False)
    position_in_set = Column(Integer, nullable=False)
    energy_at_play = Column(Integer, nullable=True)
    transition_type = Column(String(16), nullable=True)
    duration_played_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    session = relationship("DjSession", back_populates="plays")
    track = relationship("Track")

    def __repr__(self) -> str:
        return (
            f"<SessionPlay(session_id={self.session_id}, "
            f"position={self.position_in_set}, track_id={self.track_id})>"
        )


# ---------------------------------------------------------------------------
# Beats and phrases (kept from the original schema)
# ---------------------------------------------------------------------------


class Beat(Base):
    """Beat grid for phrase snapping."""

    __tablename__ = "beats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(
        Integer,
        ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    position_ms = Column(Integer, nullable=False)
    beat_number = Column(Integer, nullable=True)
    bar_number = Column(Integer, nullable=True)
    downbeat = Column(Boolean, default=False)

    def __repr__(self) -> str:
        return (
            f"<Beat(track_id={self.track_id}, bar={self.bar_number}, "
            f"beat={self.beat_number})>"
        )


class Phrase(Base):
    """Detected musical phrases (4/8/16/32 bars)."""

    __tablename__ = "phrases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(
        Integer,
        ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)
    bar_count = Column(Integer, nullable=True)
    phrase_type = Column(String(32), nullable=True)
    energy_level = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Phrase(track_id={self.track_id}, type={self.phrase_type!r}, "
            f"bars={self.bar_count})>"
        )


# ---------------------------------------------------------------------------
# Engine / session management
# ---------------------------------------------------------------------------


_engine = None
_SessionLocal: Optional[sessionmaker] = None


def _attach_sqlite_pragmas(engine) -> None:
    """Enable foreign-key support on every SQLite connection."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine(database_url: str):
    """Get or create the database engine for ``database_url``."""

    global _engine
    if _engine is None:
        _engine = create_engine(
            database_url,
            echo=False,
            connect_args={"check_same_thread": False}
            if "sqlite" in database_url
            else {},
        )
        if "sqlite" in database_url:
            _attach_sqlite_pragmas(_engine)
    return _engine


def get_session_factory(database_url: str) -> sessionmaker:
    """Get the (cached) SQLAlchemy session factory."""

    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def _create_partial_unique_indexes(engine) -> None:
    """Create the two partial unique indexes for ``audio_analysis``.

    SQLAlchemy can't model ``UNIQUE ... WHERE`` portably, so we issue the raw
    DDL ourselves. ``IF NOT EXISTS`` keeps the call idempotent.
    """

    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_audio_analysis_track_fullmix "
                "ON audio_analysis(track_id) WHERE stem_file_id IS NULL"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_audio_analysis_stem "
                "ON audio_analysis(stem_file_id) WHERE stem_file_id IS NOT NULL"
            )
        )


def init_db(database_url: str) -> None:
    """Initialize the database, creating all tables and partial indexes."""

    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    _create_partial_unique_indexes(engine)


def get_session(database_url: str) -> Session:
    """Return a new SQLAlchemy session bound to the configured engine."""

    SessionLocal = get_session_factory(database_url)
    return SessionLocal()


def _reset_engine_for_tests() -> None:
    """Drop the cached engine/session factory (test helper)."""

    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
