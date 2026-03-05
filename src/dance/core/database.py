"""
SQLAlchemy database models for Dance DJ Pipeline.

Tracks are identified by content hash (SHA256) for deduplication.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class TrackState(str, Enum):
    """Processing state of a track."""

    PENDING = "pending"  # Downloaded, awaiting processing
    ANALYZING = "analyzing"  # BPM/key/energy analysis in progress
    ANALYZED = "analyzed"  # Analysis complete, awaiting stems
    SEPARATING = "separating"  # Stem separation in progress
    SEPARATED = "separated"  # Stems complete, awaiting LLM augment
    LLM_AUGMENTING = "llm_augmenting"  # LLM analysis in progress
    LLM_AUGMENTED = "llm_augmented"  # LLM complete, awaiting cue detection
    DETECTING_CUES = "detecting_cues"  # Cue point detection in progress
    COMPLETE = "complete"  # Fully processed
    ERROR = "error"  # Processing failed


class CueType(str, Enum):
    """Type of cue point for house/techno mixing."""

    INTRO = "intro"  # Track start, safe mix-in point
    PHRASE_1 = "phrase_1"  # First major phrase boundary
    BUILDUP = "buildup"  # Pre-drop energy rise
    DROP = "drop"  # Main energy peak (don't mix here)
    BREAKDOWN = "breakdown"  # Energy dip, good mix-out opportunity
    DROP_2 = "drop_2"  # Second drop if exists
    OUTRO = "outro"  # Start of outro, mix-out point
    CUSTOM = "custom"  # User-defined


# Cue colors for Traktor (matching the plan)
CUE_COLORS = {
    CueType.INTRO: "#00FF00",  # Green
    CueType.PHRASE_1: "#00FFFF",  # Cyan
    CueType.BUILDUP: "#FFFF00",  # Yellow
    CueType.DROP: "#FF0000",  # Red
    CueType.BREAKDOWN: "#FF00FF",  # Magenta
    CueType.DROP_2: "#FF6600",  # Orange
    CueType.OUTRO: "#0000FF",  # Blue
    CueType.CUSTOM: "#FFFFFF",  # White
}


class Track(Base):
    """
    Core track table - source of truth for all tracks.

    Tracks are identified by content hash, not file path.
    This allows recognizing moved/renamed files as the same content.
    """

    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity (content-addressable)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)

    # Spotify metadata (if downloaded via spotDL)
    spotify_id = Column(String(64), unique=True, nullable=True, index=True)

    # File info
    file_path = Column(Text, nullable=False)
    file_name = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=True)

    # Basic metadata from file tags
    title = Column(String(512), nullable=True)
    artist = Column(String(512), nullable=True)
    album = Column(String(512), nullable=True)
    year = Column(Integer, nullable=True)

    # Processing state
    state = Column(String(32), default=TrackState.PENDING.value, nullable=False, index=True)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    analyzed_at = Column(DateTime, nullable=True)
    exported_at = Column(DateTime, nullable=True)

    # Relationships
    analysis = relationship("Analysis", back_populates="track", uselist=False, cascade="all, delete-orphan")
    beats = relationship("Beat", back_populates="track", cascade="all, delete-orphan")
    phrases = relationship("Phrase", back_populates="track", cascade="all, delete-orphan")
    cue_points = relationship("CuePoint", back_populates="track", cascade="all, delete-orphan")
    stems = relationship("Stems", back_populates="track", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Track(id={self.id}, artist='{self.artist}', title='{self.title}', state='{self.state}')>"


class Analysis(Base):
    """
    Audio analysis results from Essentia.

    Contains BPM, key, energy, and mood information.
    """

    __tablename__ = "analysis"

    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)

    # Tempo & Rhythm
    bpm = Column(Float, nullable=False)
    bpm_confidence = Column(Float, nullable=True)
    beats_count = Column(Integer, nullable=True)
    time_signature = Column(String(8), default="4/4")

    # Key (Camelot notation for DJ utility)
    key_camelot = Column(String(4), nullable=True, index=True)  # e.g., "8A", "11B"
    key_standard = Column(String(8), nullable=True)  # e.g., "Am", "F#m"
    key_confidence = Column(Float, nullable=True)

    # Energy metrics (0-1 scale, normalized)
    energy_overall = Column(Float, nullable=True)  # Average RMS energy
    energy_peak = Column(Float, nullable=True)  # Max energy (usually at drops)
    danceability = Column(Float, nullable=True)  # Essentia danceability model

    # Spectral characteristics
    brightness = Column(Float, nullable=True)  # Higher = brighter/harsher
    warmth = Column(Float, nullable=True)  # Bass energy ratio

    # Mood/character (from Essentia classifiers, 0-1 scale)
    mood_aggressive = Column(Float, nullable=True)  # Chill to aggressive
    mood_electronic = Column(Float, nullable=True)  # Organic to synthetic
    mood_dark = Column(Float, nullable=True)  # Light to dark

    # Computed DJ utility scores
    floor_energy = Column(Integer, nullable=True)  # 1-10 scale for set building
    peak_time_suitability = Column(Float, nullable=True)  # 0-1: opener to peak-time

    # LLM augmentation (from Qwen2-Audio)
    # Rich tagging
    llm_subgenre = Column(String(128), nullable=True)  # e.g., "tech house", "melodic techno"
    llm_mood_tags = Column(Text, nullable=True)  # JSON array: ["dark", "driving", "hypnotic"]
    llm_notable_elements = Column(Text, nullable=True)  # JSON array: ["acid line", "vocal chops"]
    llm_energy_curve = Column(Text, nullable=True)  # Description of energy progression
    llm_dj_notes = Column(Text, nullable=True)  # DJ mixing notes

    # Cue contexts (JSON dict mapping cue type to description)
    llm_cue_contexts = Column(Text, nullable=True)  # {"intro": "Minimal kick", "drop_1": "Big synth"}

    # Quality validation
    llm_bpm_validated = Column(Boolean, nullable=True)  # True if LLM agrees with detected BPM
    llm_bpm_suggestion = Column(Float, nullable=True)  # LLM's suggested BPM if different
    llm_key_validated = Column(Boolean, nullable=True)  # True if LLM agrees with detected key
    llm_key_suggestion = Column(String(8), nullable=True)  # LLM's suggested key if different
    llm_quality_issues = Column(Text, nullable=True)  # JSON array of quality issues
    llm_is_dj_track = Column(Boolean, nullable=True)  # True if suitable for DJ mixing

    # LLM metadata
    llm_model = Column(String(128), nullable=True)  # e.g., "Qwen/Qwen2-Audio-7B-Instruct"
    llm_analyzed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    track = relationship("Track", back_populates="analysis")

    def __repr__(self) -> str:
        return f"<Analysis(track_id={self.track_id}, bpm={self.bpm}, key={self.key_camelot}, energy={self.floor_energy})>"


class Beat(Base):
    """
    Beat grid for phrase snapping.

    Stores individual beat positions with bar/beat numbers.
    """

    __tablename__ = "beats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)

    position_ms = Column(Integer, nullable=False)  # Beat position in milliseconds
    beat_number = Column(Integer, nullable=True)  # Beat within bar (1-4 for 4/4)
    bar_number = Column(Integer, nullable=True)  # Bar number from start
    downbeat = Column(Boolean, default=False)  # True if this is beat 1 of a bar

    # Relationship
    track = relationship("Track", back_populates="beats")

    def __repr__(self) -> str:
        return f"<Beat(track_id={self.track_id}, bar={self.bar_number}, beat={self.beat_number})>"


class Phrase(Base):
    """
    Detected musical phrases.

    House/techno typically uses 8, 16, or 32 bar phrases.
    """

    __tablename__ = "phrases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)

    start_ms = Column(Integer, nullable=False)
    end_ms = Column(Integer, nullable=False)
    bar_count = Column(Integer, nullable=True)  # 4, 8, 16, or 32
    phrase_type = Column(String(32), nullable=True)  # 'intro', 'buildup', 'drop', 'breakdown', 'outro'
    energy_level = Column(Float, nullable=True)  # Average energy in this phrase

    # Relationship
    track = relationship("Track", back_populates="phrases")

    def __repr__(self) -> str:
        return f"<Phrase(track_id={self.track_id}, type='{self.phrase_type}', bars={self.bar_count})>"


class CuePoint(Base):
    """
    Cue points for DJ mixing.

    Both auto-detected and user-defined cue points.
    """

    __tablename__ = "cue_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)

    position_ms = Column(Integer, nullable=False)

    # Cue type for house/techno mixing
    cue_type = Column(String(32), nullable=False)

    # Traktor cue metadata
    cue_index = Column(Integer, nullable=True)  # 0-7 for Traktor hotcues
    color = Column(String(8), nullable=True)  # Hex color
    name = Column(String(64), nullable=True)  # Human-readable label

    # Detection metadata
    confidence = Column(Float, nullable=True)  # Model confidence 0-1
    source = Column(String(16), default="auto")  # 'auto', 'manual', 'adjusted'
    snapped_to_phrase = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    track = relationship("Track", back_populates="cue_points")

    def __repr__(self) -> str:
        return f"<CuePoint(track_id={self.track_id}, type='{self.cue_type}', index={self.cue_index})>"


class Stems(Base):
    """
    Stem separation results from Demucs.

    Stores paths to separated stem files.
    """

    __tablename__ = "stems"

    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)

    # Paths to separated stems (relative to stems directory)
    drums_path = Column(Text, nullable=True)
    bass_path = Column(Text, nullable=True)
    vocals_path = Column(Text, nullable=True)
    other_path = Column(Text, nullable=True)

    # Quality metadata
    model_used = Column(String(64), default="htdemucs_ft")
    separation_quality = Column(Float, nullable=True)  # SDR estimate if available

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    track = relationship("Track", back_populates="stems")

    def __repr__(self) -> str:
        return f"<Stems(track_id={self.track_id}, model='{self.model_used}')>"


class ProcessingLog(Base):
    """
    Processing history for debugging and idempotency.

    Records each processing stage attempt.
    """

    __tablename__ = "processing_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True)

    stage = Column(String(32), nullable=False)  # 'ingest', 'analyze', 'separate', 'detect_cues'
    status = Column(String(16), nullable=False)  # 'started', 'completed', 'failed', 'skipped'
    duration_seconds = Column(Float, nullable=True)
    error_details = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ProcessingLog(track_id={self.track_id}, stage='{self.stage}', status='{self.status}')>"


# Database engine and session management

_engine = None
_SessionLocal = None


def get_engine(database_url: str):
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            database_url,
            echo=False,
            # SQLite-specific settings
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        )
        # Enable foreign keys for SQLite
        if "sqlite" in database_url:
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
    return _engine


def get_session_factory(database_url: str) -> sessionmaker:
    """Get the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def init_db(database_url: str) -> None:
    """Initialize the database, creating all tables."""
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)


def get_session(database_url: str) -> Session:
    """Get a new database session."""
    SessionLocal = get_session_factory(database_url)
    return SessionLocal()
