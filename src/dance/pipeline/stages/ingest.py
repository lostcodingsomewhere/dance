"""
Ingest stage - file scanning and hashing.

Scans directories for audio files, computes content hashes for deduplication,
and extracts basic metadata from file tags.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import mutagen
from sqlalchemy.orm import Session

from dance.core.database import Track, TrackState

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".flac", ".wav", ".aiff", ".m4a", ".ogg", ".opus"}

# Chunk size for hashing
CHUNK_SIZE = 8192


@dataclass
class IngestResult:
    """Result of ingesting a single file."""

    status: str  # 'new', 'updated', 'unchanged', 'error'
    track_id: Optional[int] = None
    error: Optional[str] = None


@dataclass
class ScanResult:
    """Result of scanning a directory."""

    new: int = 0
    updated: int = 0
    unchanged: int = 0
    errors: int = 0
    error_messages: list[str] = None

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class IngestStage:
    """
    Ingests audio files into the database.

    Files are identified by content hash, not path.
    This means:
    - The same file in different locations is recognized as a duplicate
    - Moved/renamed files are updated, not duplicated
    - Different files with the same name are correctly treated as different
    """

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir

    def compute_audio_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of audio content.

        For efficiency, we hash:
        - File size (as string)
        - First 1MB of content
        - Last 1MB of content (if file is large enough)

        This catches 99.9%+ of duplicates without reading entire files.
        """
        hasher = hashlib.sha256()

        file_size = file_path.stat().st_size
        hasher.update(str(file_size).encode())

        with open(file_path, "rb") as f:
            # First 1MB
            first_chunk = f.read(1024 * 1024)
            hasher.update(first_chunk)

            # Last 1MB if file is large enough
            if file_size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, 2)
                last_chunk = f.read()
                hasher.update(last_chunk)

        return hasher.hexdigest()

    def extract_metadata(self, file_path: Path) -> dict:
        """Extract basic metadata from audio file tags."""
        try:
            audio = mutagen.File(file_path, easy=True)
            if audio is None:
                # Fallback for files without tags
                return {
                    "title": file_path.stem,
                    "artist": None,
                    "album": None,
                    "year": None,
                    "duration_seconds": None,
                }

            # Try to get duration
            duration = None
            if hasattr(audio, "info") and audio.info:
                duration = audio.info.length

            return {
                "title": self._get_first(audio, "title") or file_path.stem,
                "artist": self._get_first(audio, "artist"),
                "album": self._get_first(audio, "album"),
                "year": self._parse_year(self._get_first(audio, "date")),
                "duration_seconds": duration,
            }
        except Exception as e:
            logger.warning(f"Could not read metadata from {file_path}: {e}")
            return {
                "title": file_path.stem,
                "artist": None,
                "album": None,
                "year": None,
                "duration_seconds": None,
            }

    def _get_first(self, audio, key: str) -> Optional[str]:
        """Get first value from a potentially list-valued tag."""
        value = audio.get(key)
        if value is None:
            return None
        if isinstance(value, list):
            return value[0] if value else None
        return str(value)

    def _parse_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from various date formats."""
        if not date_str:
            return None
        try:
            # Handle "2023", "2023-01-15", etc.
            return int(date_str[:4])
        except (ValueError, TypeError):
            return None

    def scan_directory(self, directory: Optional[Path] = None, recursive: bool = True) -> Iterator[Path]:
        """
        Find all audio files in directory.

        Args:
            directory: Directory to scan. Uses library_dir if not provided.
            recursive: If True, scan subdirectories too.

        Yields:
            Path to each audio file found.
        """
        scan_dir = directory or self.library_dir

        if not scan_dir.exists():
            logger.warning(f"Directory does not exist: {scan_dir}")
            return

        if recursive:
            for file_path in scan_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                    yield file_path
        else:
            for file_path in scan_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                    yield file_path

    def ingest_file(self, session: Session, file_path: Path) -> IngestResult:
        """
        Ingest a single file into the database.

        Args:
            session: Database session.
            file_path: Path to the audio file.

        Returns:
            IngestResult with status and track_id.
        """
        try:
            # Compute content hash
            file_hash = self.compute_audio_hash(file_path)

            # Check if this content already exists
            existing = session.query(Track).filter_by(file_hash=file_hash).first()

            if existing:
                # Same content - check if file moved
                if existing.file_path != str(file_path):
                    logger.info(f"Track moved: {existing.file_path} -> {file_path}")
                    existing.file_path = str(file_path)
                    existing.file_name = file_path.name
                    session.commit()
                    return IngestResult(status="updated", track_id=existing.id)

                # Unchanged
                return IngestResult(status="unchanged", track_id=existing.id)

            # New file - extract metadata
            metadata = self.extract_metadata(file_path)

            track = Track(
                file_hash=file_hash,
                file_path=str(file_path),
                file_name=file_path.name,
                file_size_bytes=file_path.stat().st_size,
                state=TrackState.PENDING.value,
                **metadata,
            )

            session.add(track)
            session.commit()

            logger.info(f"Ingested new track: {track.artist} - {track.title}")
            return IngestResult(status="new", track_id=track.id)

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            session.rollback()
            return IngestResult(status="error", error=str(e))

    def scan_and_ingest(
        self,
        session: Session,
        directory: Optional[Path] = None,
        recursive: bool = True,
    ) -> ScanResult:
        """
        Scan directory and ingest all audio files.

        Args:
            session: Database session.
            directory: Directory to scan. Uses library_dir if not provided.
            recursive: If True, scan subdirectories.

        Returns:
            ScanResult with counts of new/updated/unchanged/error.
        """
        result = ScanResult()

        for file_path in self.scan_directory(directory, recursive):
            ingest_result = self.ingest_file(session, file_path)

            if ingest_result.status == "new":
                result.new += 1
            elif ingest_result.status == "updated":
                result.updated += 1
            elif ingest_result.status == "unchanged":
                result.unchanged += 1
            elif ingest_result.status == "error":
                result.errors += 1
                if ingest_result.error:
                    result.error_messages.append(f"{file_path}: {ingest_result.error}")

        return result


def ingest_directory(
    session: Session,
    library_dir: Path,
    directory: Optional[Path] = None,
    recursive: bool = True,
) -> ScanResult:
    """
    Convenience function to scan and ingest a directory.

    Args:
        session: Database session.
        library_dir: Base library directory.
        directory: Specific directory to scan (optional).
        recursive: If True, scan subdirectories.

    Returns:
        ScanResult with scan statistics.
    """
    stage = IngestStage(library_dir)
    return stage.scan_and_ingest(session, directory, recursive)
