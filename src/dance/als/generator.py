"""High-level entry point: ``Track`` + DB session → ``.als`` file on disk.

The Generator pulls together everything needed:
- the track's full-mix ``AudioAnalysis`` (for BPM)
- its 4 ``StemFile`` rows
- its track-level ``Region`` rows (cues + sections)
- builds the Live Set XML via :mod:`dance.als.writer`
- gzip-compresses it and writes to disk

The output path is validated against ``settings.als_output_dir`` so the
API endpoint can call us without re-implementing the same check.
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path

from sqlalchemy.orm import Session

from dance.als.markers import regions_to_locators
from dance.als.writer import LiveSetSpec, StemEntry, build_live_set_xml
from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Region,
    StemFile,
    StemKind,
    Track,
    TrackState,
)

logger = logging.getLogger(__name__)


# Strip characters that are problematic in cross-platform filenames.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9 _.\-()]+")


class AlsExportError(Exception):
    """Raised when a track can't be exported as a Live Set (missing data)."""


class AlsOutsideDirError(AlsExportError):
    """``out_path`` is outside the configured ``als_output_dir``."""


def _safe_filename(stem: str) -> str:
    """Make ``stem`` filesystem-safe (no slashes, no funky punctuation)."""
    cleaned = _FILENAME_SAFE.sub("_", stem).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "Untitled"


class AlsGenerator:
    """Produce ``.als`` files for ``Track`` rows in the database."""

    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def default_path_for(self, track: Track) -> Path:
        """Default output path: ``<als_output_dir>/<safe title> - <artist>.als``."""
        title = _safe_filename(track.title or track.file_name or f"Track {track.id}")
        artist = _safe_filename(track.artist or "Unknown")
        fname = f"{title} - {artist}.als"
        return self.settings.als_output_dir / fname

    def _validate_in_output_dir(self, out_path: Path) -> Path:
        """Ensure ``out_path`` resolves to inside ``settings.als_output_dir``."""
        out_resolved = out_path.expanduser().resolve()
        root = self.settings.als_output_dir.expanduser().resolve()
        try:
            out_resolved.relative_to(root)
        except ValueError as exc:
            raise AlsOutsideDirError(
                f"out_path {out_resolved} is outside als_output_dir {root}"
            ) from exc
        return out_resolved

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def write(self, track: Track, out_path: Path | None = None) -> Path:
        """Generate and write a ``.als`` for ``track``.

        Parameters
        ----------
        track : the Track row to export. Must be in state ``complete`` (we
            need its full-mix analysis for BPM).
        out_path : where to write. If ``None``, derived from the track
            title under ``settings.als_output_dir``. Must be inside that
            directory.

        Returns
        -------
        The absolute path written.

        Raises
        ------
        AlsExportError : if the track has no full-mix analysis or no stems.
        AlsOutsideDirError : if ``out_path`` is outside ``als_output_dir``.
        """
        if track.state != TrackState.COMPLETE.value:
            raise AlsExportError(
                f"track {track.id} is in state {track.state!r}, need 'complete'"
            )

        analysis = (
            self.session.query(AudioAnalysis)
            .filter(
                AudioAnalysis.track_id == track.id,
                AudioAnalysis.stem_file_id.is_(None),
            )
            .one_or_none()
        )
        if analysis is None or analysis.bpm is None:
            raise AlsExportError(
                f"track {track.id} has no full-mix analysis with a BPM"
            )

        stems = (
            self.session.query(StemFile)
            .filter(StemFile.track_id == track.id)
            .all()
        )
        if not stems:
            raise AlsExportError(f"track {track.id} has no stems")

        regions = (
            self.session.query(Region)
            .filter(Region.track_id == track.id, Region.stem_file_id.is_(None))
            .order_by(Region.position_ms)
            .all()
        )

        # Build StemEntry list — 4 stems + 1 mix (full-mix audio file).
        stem_entries: list[StemEntry] = []
        valid_kinds = {k.value for k in StemKind}
        for s in stems:
            if s.kind not in valid_kinds:
                logger.warning(
                    "skipping stem with unexpected kind %r on track %s",
                    s.kind,
                    track.id,
                )
                continue
            stem_entries.append(
                StemEntry(
                    kind=s.kind,
                    path=Path(s.path),
                    duration_seconds=track.duration_seconds,
                )
            )
        # Mix reference (full original file).
        stem_entries.append(
            StemEntry(
                kind="mix",
                path=Path(track.file_path),
                duration_seconds=track.duration_seconds,
            )
        )

        locators = regions_to_locators(regions, bpm=float(analysis.bpm))

        spec = LiveSetSpec(
            name=track.title or track.file_name or f"Track {track.id}",
            bpm=float(analysis.bpm),
            stems=stem_entries,
            locators=locators,
        )

        xml_bytes = build_live_set_xml(spec)
        gz_bytes = gzip.compress(xml_bytes, compresslevel=6)

        target = out_path if out_path is not None else self.default_path_for(track)
        target = self._validate_in_output_dir(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gz_bytes)

        logger.info(
            "wrote .als for track %s to %s (%d bytes)", track.id, target, len(gz_bytes)
        )
        return target


__all__ = ["AlsExportError", "AlsGenerator", "AlsOutsideDirError"]
