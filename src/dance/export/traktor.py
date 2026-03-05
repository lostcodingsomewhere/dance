"""
Traktor NML collection export.

Writes analyzed track data to Traktor's collection.nml file,
including BPM, key, cue points, and energy tags in the comment field.
"""

import logging
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import Analysis, CuePoint, Track, TrackState, CUE_COLORS

logger = logging.getLogger(__name__)


class TraktorExporter:
    """
    Exports analyzed tracks to Traktor NML collection.

    Traktor stores:
    - Track metadata (BPM, key) in TEMPO and MUSICAL_KEY elements
    - Cue points in CUE_V2 elements
    - Custom data in INFO/COMMENT field

    We use the COMMENT field to store energy tags for searching.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection_path = settings.traktor_collection_path
        self.backup_dir = settings.data_dir / "traktor_backups"

    def _backup_collection(self) -> Optional[Path]:
        """Create timestamped backup of collection.nml."""
        if not self.collection_path or not self.collection_path.exists():
            return None

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"collection_{timestamp}.nml"

        shutil.copy2(self.collection_path, backup_path)
        logger.info(f"Backed up collection to: {backup_path}")

        return backup_path

    def _load_collection(self) -> ET.ElementTree:
        """Load and parse the Traktor collection.nml file."""
        if not self.collection_path:
            raise ValueError("Traktor collection path not configured")

        if not self.collection_path.exists():
            raise FileNotFoundError(f"Collection not found: {self.collection_path}")

        return ET.parse(self.collection_path)

    def _find_or_create_entry(
        self,
        collection_element: ET.Element,
        track: Track,
    ) -> ET.Element:
        """Find existing track entry or create new one."""
        # Traktor uses file:// URLs with encoded paths
        file_path = Path(track.file_path)

        # Find existing entry by file path
        for entry in collection_element.findall(".//ENTRY"):
            location = entry.find("LOCATION")
            if location is not None:
                # Check if paths match (handle various path formats)
                entry_file = location.get("FILE", "")
                entry_dir = location.get("DIR", "")
                entry_volume = location.get("VOLUME", "")

                # Reconstruct full path
                if entry_dir and entry_file:
                    # Traktor uses /:/ as separator on Mac
                    entry_path = entry_dir.replace("/:/", "/") + entry_file
                    if entry_path == str(file_path) or entry_file == file_path.name:
                        return entry

        # Create new entry
        entry = ET.SubElement(collection_element, "ENTRY")
        entry.set("TITLE", track.title or file_path.stem)
        entry.set("ARTIST", track.artist or "Unknown")

        # Add LOCATION element
        location = ET.SubElement(entry, "LOCATION")
        location.set("DIR", self._format_traktor_dir(file_path.parent))
        location.set("FILE", file_path.name)
        location.set("VOLUME", self._get_volume_name(file_path))
        location.set("VOLUMEID", "")  # Traktor will fill this

        # Add INFO element for metadata
        info = ET.SubElement(entry, "INFO")
        info.set("PLAYTIME", str(int(track.duration_seconds or 0)))
        info.set("IMPORT_DATE", datetime.now().strftime("%Y/%m/%d"))

        return entry

    def _format_traktor_dir(self, directory: Path) -> str:
        """Format directory path for Traktor's DIR attribute."""
        # Traktor uses /:/ as separator on Mac
        path_str = str(directory)
        if path_str.startswith("/"):
            # Mac/Linux path - convert to Traktor format
            parts = path_str.split("/")
            return "/:/" + "/:/".join(p for p in parts if p) + "/:/"
        return path_str + "/"

    def _get_volume_name(self, file_path: Path) -> str:
        """Get volume name for Traktor LOCATION."""
        # On Mac, root volume is typically the disk name
        # For simplicity, use the first path component
        parts = file_path.parts
        if len(parts) > 1 and parts[0] == "/":
            return parts[1] if parts[1] != "Users" else "Macintosh HD"
        return "Macintosh HD"

    def _update_entry_metadata(
        self,
        entry: ET.Element,
        track: Track,
        analysis: Analysis,
    ) -> None:
        """Update track entry with analysis metadata."""
        # Update TEMPO element
        tempo = entry.find("TEMPO")
        if tempo is None:
            tempo = ET.SubElement(entry, "TEMPO")
        tempo.set("BPM", f"{analysis.bpm:.6f}")
        tempo.set("BPM_QUALITY", "100" if analysis.bpm_confidence and analysis.bpm_confidence > 0.8 else "75")

        # Update MUSICAL_KEY element
        if analysis.key_camelot:
            musical_key = entry.find("MUSICAL_KEY")
            if musical_key is None:
                musical_key = ET.SubElement(entry, "MUSICAL_KEY")
            musical_key.set("VALUE", self._camelot_to_traktor_key(analysis.key_camelot))

        # Update INFO/COMMENT with energy tags
        info = entry.find("INFO")
        if info is None:
            info = ET.SubElement(entry, "INFO")

        comment = self._build_comment(analysis)
        info.set("COMMENT", comment)

        # Update entry-level attributes
        entry.set("TITLE", track.title or entry.get("TITLE", ""))
        entry.set("ARTIST", track.artist or entry.get("ARTIST", ""))

    def _build_comment(self, analysis: Analysis) -> str:
        """Build comment field with energy tags for searching in Traktor.

        If LLM data is available, includes rich subgenre, mood, and DJ notes.
        Format: [Subgenre] mood1, mood2 | DJ notes | [E#] BPM Key
        """
        import json

        parts = []

        # LLM subgenre (if available)
        if analysis.llm_subgenre:
            parts.append(f"[{analysis.llm_subgenre.title()}]")

        # LLM mood tags (if available)
        if analysis.llm_mood_tags:
            try:
                mood_tags = json.loads(analysis.llm_mood_tags)
                if mood_tags:
                    parts.append(", ".join(mood_tags[:3]))
            except json.JSONDecodeError:
                pass
        else:
            # Fallback to computed mood
            mood_parts = []
            if analysis.mood_dark is not None:
                if analysis.mood_dark > 0.65:
                    mood_parts.append("dark")
                elif analysis.mood_dark < 0.35:
                    mood_parts.append("light")

            if analysis.mood_aggressive is not None:
                if analysis.mood_aggressive > 0.7:
                    mood_parts.append("driving")
                elif analysis.mood_aggressive < 0.3:
                    mood_parts.append("chill")

            if mood_parts:
                parts.append(", ".join(mood_parts))

        # LLM DJ notes (if available)
        if analysis.llm_dj_notes:
            # Truncate if too long
            notes = analysis.llm_dj_notes[:50]
            if len(analysis.llm_dj_notes) > 50:
                notes += "..."
            parts.append(f"| {notes}")

        # Technical info (always include)
        tech_parts = []
        if analysis.floor_energy:
            tech_parts.append(f"E{analysis.floor_energy}")
        if analysis.bpm:
            tech_parts.append(f"{int(round(analysis.bpm))}bpm")
        if analysis.key_camelot:
            tech_parts.append(analysis.key_camelot)

        if tech_parts:
            parts.append(f"| {' '.join(tech_parts)}")

        return " ".join(parts)

    def _camelot_to_traktor_key(self, camelot: str) -> str:
        """Convert Camelot notation to Traktor's key format."""
        # Traktor uses Open Key notation which is similar to Camelot
        # but with 'd' for major and 'm' for minor
        if not camelot or len(camelot) < 2:
            return "0"

        number = camelot[:-1]
        letter = camelot[-1].upper()

        # Traktor key values are 0-23 (0 = 1d, 1 = 1m, etc.)
        try:
            num = int(number) - 1  # Convert 1-12 to 0-11
            if letter == "B":  # Major
                return str(num * 2)
            else:  # Minor
                return str(num * 2 + 1)
        except ValueError:
            return "0"

    def _update_cue_points(
        self,
        entry: ET.Element,
        cue_points: list[CuePoint],
        analysis: Optional[Analysis] = None,
    ) -> None:
        """Update track entry with cue points.

        If LLM cue contexts are available, uses them for richer cue point names.
        """
        import json

        # Parse LLM cue contexts if available
        llm_contexts = {}
        if analysis and analysis.llm_cue_contexts:
            try:
                llm_contexts = json.loads(analysis.llm_cue_contexts)
            except json.JSONDecodeError:
                pass

        # Map cue types to LLM context keys
        CUE_TYPE_TO_CONTEXT = {
            "intro": "intro",
            "phrase_1": "buildup_1",
            "buildup": "buildup_1",
            "drop": "drop_1",
            "breakdown": "breakdown",
            "drop_2": "drop_2",
            "outro": "outro",
        }

        # Remove existing auto-generated cue points (keep manual ones)
        # We identify auto-generated by name pattern
        auto_names = ["Intro", "Phrase 1", "Build", "Drop 1", "Breakdown", "Drop 2", "Outro"]
        for cue in entry.findall("CUE_V2"):
            name = cue.get("NAME", "")
            # Keep cues that don't match our naming pattern
            # Also remove cues with LLM-style names (contain " - ")
            if name in auto_names or " - " in name:
                entry.remove(cue)

        # Add new cue points
        for cue in cue_points:
            cue_elem = ET.SubElement(entry, "CUE_V2")

            # Build cue name: prefer LLM context, fall back to basic name
            base_name = cue.name or cue.cue_type.replace("_", " ").title()
            context_key = CUE_TYPE_TO_CONTEXT.get(cue.cue_type.lower(), "")
            llm_context = llm_contexts.get(context_key, "")

            if llm_context:
                # Format: "Drop 1 - Big acid synth" (truncate context to fit)
                max_context_len = 20
                if len(llm_context) > max_context_len:
                    llm_context = llm_context[:max_context_len-3] + "..."
                cue_name = f"{base_name} - {llm_context}"
            else:
                cue_name = base_name

            cue_elem.set("NAME", cue_name)
            cue_elem.set("DISPL_ORDER", str(cue.cue_index or 0))
            cue_elem.set("TYPE", "0")  # 0 = cue point, 4 = loop
            cue_elem.set("START", str(cue.position_ms))
            cue_elem.set("LEN", "0")
            cue_elem.set("REPEATS", "-1")
            cue_elem.set("HOTCUE", str(cue.cue_index or 0))

    def export_track(
        self,
        session: Session,
        track: Track,
        collection_tree: ET.ElementTree,
    ) -> bool:
        """
        Export a single track to Traktor collection.

        Args:
            session: Database session.
            track: Track to export.
            collection_tree: Parsed collection.nml.

        Returns:
            True if export succeeded.
        """
        try:
            analysis = session.query(Analysis).filter_by(track_id=track.id).first()
            if not analysis:
                logger.warning(f"No analysis for track: {track.title}")
                return False

            cue_points = (
                session.query(CuePoint)
                .filter_by(track_id=track.id)
                .order_by(CuePoint.cue_index)
                .all()
            )

            # Find collection element
            root = collection_tree.getroot()
            collection = root.find("COLLECTION")
            if collection is None:
                collection = ET.SubElement(root, "COLLECTION")

            # Find or create entry
            entry = self._find_or_create_entry(collection, track)

            # Update metadata
            self._update_entry_metadata(entry, track, analysis)

            # Update cue points (pass analysis for LLM context)
            if cue_points:
                self._update_cue_points(entry, cue_points, analysis)

            # Mark as exported
            track.exported_at = datetime.utcnow()

            return True

        except Exception as e:
            logger.error(f"Failed to export track {track.title}: {e}")
            return False

    def export_all(self, session: Session) -> dict:
        """
        Export all completed tracks to Traktor.

        Returns:
            Dict with counts: {'exported': N, 'skipped': N, 'failed': N}
        """
        if not self.collection_path:
            raise ValueError(
                "Traktor collection path not configured. "
                "Set DANCE_TRAKTOR_COLLECTION_PATH or install Traktor."
            )

        # Backup first
        self._backup_collection()

        # Load collection
        collection_tree = self._load_collection()

        # Get all tracks ready for export
        tracks = (
            session.query(Track)
            .filter(Track.state.in_([
                TrackState.ANALYZED.value,
                TrackState.SEPARATED.value,
                TrackState.LLM_AUGMENTED.value,
                TrackState.COMPLETE.value,
            ]))
            .all()
        )

        results = {"exported": 0, "skipped": 0, "failed": 0}

        for track in tracks:
            if self.export_track(session, track, collection_tree):
                results["exported"] += 1
            else:
                results["failed"] += 1

        # Save collection
        if results["exported"] > 0:
            collection_tree.write(
                self.collection_path,
                encoding="utf-8",
                xml_declaration=True,
            )
            session.commit()
            logger.info(f"Exported {results['exported']} tracks to Traktor")

        return results


def export_to_traktor(
    session: Session,
    settings: Optional[Settings] = None,
) -> dict:
    """
    Convenience function to export all tracks to Traktor.

    Args:
        session: Database session.
        settings: Settings to use.

    Returns:
        Dict with export statistics.
    """
    if settings is None:
        from dance.config import get_settings
        settings = get_settings()

    exporter = TraktorExporter(settings)
    return exporter.export_all(session)
