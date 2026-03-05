"""
Cue point detection stage.

Automatically places cue points at musically significant positions:
- Intro (track start)
- Phrase 1 (first major phrase boundary)
- Buildup (pre-drop)
- Drop (main energy peak)
- Breakdown (energy dip)
- Drop 2 (second drop if exists)
- Outro (mix-out point)

All cues are snapped to phrase boundaries for clean mixing.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from dance.core.database import (
    Analysis,
    Beat,
    CuePoint,
    CueType,
    Phrase,
    Track,
    TrackState,
    CUE_COLORS,
)
from dance.pipeline.utils.beats import (
    detect_beats,
    detect_phrases,
    snap_to_phrase,
)

logger = logging.getLogger(__name__)

# Try to import audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Cue point configuration
CUE_CONFIG = {
    CueType.INTRO: {"index": 0, "name": "Intro", "color": CUE_COLORS[CueType.INTRO]},
    CueType.PHRASE_1: {"index": 1, "name": "Phrase 1", "color": CUE_COLORS[CueType.PHRASE_1]},
    CueType.BUILDUP: {"index": 2, "name": "Build", "color": CUE_COLORS[CueType.BUILDUP]},
    CueType.DROP: {"index": 3, "name": "Drop 1", "color": CUE_COLORS[CueType.DROP]},
    CueType.BREAKDOWN: {"index": 4, "name": "Breakdown", "color": CUE_COLORS[CueType.BREAKDOWN]},
    CueType.DROP_2: {"index": 5, "name": "Drop 2", "color": CUE_COLORS[CueType.DROP_2]},
    CueType.OUTRO: {"index": 6, "name": "Outro", "color": CUE_COLORS[CueType.OUTRO]},
}


class CueDetectionStage:
    """
    Detects and places cue points for house/techno mixing.

    Cue points are placed at phrase boundaries to enable clean mixing.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def detect_cues(self, session: Session, track: Track) -> bool:
        """
        Detect and place cue points for a track.

        Args:
            session: Database session.
            track: Track to analyze.

        Returns:
            True if detection succeeded.
        """
        # Update state
        track.state = TrackState.DETECTING_CUES.value
        session.commit()

        try:
            # Get analysis
            analysis = session.query(Analysis).filter_by(track_id=track.id).first()
            if not analysis:
                raise ValueError("No analysis found - run analysis stage first")

            # Load audio
            file_path = Path(track.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Track file not found: {file_path}")

            logger.info(f"Detecting cues: {track.artist} - {track.title}")

            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            else:
                raise RuntimeError("librosa required for cue detection")

            # Detect beats
            beat_times, _ = detect_beats(audio, sr, analysis.bpm)
            beat_times_ms = [int(t * 1000) for t in beat_times]

            # Store beats in database
            self._store_beats(session, track.id, beat_times_ms, analysis.bpm)

            # Detect phrases
            phrases = detect_phrases(audio, sr, beat_times, analysis.bpm)
            self._store_phrases(session, track.id, phrases)

            # Generate cue points
            cues = self._generate_cues(
                phrases,
                beat_times_ms,
                track.duration_seconds or (len(audio) / sr),
                analysis.bpm,
            )

            # Store cue points
            self._store_cues(session, track.id, cues)

            # Update track state
            track.state = TrackState.COMPLETE.value
            session.commit()

            logger.info(f"Detected {len(cues)} cue points for: {track.title}")
            return True

        except Exception as e:
            logger.error(f"Cue detection failed for {track.title}: {e}")
            track.state = TrackState.ERROR.value
            track.error_message = f"Cue detection failed: {str(e)[:400]}"
            session.commit()
            return False

    def _store_beats(
        self,
        session: Session,
        track_id: int,
        beat_times_ms: list[int],
        bpm: float,
    ) -> None:
        """Store beat grid in database."""
        # Clear existing beats
        session.query(Beat).filter_by(track_id=track_id).delete()

        for i, position_ms in enumerate(beat_times_ms):
            beat_number = (i % 4) + 1  # 1-4 within bar
            bar_number = i // 4 + 1

            beat = Beat(
                track_id=track_id,
                position_ms=position_ms,
                beat_number=beat_number,
                bar_number=bar_number,
                downbeat=(beat_number == 1),
            )
            session.add(beat)

    def _store_phrases(
        self,
        session: Session,
        track_id: int,
        phrases: list[dict],
    ) -> None:
        """Store detected phrases in database."""
        # Clear existing phrases
        session.query(Phrase).filter_by(track_id=track_id).delete()

        for phrase_data in phrases:
            phrase = Phrase(
                track_id=track_id,
                start_ms=phrase_data["start_ms"],
                end_ms=phrase_data["end_ms"],
                bar_count=phrase_data["bar_count"],
                phrase_type=phrase_data["phrase_type"],
                energy_level=phrase_data["energy_level"],
            )
            session.add(phrase)

    def _store_cues(
        self,
        session: Session,
        track_id: int,
        cues: list[dict],
    ) -> None:
        """Store cue points in database."""
        # Clear existing auto-detected cues
        session.query(CuePoint).filter_by(
            track_id=track_id,
            source="auto",
        ).delete()

        for cue_data in cues:
            cue = CuePoint(
                track_id=track_id,
                position_ms=cue_data["position_ms"],
                cue_type=cue_data["type"].value,
                cue_index=cue_data["index"],
                color=cue_data["color"],
                name=cue_data["name"],
                confidence=cue_data.get("confidence", 0.8),
                source="auto",
                snapped_to_phrase=True,
            )
            session.add(cue)

    def _generate_cues(
        self,
        phrases: list[dict],
        beat_times_ms: list[int],
        duration_seconds: float,
        bpm: float,
    ) -> list[dict]:
        """
        Generate cue points from phrase analysis.

        Returns list of cue dicts with position, type, etc.
        """
        cues = []

        # Categorize phrases
        drops = [p for p in phrases if p["phrase_type"] == "drop"]
        breakdowns = [p for p in phrases if p["phrase_type"] == "breakdown"]
        buildups = [p for p in phrases if p["phrase_type"] == "buildup"]
        intros = [p for p in phrases if p["phrase_type"] == "intro"]
        outros = [p for p in phrases if p["phrase_type"] == "outro"]

        # 1. INTRO - first beat
        if beat_times_ms:
            cues.append({
                "type": CueType.INTRO,
                "position_ms": beat_times_ms[0],
                "confidence": 0.95,
                **CUE_CONFIG[CueType.INTRO],
            })

        # 2. PHRASE_1 - first major phrase boundary (usually 16 or 32 bars in)
        if len(phrases) > 1:
            phrase_1 = phrases[1]
            cues.append({
                "type": CueType.PHRASE_1,
                "position_ms": phrase_1["start_ms"],
                "confidence": 0.85,
                **CUE_CONFIG[CueType.PHRASE_1],
            })

        # 3. BUILDUP - before first drop
        if buildups:
            first_buildup = buildups[0]
            cues.append({
                "type": CueType.BUILDUP,
                "position_ms": first_buildup["start_ms"],
                "confidence": 0.8,
                **CUE_CONFIG[CueType.BUILDUP],
            })

        # 4. DROP - first drop
        if drops:
            first_drop = drops[0]
            cues.append({
                "type": CueType.DROP,
                "position_ms": first_drop["start_ms"],
                "confidence": 0.9,
                **CUE_CONFIG[CueType.DROP],
            })

        # 5. BREAKDOWN - after first drop
        post_drop_breakdowns = [
            b for b in breakdowns
            if drops and b["start_ms"] > drops[0]["start_ms"]
        ]
        if post_drop_breakdowns:
            cues.append({
                "type": CueType.BREAKDOWN,
                "position_ms": post_drop_breakdowns[0]["start_ms"],
                "confidence": 0.85,
                **CUE_CONFIG[CueType.BREAKDOWN],
            })

        # 6. DROP_2 - second drop if exists
        if len(drops) > 1:
            cues.append({
                "type": CueType.DROP_2,
                "position_ms": drops[1]["start_ms"],
                "confidence": 0.85,
                **CUE_CONFIG[CueType.DROP_2],
            })

        # 7. OUTRO - outro start or estimated from end
        if outros:
            cues.append({
                "type": CueType.OUTRO,
                "position_ms": outros[0]["start_ms"],
                "confidence": 0.8,
                **CUE_CONFIG[CueType.OUTRO],
            })
        elif beat_times_ms and duration_seconds:
            # Estimate outro at ~32 bars from end
            bar_duration_ms = (60000 / bpm) * 4
            outro_start = int((duration_seconds * 1000) - (32 * bar_duration_ms))

            # Snap to nearest phrase boundary
            outro_start = snap_to_phrase(
                outro_start,
                beat_times_ms,
                phrase_bars=8,
            )

            cues.append({
                "type": CueType.OUTRO,
                "position_ms": max(0, outro_start),
                "confidence": 0.6,
                **CUE_CONFIG[CueType.OUTRO],
            })

        # Sort by position
        cues.sort(key=lambda c: c["position_ms"])

        return cues


def detect_cues_for_pending(
    session: Session,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """
    Detect cue points for all tracks in SEPARATED or ANALYZED state.

    Args:
        session: Database session.
        limit: Maximum number of tracks to process.

    Returns:
        Tuple of (success_count, error_count).
    """
    # Process separated tracks first (better quality), then analyzed
    query = session.query(Track).filter(
        Track.state.in_([
            TrackState.SEPARATED.value,
            TrackState.ANALYZED.value,
        ])
    ).order_by(
        # Prioritize separated tracks
        Track.state.desc()
    )

    if limit:
        query = query.limit(limit)

    tracks = query.all()

    if not tracks:
        return 0, 0

    stage = CueDetectionStage()
    success = 0
    errors = 0

    for track in tracks:
        if stage.detect_cues(session, track):
            success += 1
        else:
            errors += 1

    return success, errors
