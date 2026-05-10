"""Region detection stage — sections, cues, loops.

Consumes STEMS_ANALYZED tracks and writes ``regions`` rows of three kinds:
SECTION (one per detected phrase, labelled intro/drop/...), CUE (at position 0
plus each phrase boundary that lands on a downbeat), and LOOP (an 8-bar
candidate at the start of each section >= 8 bars, whole-track plus per-stem
when that stem's ``presence_ratio`` > 0.3).

Also persists the beat grid + detected phrases to the ``beats`` / ``phrases``
tables. All emitted regions use ``source = RegionSource.AUTO``; manual /
imported regions are preserved across re-runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Beat,
    Phrase,
    Region,
    RegionSource,
    RegionType,
    SectionLabel,
    StemFile,
    StemKind,
    Track,
    TrackState,
)
from dance.pipeline.utils.beats import detect_beats, detect_phrases

logger = logging.getLogger(__name__)

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.warning("librosa not available — region detection disabled")


_PHRASE_TYPE_TO_SECTION: dict[str, SectionLabel] = {
    "intro": SectionLabel.INTRO,
    "buildup": SectionLabel.BUILDUP,
    "drop": SectionLabel.DROP,
    "breakdown": SectionLabel.BREAKDOWN,
    "outro": SectionLabel.OUTRO,
}

_BEATS_PER_BAR = 4
_LOOP_BARS = 8
_PRESENCE_THRESHOLD = 0.3   # stem presence_ratio above this counts as "present"
_DOWNBEAT_TOLERANCE_MS = 200
_LOOP_CONF_FULL = 0.7
_LOOP_CONF_STEM = 0.5


class RegionDetectionStage:
    """Detect sections, cues, and loop candidates for each analyzed track."""

    name = "detect_regions"
    input_state = TrackState.STEMS_ANALYZED
    output_state = TrackState.REGIONS_DETECTED
    error_state = TrackState.ERROR

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------

    def process(self, session: Session, track: Track, settings: Settings) -> bool:
        if not _LIBROSA_OK:
            track.state = self.error_state.value
            track.error_message = "librosa not installed"
            session.commit()
            return False

        track.state = TrackState.DETECTING_REGIONS.value
        session.commit()

        try:
            full_analysis = (
                session.query(AudioAnalysis)
                .filter_by(track_id=track.id, stem_file_id=None)
                .one_or_none()
            )
            if full_analysis is None or not full_analysis.bpm:
                raise RuntimeError(
                    f"track {track.id} has no full-mix AudioAnalysis with BPM"
                )

            path = Path(track.file_path)
            if not path.exists():
                raise FileNotFoundError(path)

            audio, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
            duration_ms = int(len(audio) * 1000 / sr)

            beat_times, _ = detect_beats(audio, sr, bpm=full_analysis.bpm)
            phrases = detect_phrases(audio, sr, beat_times, full_analysis.bpm)

            self._clear_auto_data(session, track.id)
            self._persist_beats(session, track.id, beat_times)
            self._persist_phrases(session, track.id, phrases)

            beat_times_ms = [int(t * 1000) for t in beat_times]
            self._write_sections(session, track.id, phrases)
            self._write_cues(session, track.id, phrases, beat_times_ms)
            self._write_loops(
                session, track, phrases, full_analysis.bpm, duration_ms
            )

            track.state = self.output_state.value
            session.commit()
            logger.info(
                "Regions for track %s: %d phrases, %d beats",
                track.id, len(phrases), len(beat_times),
            )
            return True

        except Exception as exc:
            logger.exception("Region detection failed for track %s", track.id)
            session.rollback()
            track.state = self.error_state.value
            track.error_message = f"detect_regions: {exc}"[:500]
            session.commit()
            return False

    # ------------------------------------------------------------------
    # Cleanup + persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_auto_data(session: Session, track_id: int) -> None:
        """Drop auto-source regions, beats, and phrases for this track."""
        session.query(Region).filter(
            Region.track_id == track_id,
            Region.source == RegionSource.AUTO.value,
        ).delete(synchronize_session=False)
        session.query(Beat).filter(Beat.track_id == track_id).delete(
            synchronize_session=False
        )
        session.query(Phrase).filter(Phrase.track_id == track_id).delete(
            synchronize_session=False
        )
        session.flush()

    @staticmethod
    def _persist_beats(session: Session, track_id: int, beat_times) -> None:
        for i, t in enumerate(beat_times):
            session.add(
                Beat(
                    track_id=track_id,
                    position_ms=int(t * 1000),
                    beat_number=(i % _BEATS_PER_BAR) + 1,
                    bar_number=(i // _BEATS_PER_BAR) + 1,
                    downbeat=(i % _BEATS_PER_BAR == 0),
                )
            )

    @staticmethod
    def _persist_phrases(
        session: Session, track_id: int, phrases: list[dict]
    ) -> None:
        for p in phrases:
            session.add(
                Phrase(
                    track_id=track_id,
                    start_ms=p["start_ms"],
                    end_ms=p["end_ms"],
                    bar_count=p.get("bar_count"),
                    phrase_type=p.get("phrase_type"),
                    energy_level=p.get("energy_level"),
                )
            )

    # ------------------------------------------------------------------
    # Region emission
    # ------------------------------------------------------------------

    @staticmethod
    def _write_sections(
        session: Session, track_id: int, phrases: list[dict]
    ) -> None:
        for p in phrases:
            label = _PHRASE_TYPE_TO_SECTION.get(
                p.get("phrase_type") or "", SectionLabel.OTHER
            )
            session.add(Region(
                track_id=track_id, stem_file_id=None,
                position_ms=p["start_ms"],
                length_ms=max(0, p["end_ms"] - p["start_ms"]),
                region_type=RegionType.SECTION.value,
                section_label=label.value, length_bars=p.get("bar_count"),
                name=label.value.title(), snapped_to="phrase",
                source=RegionSource.AUTO.value, confidence=0.6,
            ))

    @staticmethod
    def _write_cues(
        session: Session,
        track_id: int,
        phrases: list[dict],
        beat_times_ms: list[int],
    ) -> None:
        """Cue at position 0 + at any phrase boundary near a downbeat."""
        emitted: set[int] = set()

        def emit(pos: int, name: str) -> None:
            if pos in emitted:
                return
            emitted.add(pos)
            session.add(Region(
                track_id=track_id, stem_file_id=None,
                position_ms=pos, length_ms=None,
                region_type=RegionType.CUE.value, name=name,
                snapped_to="bar", source=RegionSource.AUTO.value,
                confidence=0.7,
            ))

        emit(0, "Start")
        downbeats_ms = beat_times_ms[::_BEATS_PER_BAR]
        if not downbeats_ms:
            return
        arr = np.array(downbeats_ms)
        # Skip phrase[0] — its boundary is the track start.
        for p in phrases[1:]:
            pos = p["start_ms"]
            idx = int(np.argmin(np.abs(arr - pos)))
            if abs(arr[idx] - pos) <= _DOWNBEAT_TOLERANCE_MS:
                emit(int(arr[idx]), f"Cue {len(emitted)}")

    def _write_loops(
        self,
        session: Session,
        track: Track,
        phrases: list[dict],
        bpm: float,
        duration_ms: int,
    ) -> None:
        if bpm <= 0:
            return
        loop_length_ms = int(_LOOP_BARS * _BEATS_PER_BAR * 60_000 / bpm)
        present_stems = self._present_stems(session, track.id)

        def loop(stem_id: Optional[int], name: str, conf: float, start: int) -> Region:
            return Region(
                track_id=track.id, stem_file_id=stem_id,
                position_ms=start, length_ms=loop_length_ms,
                region_type=RegionType.LOOP.value, length_bars=_LOOP_BARS,
                name=name, snapped_to="bar",
                source=RegionSource.AUTO.value, confidence=conf,
            )

        for p in phrases:
            if (p.get("bar_count") or 0) < _LOOP_BARS:
                continue
            start = p["start_ms"]
            if start + loop_length_ms > duration_ms:
                continue
            session.add(loop(None, f"Loop {_LOOP_BARS}b", _LOOP_CONF_FULL, start))
            for kind, stem_id in present_stems.items():
                session.add(loop(
                    stem_id,
                    f"{kind.value.title()} loop {_LOOP_BARS}b",
                    _LOOP_CONF_STEM, start,
                ))

    @staticmethod
    def _present_stems(session: Session, track_id: int) -> dict[StemKind, int]:
        """Return {kind: stem_file_id} for stems flagged as present globally."""
        rows = (
            session.query(StemFile, AudioAnalysis)
            .join(AudioAnalysis, AudioAnalysis.stem_file_id == StemFile.id)
            .filter(StemFile.track_id == track_id)
            .all()
        )
        present: dict[StemKind, int] = {}
        for stem, analysis in rows:
            if analysis.presence_ratio is None:
                continue
            if analysis.presence_ratio <= _PRESENCE_THRESHOLD:
                continue
            try:
                kind = StemKind(stem.kind)
            except ValueError:
                continue
            present[kind] = stem.id
        return present
