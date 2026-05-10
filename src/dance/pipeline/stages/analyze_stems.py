"""Per-stem analysis stage.

Consumes ``StemFile`` rows produced by stem separation and writes one
``AudioAnalysis`` row per stem (with ``stem_file_id`` set). Each stem gets
generic metrics (BPM, RMS curve, presence intervals, energy) plus a small
set of kind-specific metrics:

- **drums**: ``kick_density`` (kicks/sec from low-freq onset detection)
- **bass**: ``dominant_pitch_camelot`` (chroma-based, minor mode assumed)
- **vocals**: ``vocal_present`` (presence_ratio > 0.15)
- **other**: ``dominant_pitch_camelot`` (mode inferred from chroma)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np

from sqlalchemy.orm import Session

from dance.config import Settings
from dance.core.database import now_utc, AudioAnalysis, StemFile, StemKind, Track, TrackState
from dance.core.serialization import encode_curve
from dance.pipeline.utils.audio import (
    aggregate_rms,
    detect_key_from_chroma,
    normalize_bpm,
)
from dance.pipeline.utils.camelot import key_to_camelot
from dance.pipeline.utils.db import get_stems_for_track, upsert

logger = logging.getLogger(__name__)


# Constants. Tuneable but stable across stems.
SAMPLE_RATE = 22050
RMS_HOP_MS = 100
RMS_FRAME_LENGTH = 2048
RMS_NORM_SCALE = 0.3       # Divide RMS by this before clipping to [0, 1].
PRESENCE_THRESHOLD = 0.05  # Normalised-RMS threshold for "present".
MIN_INTERVAL_MS = 250      # Drop sub-quarter-second presence runs.


class StemAnalysisStage:
    """Compute per-stem audio metrics. One AudioAnalysis row per StemFile."""

    name = "analyze_stems"
    input_state = TrackState.SEPARATED
    output_state = TrackState.STEMS_ANALYZED
    error_state = TrackState.ERROR

    # ------------------------------------------------------------------

    def process(self, session: Session, track: Track, settings: Settings) -> bool:
        track.state = TrackState.ANALYZING_STEMS.value
        session.commit()

        try:
            stems = get_stems_for_track(session, track.id)
            if not stems:
                raise RuntimeError(f"No stem files found for track {track.id}")

            now = now_utc()
            for stem in stems:
                self._analyze_stem(session, track, stem, now)

            track.state = self.output_state.value
            session.commit()
            logger.info("Analyzed %d stems for track %s", len(stems), track.id)
            return True

        except Exception as exc:  # noqa: BLE001 — stage reports via state
            logger.error("Stem analysis failed for track %s: %s", track.id, exc)
            track.state = self.error_state.value
            track.error_message = f"{self.name}: {exc}"[:500]
            session.commit()
            return False

    # ------------------------------------------------------------------

    def _analyze_stem(
        self,
        session: Session,
        track: Track,
        stem: StemFile,
        now: datetime,
    ) -> None:
        stem_path = Path(stem.path)
        if not stem_path.exists():
            raise FileNotFoundError(f"Stem audio missing: {stem_path}")

        audio, sr = librosa.load(str(stem_path), sr=SAMPLE_RATE, mono=True)
        duration_s = len(audio) / sr if sr else 0.0

        # ---- RMS curve + energy summary --------------------------------
        hop_length = max(1, int(RMS_HOP_MS / 1000.0 * sr))
        raw_rms = librosa.feature.rms(
            y=audio, frame_length=RMS_FRAME_LENGTH, hop_length=hop_length
        )[0]
        norm_curve = np.clip(raw_rms / RMS_NORM_SCALE, 0.0, 1.0).astype(np.float32)
        energy_overall, energy_peak = aggregate_rms(raw_rms)

        # ---- Presence ratio + intervals --------------------------------
        present_mask = norm_curve > PRESENCE_THRESHOLD
        presence_ratio = float(present_mask.mean()) if present_mask.size else 0.0
        intervals = _runs_to_intervals(present_mask, RMS_HOP_MS, MIN_INTERVAL_MS)

        # ---- BPM (per-stem; unreliable on non-drums) -------------------
        bpm, bpm_confidence = _stem_bpm(audio, sr, duration_s)

        # ---- Kind-specific metrics -------------------------------------
        kind = stem.kind
        kick_density: float | None = None
        dominant_pitch_camelot: str | None = None
        dominant_pitch_confidence: float | None = None
        vocal_present: bool | None = None

        if kind == StemKind.DRUMS.value:
            kick_density = _kick_density(audio, sr, duration_s)
        elif kind == StemKind.BASS.value:
            dominant_pitch_camelot, dominant_pitch_confidence = _dominant_camelot(
                audio, sr, force_mode="minor"
            )
        elif kind == StemKind.VOCALS.value:
            vocal_present = presence_ratio > 0.15
        elif kind == StemKind.OTHER.value:
            dominant_pitch_camelot, dominant_pitch_confidence = _dominant_camelot(
                audio, sr, force_mode=None
            )

        # ---- Upsert AudioAnalysis row ----------------------------------
        upsert(
            session,
            AudioAnalysis,
            where={"track_id": track.id, "stem_file_id": stem.id},
            bpm=bpm,
            bpm_confidence=bpm_confidence,
            energy_overall=energy_overall,
            energy_peak=energy_peak,
            floor_energy=_floor_energy(energy_overall, energy_peak),
            presence_ratio=presence_ratio,
            rms_curve=encode_curve(norm_curve),
            rms_curve_hop_ms=RMS_HOP_MS,
            rms_curve_length=int(norm_curve.shape[0]),
            presence_intervals=json.dumps(intervals),
            kick_density=kick_density,
            dominant_pitch_camelot=dominant_pitch_camelot,
            dominant_pitch_confidence=dominant_pitch_confidence,
            vocal_present=vocal_present,
            analyzed_at=now,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------


def _runs_to_intervals(
    mask: np.ndarray, hop_ms: int, min_ms: int
) -> list[list[int]]:
    """Convert a boolean per-frame mask into `[start_ms, end_ms]` runs."""
    if mask.size == 0:
        return []
    # Pad with False on both ends so we always close runs.
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    out: list[list[int]] = []
    for s, e in zip(starts, ends):
        start_ms = int(s * hop_ms)
        end_ms = int(e * hop_ms)
        if end_ms - start_ms >= min_ms:
            out.append([start_ms, end_ms])
    return out


def _stem_bpm(audio: np.ndarray, sr: int, duration_s: float) -> tuple[float, float]:
    """Per-stem BPM with a confidence that drops on sparse beat detection."""
    if len(audio) == 0:
        return 0.0, 0.0
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    bpm = normalize_bpm(bpm)

    # Confidence heuristic: if we got fewer than 8 beats per estimated bar
    # (i.e. roughly 2 bars at 4 beats/bar), the detector is unsure.
    est_bars = max(1.0, duration_s * bpm / (60.0 * 4.0))
    beats_per_bar = len(beat_frames) / est_bars if est_bars else 0.0
    confidence = 0.7 if beats_per_bar >= 8 else 0.4
    return bpm, confidence


def _kick_density(audio: np.ndarray, sr: int, duration_s: float) -> float:
    """Onsets per second restricted to low (kick) frequencies."""
    if duration_s <= 0:
        return 0.0
    # 16 mel bins between 20-200 Hz keeps each filter wider than one FFT bin
    # at 22050 Hz; using more bins triggers librosa's empty-filter warning.
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=16, fmin=20, fmax=200
    )
    low_band = mel.sum(axis=0)
    onsets = librosa.onset.onset_detect(
        onset_envelope=low_band, sr=sr, units="frames"
    )
    return float(len(onsets)) / duration_s


def _dominant_camelot(
    audio: np.ndarray, sr: int, *, force_mode: str | None
) -> tuple[str | None, float | None]:
    """Chroma-based dominant pitch → Camelot. Returns (None, None) on empty input."""
    if len(audio) == 0:
        return None, None
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    pitch, mode, confidence = detect_key_from_chroma(chroma, mode=force_mode)
    return key_to_camelot(pitch, mode), confidence


def _floor_energy(overall: float, peak: float) -> int:
    """Per-stem floor-energy score on the 1-10 scale."""
    raw = overall * 0.6 + peak * 0.4
    return max(1, min(10, int(raw * 10) + 1))
