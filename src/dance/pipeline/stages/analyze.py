"""
Audio analysis stage using Essentia and librosa.

Extracts BPM, key, energy, and mood from audio files.
This is the core value-add for DJ workflow automation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from dance.core.database import now_utc, Analysis, Track, TrackState
from dance.pipeline.utils.audio import (
    aggregate_rms,
    detect_key_from_chroma,
    normalize_bpm,
)
from dance.pipeline.utils.camelot import key_to_camelot
from dance.pipeline.utils.db import upsert

logger = logging.getLogger(__name__)

# Try to import essentia - it has complex deps
try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logger.warning("Essentia not available - using librosa fallback")

# librosa is more reliable as a fallback
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available")


class AnalysisStage:
    """
    Full-mix audio analysis: BPM, key (Camelot), energy, brightness, warmth,
    danceability. Writes one ``AudioAnalysis`` row per track with
    ``stem_file_id=NULL``.
    """

    name = "analyze"
    input_state = TrackState.PENDING
    output_state = TrackState.ANALYZED
    error_state = TrackState.ERROR

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    # Backwards-compat shim — older code/tests call .analyze_track().
    def analyze_track(self, session: Session, track: Track) -> bool:
        from dance.config import get_settings

        return self.process(session, track, get_settings())

    def process(self, session, track, settings) -> bool:
        """
        Run full analysis on a track.

        Args:
            session: Database session.
            track: Track to analyze.

        Returns:
            True if analysis succeeded, False otherwise.
        """
        # Update state
        track.state = TrackState.ANALYZING.value
        session.commit()

        try:
            file_path = Path(track.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Track file not found: {file_path}")

            # Load audio
            logger.info(f"Analyzing: {track.artist} - {track.title}")
            audio, sr = self._load_audio(file_path)

            # Update duration if not set
            if not track.duration_seconds:
                track.duration_seconds = len(audio) / sr

            # Analyze BPM
            bpm, bpm_confidence = self._analyze_bpm(audio, sr)

            # Analyze key
            key, mode, key_confidence = self._analyze_key(audio, sr)
            camelot = key_to_camelot(key, mode)
            standard_key = f"{key}{mode[0]}"  # e.g., "Am" or "C"

            # Analyze energy
            energy_stats = self._analyze_energy(audio, sr)

            # Compute floor energy (1-10 scale)
            floor_energy = self._compute_floor_energy(
                energy_stats["overall"],
                energy_stats["peak"],
                energy_stats.get("brightness", 0.5),
            )

            # Create or update analysis row for the full mix (stem_file_id = NULL).
            upsert(
                session,
                Analysis,
                where={"track_id": track.id, "stem_file_id": None},
                bpm=bpm,
                bpm_confidence=bpm_confidence,
                key_camelot=camelot,
                key_standard=standard_key,
                key_confidence=key_confidence,
                energy_overall=energy_stats["overall"],
                energy_peak=energy_stats["peak"],
                brightness=energy_stats.get("brightness", 0.5),
                warmth=energy_stats.get("warmth", 0.5),
                floor_energy=floor_energy,
                danceability=self._estimate_danceability(audio, sr, bpm),
                analyzed_at=now_utc(),
            )

            # Update track state
            track.state = TrackState.ANALYZED.value
            track.analyzed_at = now_utc()
            session.commit()

            logger.info(
                f"Analyzed: {track.artist} - {track.title} | "
                f"BPM: {bpm:.1f}, Key: {camelot}, Energy: E{floor_energy}"
            )
            return True

        except Exception as e:
            logger.error(f"Analysis failed for {track.title}: {e}")
            track.state = TrackState.ERROR.value
            track.error_message = str(e)[:500]
            session.commit()
            return False

    def _load_audio(self, file_path: Path) -> tuple[np.ndarray, int]:
        """Load audio file as mono float array."""
        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            return audio, sr
        elif ESSENTIA_AVAILABLE:
            loader = es.MonoLoader(filename=str(file_path), sampleRate=self.sample_rate)
            return loader(), self.sample_rate
        else:
            raise RuntimeError("No audio loading library available (need librosa or essentia)")

    def _analyze_bpm(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """
        Detect BPM with confidence.

        For house/techno, BPM should typically be 118-145.
        Handles half/double time detection.
        """
        if ESSENTIA_AVAILABLE:
            try:
                rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
                bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
                confidence = float(np.mean(beats_confidence)) if len(beats_confidence) > 0 else 0.5
                return normalize_bpm(bpm), confidence
            except Exception as e:
                logger.warning(f"Essentia BPM failed, falling back to librosa: {e}")

        if LIBROSA_AVAILABLE:
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            # tempo can be a float or ndarray depending on librosa version
            bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            return normalize_bpm(bpm), 0.7  # Default confidence for librosa

        raise RuntimeError("No BPM detection available")

    def _analyze_key(self, audio: np.ndarray, sr: int) -> tuple[str, str, float]:
        """
        Detect musical key and mode.

        Returns:
            (key, mode, confidence) - e.g., ("A", "minor", 0.8)
        """
        if ESSENTIA_AVAILABLE:
            try:
                key_extractor = es.KeyExtractor()
                key, scale, strength = key_extractor(audio)
                return key, scale, strength
            except Exception as e:
                logger.warning(f"Essentia key detection failed: {e}")

        if LIBROSA_AVAILABLE:
            # Use chroma features for key detection
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            key, mode, _ = detect_key_from_chroma(chroma)
            return key, mode, 0.6  # Lower confidence than essentia

        return "C", "major", 0.5  # Default fallback

    def _analyze_energy(self, audio: np.ndarray, sr: int) -> dict:
        """
        Compute energy metrics for DJ set building.

        Returns dict with:
        - overall: Average RMS energy (0-1)
        - peak: Peak energy (0-1)
        - brightness: Spectral centroid indicator (0-1)
        - warmth: Bass energy ratio (0-1)
        """
        # RMS energy
        if LIBROSA_AVAILABLE:
            rms = librosa.feature.rms(y=audio)[0]
        else:
            # Manual RMS calculation
            frame_length = 2048
            hop_length = 512
            rms_list = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                rms_list.append(np.sqrt(np.mean(frame ** 2)))
            rms = np.array(rms_list)

        overall, peak = aggregate_rms(rms)

        # Spectral centroid (brightness)
        if LIBROSA_AVAILABLE:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            brightness = np.clip(np.mean(centroid) / 5000, 0, 1)
        else:
            brightness = 0.5

        # Bass energy ratio (warmth)
        if LIBROSA_AVAILABLE:
            # Simple bass band energy ratio
            stft = np.abs(librosa.stft(audio))
            freqs = librosa.fft_frequencies(sr=sr)
            bass_mask = freqs < 250
            bass_energy = np.mean(stft[bass_mask, :])
            total_energy = np.mean(stft)
            warmth = np.clip(bass_energy / total_energy * 2, 0, 1) if total_energy > 0 else 0.5
        else:
            warmth = 0.5

        return {
            "overall": float(overall),
            "peak": float(peak),
            "brightness": float(brightness),
            "warmth": float(warmth),
        }

    def _compute_floor_energy(
        self,
        overall: float,
        peak: float,
        brightness: float,
    ) -> int:
        """
        Compute 1-10 floor energy score for set building.

        1-2: Opener/closer, ambient
        3-4: Warm-up deep house
        5-6: Groove-focused house
        7-8: Peak-time, driving
        9-10: Peak intensity, hard techno
        """
        # Weighted combination
        raw_score = overall * 0.5 + peak * 0.3 + brightness * 0.2

        # Scale to 1-10
        score = int(raw_score * 10) + 1
        return max(1, min(10, score))

    def _estimate_danceability(
        self,
        audio: np.ndarray,
        sr: int,
        bpm: float,
    ) -> float:
        """
        Estimate danceability from rhythm regularity.

        House/techno is highly danceable due to consistent 4/4 beat.
        """
        # For house/techno, if BPM is in range and consistent, it's danceable
        if 118 <= bpm <= 145:
            base_danceability = 0.85
        elif 90 <= bpm <= 180:
            base_danceability = 0.7
        else:
            base_danceability = 0.5

        # Could add more sophisticated analysis here
        return base_danceability
