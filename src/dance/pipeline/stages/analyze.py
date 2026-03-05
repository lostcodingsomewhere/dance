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

from dance.core.database import Analysis, Track, TrackState
from dance.pipeline.utils.camelot import key_to_camelot

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
    Comprehensive audio analysis for DJ mixing.

    For house/techno, we prioritize:
    1. Accurate BPM (within 0.1 BPM for beatmatching)
    2. Energy levels (for set building)
    3. Key detection (for harmonic mixing)
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def analyze_track(self, session: Session, track: Track) -> bool:
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

            # Create or update analysis record
            analysis = session.query(Analysis).filter_by(track_id=track.id).first()
            if analysis is None:
                analysis = Analysis(track_id=track.id)
                session.add(analysis)

            # Update analysis fields
            analysis.bpm = bpm
            analysis.bpm_confidence = bpm_confidence
            analysis.key_camelot = camelot
            analysis.key_standard = standard_key
            analysis.key_confidence = key_confidence
            analysis.energy_overall = energy_stats["overall"]
            analysis.energy_peak = energy_stats["peak"]
            analysis.brightness = energy_stats.get("brightness", 0.5)
            analysis.warmth = energy_stats.get("warmth", 0.5)
            analysis.floor_energy = floor_energy
            analysis.peak_time_suitability = floor_energy / 10.0

            # Mood analysis (simplified - would use Essentia TF models in production)
            analysis.mood_dark = self._estimate_darkness(energy_stats)
            analysis.mood_aggressive = energy_stats.get("brightness", 0.5)
            analysis.mood_electronic = 0.9  # Assume electronic for house/techno

            # Danceability from rhythm regularity
            analysis.danceability = self._estimate_danceability(audio, sr, bpm)

            # Update track state
            track.state = TrackState.ANALYZED.value
            track.analyzed_at = datetime.utcnow()
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
                bpm = self._normalize_bpm(bpm)
                return bpm, confidence
            except Exception as e:
                logger.warning(f"Essentia BPM failed, falling back to librosa: {e}")

        if LIBROSA_AVAILABLE:
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            # tempo can be a float or ndarray depending on librosa version
            bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            bpm = self._normalize_bpm(bpm)
            return bpm, 0.7  # Default confidence for librosa

        raise RuntimeError("No BPM detection available")

    def _normalize_bpm(self, bpm: float) -> float:
        """
        Normalize BPM to house/techno range (118-145).

        Most house is 120-130, techno 125-140.
        If detected BPM is half or double, correct it.
        """
        # Handle extreme values
        if bpm < 90:
            return bpm * 2
        elif bpm > 180:
            return bpm / 2
        elif bpm < 110:
            # Could be half-time, check if double is in range
            if 118 <= bpm * 2 <= 145:
                return bpm * 2
        return bpm

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
            chroma_mean = np.mean(chroma, axis=1)

            # Simple key detection from chroma
            key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key_idx = int(np.argmax(chroma_mean))
            key = key_names[key_idx]

            # Estimate major/minor from chroma pattern
            # This is a simplification - real key detection is more complex
            mode = "minor" if chroma_mean[(key_idx + 3) % 12] > chroma_mean[(key_idx + 4) % 12] else "major"

            return key, mode, 0.6  # Lower confidence for simple method

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
            rms = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                rms.append(np.sqrt(np.mean(frame ** 2)))
            rms = np.array(rms)

        # Normalize to 0-1 (calibrated for electronic music)
        overall = np.clip(np.mean(rms) / 0.2, 0, 1)
        peak = np.clip(np.percentile(rms, 95) / 0.3, 0, 1)

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

    def _estimate_darkness(self, energy_stats: dict) -> float:
        """
        Estimate how 'dark' a track sounds.

        Dark = low brightness, high warmth (bass-heavy)
        """
        brightness = energy_stats.get("brightness", 0.5)
        warmth = energy_stats.get("warmth", 0.5)

        # Inverse of brightness, weighted with warmth
        darkness = (1 - brightness) * 0.6 + warmth * 0.4
        return float(np.clip(darkness, 0, 1))

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


def analyze_pending_tracks(
    session: Session,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """
    Analyze all tracks in PENDING state.

    Args:
        session: Database session.
        limit: Maximum number of tracks to analyze.

    Returns:
        Tuple of (success_count, error_count).
    """
    query = session.query(Track).filter(
        Track.state.in_([TrackState.PENDING.value])
    )

    if limit:
        query = query.limit(limit)

    tracks = query.all()

    stage = AnalysisStage()
    success = 0
    errors = 0

    for track in tracks:
        if stage.analyze_track(session, track):
            success += 1
        else:
            errors += 1

    return success, errors
