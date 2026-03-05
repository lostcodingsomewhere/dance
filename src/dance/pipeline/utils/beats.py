"""
Beat grid and phrase detection utilities.

House/techno is structured around:
- 4/4 time signature (4 beats per bar)
- Phrases of 4, 8, 16, or 32 bars
- Energy changes at phrase boundaries
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def detect_beats(
    audio: np.ndarray,
    sr: int,
    bpm: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Detect beat positions in audio.

    Args:
        audio: Audio signal (mono).
        sr: Sample rate.
        bpm: Known BPM (optional, improves accuracy).

    Returns:
        Tuple of (beat_times in seconds, estimated BPM).
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required for beat detection")

    # Use librosa's beat tracker
    if bpm:
        # Use known BPM as prior
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            bpm=bpm,
            units="frames",
        )
    else:
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            units="frames",
        )

    # Convert frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Handle tempo being array or scalar
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    return beat_times, tempo


def snap_to_beat(
    position_ms: int,
    beat_times_ms: list[int],
    tolerance_ms: int = 100,
) -> int:
    """
    Snap a position to the nearest beat.

    Args:
        position_ms: Position to snap (milliseconds).
        beat_times_ms: List of beat positions (milliseconds).
        tolerance_ms: Maximum distance to snap.

    Returns:
        Snapped position (milliseconds).
    """
    if not beat_times_ms:
        return position_ms

    # Find nearest beat
    beat_array = np.array(beat_times_ms)
    distances = np.abs(beat_array - position_ms)
    nearest_idx = np.argmin(distances)

    if distances[nearest_idx] <= tolerance_ms:
        return int(beat_times_ms[nearest_idx])

    return position_ms


def snap_to_downbeat(
    position_ms: int,
    beat_times_ms: list[int],
    beats_per_bar: int = 4,
    tolerance_ms: int = 200,
) -> int:
    """
    Snap a position to the nearest downbeat (first beat of bar).

    Args:
        position_ms: Position to snap (milliseconds).
        beat_times_ms: List of beat positions (milliseconds).
        beats_per_bar: Beats per bar (4 for house/techno).
        tolerance_ms: Maximum distance to snap.

    Returns:
        Snapped position (milliseconds).
    """
    if not beat_times_ms:
        return position_ms

    # Get downbeats (every 4th beat)
    downbeats = beat_times_ms[::beats_per_bar]

    return snap_to_beat(position_ms, downbeats, tolerance_ms)


def snap_to_phrase(
    position_ms: int,
    beat_times_ms: list[int],
    phrase_bars: int = 8,
    beats_per_bar: int = 4,
    tolerance_ms: int = 500,
) -> int:
    """
    Snap a position to the nearest phrase boundary.

    Args:
        position_ms: Position to snap (milliseconds).
        beat_times_ms: List of beat positions (milliseconds).
        phrase_bars: Bars per phrase (8 or 16 typical for house/techno).
        beats_per_bar: Beats per bar.
        tolerance_ms: Maximum distance to snap.

    Returns:
        Snapped position (milliseconds).
    """
    if not beat_times_ms:
        return position_ms

    # Get phrase boundaries
    beats_per_phrase = phrase_bars * beats_per_bar
    phrase_boundaries = beat_times_ms[::beats_per_phrase]

    return snap_to_beat(position_ms, phrase_boundaries, tolerance_ms)


def detect_phrases(
    audio: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    bpm: float,
) -> list[dict]:
    """
    Detect phrase boundaries and types in audio.

    Args:
        audio: Audio signal (mono).
        sr: Sample rate.
        beat_times: Array of beat times in seconds.
        bpm: Track BPM.

    Returns:
        List of phrase dicts with start_ms, end_ms, bar_count, phrase_type.
    """
    if not LIBROSA_AVAILABLE:
        return []

    # Calculate energy per bar
    beats_per_bar = 4
    bar_energies = []

    for i in range(0, len(beat_times) - beats_per_bar, beats_per_bar):
        start_time = beat_times[i]
        end_time = beat_times[i + beats_per_bar] if i + beats_per_bar < len(beat_times) else len(audio) / sr

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        if end_sample <= len(audio):
            bar_audio = audio[start_sample:end_sample]
            bar_energy = np.sqrt(np.mean(bar_audio ** 2))
            bar_energies.append(bar_energy)

    if not bar_energies:
        return []

    bar_energies = np.array(bar_energies)
    mean_energy = np.mean(bar_energies)

    # Detect significant energy changes
    energy_diff = np.abs(np.diff(bar_energies))
    threshold = np.percentile(energy_diff, 80)

    # Find phrase boundaries (significant changes)
    phrase_starts = [0]
    for i, diff in enumerate(energy_diff):
        if diff > threshold:
            # Snap to nearest 8-bar boundary
            snapped = ((i + 1) // 8) * 8
            if snapped > 0 and snapped not in phrase_starts and snapped > phrase_starts[-1] + 4:
                phrase_starts.append(snapped)

    # Add end of track
    phrase_starts.append(len(bar_energies))

    # Build phrase list
    phrases = []
    for i in range(len(phrase_starts) - 1):
        start_bar = phrase_starts[i]
        end_bar = phrase_starts[i + 1]
        bar_count = end_bar - start_bar

        # Get start/end times
        if start_bar * beats_per_bar >= len(beat_times):
            continue

        start_beat_idx = start_bar * beats_per_bar
        end_beat_idx = min(end_bar * beats_per_bar, len(beat_times) - 1)

        start_ms = int(beat_times[start_beat_idx] * 1000)
        end_ms = int(beat_times[end_beat_idx] * 1000)

        # Calculate phrase energy
        phrase_energy = np.mean(bar_energies[start_bar:end_bar]) if end_bar <= len(bar_energies) else 0

        # Classify phrase type
        if i == 0:
            phrase_type = "intro"
        elif i == len(phrase_starts) - 2:
            phrase_type = "outro"
        elif phrase_energy > mean_energy * 1.15:
            phrase_type = "drop"
        elif phrase_energy < mean_energy * 0.85:
            phrase_type = "breakdown"
        else:
            phrase_type = "buildup"

        phrases.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "bar_count": bar_count,
            "phrase_type": phrase_type,
            "energy_level": float(phrase_energy),
        })

    return phrases


def estimate_intro_length(
    bar_energies: np.ndarray,
    threshold_percentile: float = 60,
) -> int:
    """
    Estimate intro length in bars.

    Intro is typically where energy is below a threshold.

    Args:
        bar_energies: Array of per-bar energy values.
        threshold_percentile: Percentile to use as threshold.

    Returns:
        Estimated intro length in bars.
    """
    if len(bar_energies) < 4:
        return 0

    threshold = np.percentile(bar_energies, threshold_percentile)

    # Find first bar above threshold
    for i, energy in enumerate(bar_energies):
        if energy > threshold:
            # Snap to 8 or 16 bar boundary
            if i <= 16:
                return 16 if i > 8 else 8
            elif i <= 32:
                return 32
            return i

    return 16  # Default


def estimate_outro_start(
    bar_energies: np.ndarray,
    threshold_percentile: float = 60,
) -> int:
    """
    Estimate outro start in bars from beginning.

    Args:
        bar_energies: Array of per-bar energy values.
        threshold_percentile: Percentile to use as threshold.

    Returns:
        Estimated outro start bar number.
    """
    if len(bar_energies) < 4:
        return max(0, len(bar_energies) - 8)

    threshold = np.percentile(bar_energies, threshold_percentile)

    # Find last bar above threshold (working backwards)
    for i in range(len(bar_energies) - 1, -1, -1):
        if bar_energies[i] > threshold:
            # Snap to 8 bar boundary
            outro_start = ((i + 8) // 8) * 8
            return min(outro_start, len(bar_energies) - 8)

    return max(0, len(bar_energies) - 16)  # Default to last 16 bars
