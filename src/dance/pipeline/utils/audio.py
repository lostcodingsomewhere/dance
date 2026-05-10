"""Shared audio-analysis helpers.

Small, generic primitives used by both full-mix analysis and per-stem
analysis. Keep this thin — only put things here that are actually shared.
"""

from __future__ import annotations

import numpy as np


# Pitch class names in semitone order. Indexed by chroma bin (0 = C).
PITCH_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)


# ---------------------------------------------------------------------------
# RMS energy
# ---------------------------------------------------------------------------


def aggregate_rms(rms: np.ndarray) -> tuple[float, float]:
    """Return (overall, peak) summary from an RMS curve.

    Scales calibrated for electronic music — same normalisation used by the
    full-mix :class:`AnalysisStage`. ``overall`` divides by 0.2, ``peak`` (p95)
    by 0.3, both clipped to [0, 1].
    """
    if rms.size == 0:
        return 0.0, 0.0
    overall = float(np.clip(np.mean(rms) / 0.2, 0.0, 1.0))
    peak = float(np.clip(np.percentile(rms, 95) / 0.3, 0.0, 1.0))
    return overall, peak


# ---------------------------------------------------------------------------
# BPM normalisation
# ---------------------------------------------------------------------------


def normalize_bpm(bpm: float) -> float:
    """Pull a detected BPM into a house/techno-typical range (118-145).

    If the value looks like half/double time, correct it; otherwise leave it.
    """
    if bpm < 90:
        return bpm * 2
    if bpm > 180:
        return bpm / 2
    if bpm < 110 and 118 <= bpm * 2 <= 145:
        return bpm * 2
    return bpm


# ---------------------------------------------------------------------------
# Chroma / key detection
# ---------------------------------------------------------------------------


def detect_key_from_chroma(
    chroma: np.ndarray,
    *,
    mode: str | None = None,
) -> tuple[str, str, float]:
    """Pick the dominant pitch class from a chroma matrix.

    Args:
        chroma: (12, T) chroma matrix from librosa.
        mode: If "minor" or "major", force that mode. Otherwise guess from
            whether the 3rd minor or 3rd major interval has more energy.

    Returns:
        ``(pitch_name, mode, confidence)`` where confidence is the dominant
        bin's share of the chroma sum (0-1).
    """
    chroma_mean = chroma.mean(axis=1)
    pitch_idx = int(chroma_mean.argmax())
    pitch = PITCH_NAMES[pitch_idx]

    total = float(chroma_mean.sum())
    confidence = float(chroma_mean[pitch_idx] / total) if total > 0 else 0.0

    if mode is None:
        third_minor = chroma_mean[(pitch_idx + 3) % 12]
        third_major = chroma_mean[(pitch_idx + 4) % 12]
        mode = "minor" if third_minor > third_major else "major"

    return pitch, mode, confidence
