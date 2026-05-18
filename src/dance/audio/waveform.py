"""Waveform-peak extraction for the companion app's audio renderer.

The frontend can't read the user's audio files directly — they live on disk
under ``library_dir`` (full tracks) and ``stems_dir`` (Demucs output). So the
backend pre-computes a tiny amplitude envelope (default 200 floats in
``[0, 1]``) and serves it as JSON.

Peaks are cached as a sidecar file (``<audio>.waveform.json``) next to the
source audio. The cache key is ``num_peaks`` + a schema ``version`` — change
the shape, bump the version, and old caches are silently regenerated. Writes
are atomic (tmp file + ``os.replace``), and any cache I/O error is non-fatal
(we just fall back to recomputing on the next call).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# Sidecar schema version. Bump when the cache payload shape changes so old
# caches get regenerated automatically.
_CACHE_VERSION = 1


def compute_peaks(audio_path: Path, num_peaks: int = 200) -> tuple[list[float], float]:
    """Decimate an audio file into ``num_peaks`` amplitude maxima.

    Loads the file as mono via librosa, takes the absolute value of the
    samples, splits them into ``num_peaks`` equal-width windows, and takes
    the max per window. Peaks are normalized to ``[0, 1]`` so the
    frontend can render them at any pixel height without knowing absolute
    amplitudes.

    Returns ``(peaks, duration_seconds)``. ``duration_seconds`` is derived
    from the loaded sample count, so it reflects what was actually
    decoded (not the file's metadata header).
    """
    if num_peaks <= 0:
        raise ValueError(f"num_peaks must be positive, got {num_peaks}")

    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if audio.size == 0:
        return [0.0] * num_peaks, 0.0

    duration_seconds = float(audio.size / float(sr))

    abs_audio = np.abs(audio)

    # If the file is shorter than ``num_peaks`` samples (pathological), just
    # zero-pad up to num_peaks so the output shape is stable.
    if abs_audio.size < num_peaks:
        padded = np.zeros(num_peaks, dtype=abs_audio.dtype)
        padded[: abs_audio.size] = abs_audio
        peaks = padded
    else:
        # Trim to an exact multiple of num_peaks so reshape works cleanly,
        # then take the max over each window.
        window_size = abs_audio.size // num_peaks
        trimmed = abs_audio[: window_size * num_peaks]
        peaks = trimmed.reshape(num_peaks, window_size).max(axis=1)

    peak_max = float(peaks.max()) if peaks.size else 0.0
    peaks = peaks / max(peak_max, 1e-9)

    return [float(p) for p in peaks], duration_seconds


def _cache_path_for(audio_path: Path) -> Path:
    """Sidecar location: ``<audio_path>.waveform.json`` (preserves extension)."""
    return audio_path.with_name(audio_path.name + ".waveform.json")


def _read_cache(cache_path: Path, num_peaks: int) -> tuple[list[float], float] | None:
    """Return cached peaks if the file exists and matches num_peaks+version."""
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _CACHE_VERSION:
        return None
    if payload.get("num_peaks") != num_peaks:
        return None
    peaks = payload.get("peaks")
    duration = payload.get("duration_seconds")
    if not isinstance(peaks, list) or not isinstance(duration, (int, float)):
        return None
    if len(peaks) != num_peaks:
        return None
    return [float(p) for p in peaks], float(duration)


def _write_cache_atomic(
    cache_path: Path, peaks: list[float], duration_seconds: float, num_peaks: int
) -> None:
    """Write the sidecar cache atomically. Failures are logged, not raised."""
    payload = {
        "peaks": peaks,
        "duration_seconds": duration_seconds,
        "num_peaks": num_peaks,
        "version": _CACHE_VERSION,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a tmp file in the same dir, then rename.
        # Same-dir is important because os.replace is only atomic on the
        # same filesystem.
        fd, tmp_name = tempfile.mkstemp(
            prefix=".waveform-", suffix=".tmp", dir=str(cache_path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp_name, cache_path)
        except Exception:
            # Make sure we don't leave the tmp file behind on failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError as exc:
        # Read-only fs, no permission, disk full — fall through and let the
        # caller return the freshly computed value. The next call will retry.
        logger.warning("Could not write waveform cache to %s: %s", cache_path, exc)


def get_or_compute_peaks(audio_path: Path, num_peaks: int = 200) -> tuple[list[float], float]:
    """Return peaks, hitting the sidecar cache when possible.

    Cache miss / mismatch / read failure → compute, then best-effort write
    the sidecar. Cache writes never raise; if they fail (read-only fs,
    full disk) the caller still gets the freshly computed peaks back.
    """
    cache_path = _cache_path_for(audio_path)
    cached = _read_cache(cache_path, num_peaks)
    if cached is not None:
        return cached

    peaks, duration_seconds = compute_peaks(audio_path, num_peaks=num_peaks)
    _write_cache_atomic(cache_path, peaks, duration_seconds, num_peaks)
    return peaks, duration_seconds
