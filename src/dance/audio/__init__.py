"""Audio utilities for the Dance backend.

Currently exposes only waveform-peak computation used by the API to feed
the companion app's waveform renderer. Audio analysis lives in
``dance.pipeline.stages`` and isn't re-exported here.
"""

from dance.audio.waveform import compute_peaks, get_or_compute_peaks

__all__ = ["compute_peaks", "get_or_compute_peaks"]
