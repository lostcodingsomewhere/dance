"""Confirm two files are the same recording by listening to them.

Metadata proposes, audio disposes.

:mod:`dance.recommender.dedup` finds *candidate* duplicates from title and
duration. That is not enough to hide a track. Measured over the 125 candidates
this library produces, an alignment-tolerant chroma comparison found **111
genuinely identical (median 0.995) and 14 that are not** — so a metadata-only
cleanup would have hidden 14 real tracks, including a "Take Off" pair at 0.822
and a "LA FAMA" pair at 0.696 that share a title and a duration but are
different audio.

Two traps this module exists to avoid:

* **CLAP embeddings are the wrong tool.** Cosine over the stored full-mix
  embeddings ranked the *most* identical pair in the library (#218/#354,
  chroma 0.994) as the *least* similar (cosine 0.379). CLAP is semantic and
  window-sensitive; it does not answer "is this the same recording".
* **Fixed-offset comparison lies.** Copies from different sources carry
  different leading silence, so sampling "the middle" of each lands on
  different bars. The same "Take Off" pair scores 0.715 unaligned and 0.822
  aligned. Every comparison here searches for the offset first.

Source audio is the right thing to compare: stems are derived from it, so two
files that are the same recording yield equivalent stems.
"""

from __future__ import annotations

from dataclasses import dataclass

# Verified same-recording threshold. The measured distribution is strongly
# bimodal — 111 pairs at >=0.95 (median 0.995) and a thin 14-pair tail spread
# from 0.696 to 0.947 — so the cut sits in real empty space, not on a guess.
SAME_RECORDING_SIMILARITY = 0.95

_SR = 22050
_HOP = 2048
# Seconds of audio compared. Long enough to span a section change, short
# enough that verifying a whole library stays a couple of minutes.
_WINDOW_S = 60.0
# Misalignment searched in each direction. Different sources differ by a few
# seconds of leading silence; beyond this the pair is reported unverified
# rather than silently scored low.
_MAX_SHIFT_S = 12.0


@dataclass
class AudioMatch:
    """Outcome of comparing two files."""

    similarity: float | None
    offset_s: float | None
    error: str | None = None

    @property
    def same_recording(self) -> bool:
        return self.similarity is not None and self.similarity >= SAME_RECORDING_SIMILARITY

    @property
    def at_search_limit(self) -> bool:
        """True when the best offset sits at the edge of the search window, so
        the real alignment may be further out and the score understated."""
        return self.offset_s is not None and abs(self.offset_s) >= _MAX_SHIFT_S - 0.5


def compare_recordings(path_a: str, path_b: str, dur_a: float, dur_b: float) -> AudioMatch:
    """Aligned chroma similarity between two audio files.

    Returns ``similarity`` in roughly ``[0, 1]`` — ~1.0 for the same recording
    at a different bitrate, and materially lower for different audio — plus the
    offset that produced it. ``error`` is set (and similarity ``None``) when a
    file could not be decoded; callers must treat that as "not verified", never
    as a match.
    """
    # librosa pulls in a large scientific stack; keep it out of API import time.
    import numpy as np

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return AudioMatch(None, None, f"librosa unavailable: {exc}")

    def chroma(path: str, offset: float, duration: float):
        y, _ = librosa.load(path, sr=_SR, mono=True, offset=offset, duration=duration)
        if y.size < _SR:
            return None
        c = librosa.feature.chroma_cqt(y=y, sr=_SR, hop_length=_HOP)
        return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)

    try:
        # Start a quarter in: past intros, before outros, on both files.
        start = max(0.0, min(dur_a, dur_b) * 0.25)
        ca = chroma(path_a, start, _WINDOW_S)
        cb = chroma(path_b, max(0.0, start - _MAX_SHIFT_S), _WINDOW_S + 2 * _MAX_SHIFT_S)
    except Exception as exc:  # noqa: BLE001 — any decode failure is "unverified"
        return AudioMatch(None, None, f"decode failed: {exc}")

    if ca is None or cb is None:
        return AudioMatch(None, None, "too short to compare")

    frames = int(_MAX_SHIFT_S * _SR / _HOP)
    n = ca.shape[1]
    best, best_off = -1.0, 0
    for off in range(0, max(1, min(2 * frames, cb.shape[1] - n))):
        seg = cb[:, off : off + n]
        if seg.shape[1] < n:
            break
        sim = float(np.mean(np.sum(ca * seg, axis=0)))
        if sim > best:
            best, best_off = sim, off
    if best < 0:
        return AudioMatch(None, None, "no overlap to compare")
    return AudioMatch(best, (best_off - frames) * _HOP / _SR)
