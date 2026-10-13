"""Deterministic windowing + pooling for CLAP audio embeddings.

CLAP's feature extractor (``truncation="rand_trunc"`` for
laion/clap-htsat-unfused) takes ONE RANDOM crop of any input longer than its
``max_length`` (10 s at 48 kHz). Feeding it a whole track therefore embeds an
arbitrary 10 s slice that changes on every run. Here every audio is cut into
windows of exactly ``max_length`` samples — so the extractor never crops —
and the embedding is the mean over a deterministic, evenly spread subset of
the non-silent ones.

Pure numpy; no model imports, so it is unit-testable on its own.
"""

from __future__ import annotations

import math

import numpy as np


def window_starts(n_samples: int, window: int) -> list[int]:
    """Start offsets of the candidate windows for an audio of ``n_samples``.

    ``ceil(n / window)`` windows of ``window`` samples, the first starting at
    0 and the last ending exactly at ``n`` — so every sample is covered and
    the tail is handled without padding or cropping. Overlap is spread evenly
    between neighbours. Audio no longer than one window gets one window at 0
    (padded later by :func:`extract_windows`).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if n_samples <= 0:
        return []
    if n_samples <= window:
        return [0]
    k = -(-n_samples // window)  # ceil; >= 2 here
    span = n_samples - window
    return [(i * span) // (k - 1) for i in range(k)]


def window_rms_db(audio: np.ndarray, starts: list[int], window: int) -> np.ndarray:
    """RMS level (dBFS) of each window. Digital silence is ``-inf``."""
    levels = np.empty(len(starts), dtype=np.float64)
    for i, start in enumerate(starts):
        chunk = audio[start : start + window]
        if chunk.size == 0:
            levels[i] = -math.inf
            continue
        power = float(np.mean(np.square(chunk, dtype=np.float64)))
        levels[i] = 10.0 * math.log10(power) if power > 0.0 else -math.inf
    return levels


def spread(items: list[int], k: int) -> list[int]:
    """Pick ``k`` items evenly from ``items``, keeping order.

    Takes the centre of each of ``k`` equal segments, so a cap of 1 picks the
    middle and the first/last items (often DJ-tool intros/outros) are not
    over-represented. Returns ``items`` unchanged when there are ``<= k``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    m = len(items)
    if m <= k:
        return list(items)
    return [items[((2 * i + 1) * m) // (2 * k)] for i in range(k)]


def select_windows(
    audio: np.ndarray,
    window: int,
    *,
    max_windows: int,
    silence_db: float,
) -> list[int]:
    """Start offsets of the windows to embed, in time order.

    Candidates from :func:`window_starts`; windows whose RMS is below
    ``silence_db`` are dropped; at most ``max_windows`` are kept, evenly
    spread over what remains. Empty when the audio is empty or entirely below
    the threshold — the caller stores no embedding in that case.
    """
    starts = window_starts(int(audio.shape[0]), window)
    if not starts:
        return []
    levels = window_rms_db(audio, starts, window)
    live = [s for s, level in zip(starts, levels) if level >= silence_db]
    return spread(live, max_windows)


def extract_windows(audio: np.ndarray, starts: list[int], window: int) -> list[np.ndarray]:
    """Slice the windows at ``starts``, each exactly ``window`` samples long.

    Only an audio shorter than one window yields a short slice; it is
    repeat-padded the way CLAP's own extractor does (``padding="repeatpad"``:
    tile whole copies, zero-fill the rest), so the extractor sees exactly
    ``max_length`` samples every time and neither crops nor pads.
    """
    out: list[np.ndarray] = []
    for start in starts:
        chunk = np.asarray(audio[start : start + window], dtype=np.float32)
        n = chunk.shape[0]
        if n == 0:
            raise ValueError("empty window")
        if n < window:
            chunk = np.tile(chunk, window // n)
            chunk = np.pad(chunk, (0, window - chunk.shape[0]))
        out.append(chunk)
    return out


def pool_embeddings(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise each row, mean-pool, L2-normalise the mean (float32)."""
    mat = np.asarray(vectors, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise ValueError(f"expected a non-empty (n, dim) matrix, got shape {mat.shape}")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mean = (mat / norms).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        raise ValueError("window embeddings cancel out; cannot pool")
    pooled: np.ndarray = (mean / norm).astype(np.float32)
    return pooled
