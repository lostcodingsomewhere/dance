"""Tempo estimation shared by the full-mix and per-stem analysis stages.

Two jobs:

- :func:`estimate_bpm` — one precise tempo for a whole track. Essentia's
  ``RhythmExtractor2013`` is the primary detector; its tick grid is refined
  with a robust linear fit. When Essentia is missing or fails, librosa's beat
  tracker supplies beat times and the tempo comes from the same fit — never
  from librosa's own ``tempo`` value, which is quantised to onset-envelope lag
  bins (at 44.1 kHz / hop 512 the only values near 124 BPM are 123.05 and
  126.05, so every track in that neighbourhood collapsed onto two numbers).

- :func:`tempo_coherence` — how strongly one stem's rhythm sits on the
  track's beat grid, in ``[0, 1]``. Stems are sample-aligned with the mix, so
  their tempo *is* the track tempo; what varies per stem is whether that stem
  carries the pulse (drums) or floats over it (pads, vocals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dance.pipeline.utils.audio import normalize_bpm

logger = logging.getLogger(__name__)

try:
    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
except ImportError:  # pragma: no cover — essentia is a hard dependency
    es = None
    ESSENTIA_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover — librosa is a hard dependency
    librosa = None
    LIBROSA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Upper bound of ``BeatTrackerMultiFeature``'s confidence output. Essentia's
#: algorithm reference documents the output as "confidence of the beat tracker
#: [0, 5.32]" and grades it [0, 1) very low, [1, 1.5] low, (1.5, 3.5] good,
#: (3.5, 5.32] excellent (Zapata et al., ISMIR 2012). ``RhythmExtractor2013``
#: with ``method="multifeature"`` passes that value through unchanged.
#: https://essentia.upf.edu/reference/std_BeatTrackerMultiFeature.html
ESSENTIA_MAX_CONFIDENCE = 5.32

#: Essentia's rhythm extractor only works correctly at 44.1 kHz (its docs say
#: so explicitly); other rates are resampled before the call.
ESSENTIA_SAMPLE_RATE = 44100

#: Ceiling for the librosa fallback's confidence. The fallback is precise when
#: the beat tracker is (the fit removes lag quantisation) but it is a single
#: onset function with no agreement check, so it never claims more than
#: Essentia's "good" band (~0.28-0.66 after normalisation).
LIBROSA_FIT_MAX_CONFIDENCE = 0.4

#: Confidence for the last-resort path: librosa's quantised global tempo,
#: used only when there are too few beats to fit. "Very low" in Essentia's
#: grading.
LIBROSA_TEMPO_ONLY_CONFIDENCE = 0.1

#: A fitted grid replaces Essentia's own BPM only when the two agree this
#: closely — the fit is a precision refinement, never a different tempo.
_FIT_AGREEMENT_REL = 0.01

#: The fit refines Essentia only when this share of its ticks sit on the
#: grid. Chance level is ~12 % (a ±0.06-beat window); real tracks whose
#: breakdown ticks wander land around 0.4-0.6, clean ones above 0.8.
_MIN_FIT_INLIER_RATIO = 0.35

#: A beat is on the grid when it is within this fraction of a beat of it:
#: ±29 ms at 124 BPM — wider than Essentia's ~11.6 ms tick quantisation,
#: well under a 16th note.
_GRID_TOL_BEATS = 0.06

#: Period search range around the starting estimate, and the search's
#: phase-histogram resolution / chunking (memory bound for long tracks).
_PERIOD_SEARCH_REL = 0.04
_PHASE_BINS = 50
_PERIOD_CHUNK = 512

MIN_BEATS_FOR_FIT = 8

#: Peak amplitude below which a signal is treated as silence (-80 dBFS).
_SILENCE_PEAK = 1e-4

#: Essentia's own tempo search range (minTempo 40 .. maxTempo 208, with
#: headroom); anything outside it is a detector failure, not a tempo.
_PLAUSIBLE_BPM = (40.0, 250.0)

# Coherence: onset envelope resolution and the beat multiples it is read at.
COHERENCE_HOP = 512
COHERENCE_BEAT_MULTIPLES = (1, 2, 4)  # beat, half bar, bar
_COHERENCE_LAG_TOL = 0.03  # ±3 % search window around each lag
_COHERENCE_DETREND_BEATS = 0.5  # moving average removed before correlating


# ---------------------------------------------------------------------------
# Beat-grid fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeatGridFit:
    """A straight line through beat times: ``time = offset + index * period``."""

    bpm: float
    period_s: float
    offset_s: float
    n_beats: int
    inlier_ratio: float  # share of all beats within ±0.06 beat of the grid


def fit_beat_grid(
    beat_times: np.ndarray | list[float], *, approx_bpm: float | None = None
) -> BeatGridFit | None:
    """Fit one constant-tempo grid ``time = offset + k * period`` to beat times.

    Beat trackers wander: through a breakdown Essentia's multifeature ticks
    can run at a different rate for a minute before relocking, and librosa
    drops or adds beats. A line through *all* ticks — or one that assumes
    consecutive ticks are consecutive beats — gets dragged off by such a
    section (on real tracks that cost 0.1-0.2 BPM, i.e. a third of a second
    of warp drift over six minutes). Instead:

    1. Search periods within ±4 % of the starting estimate (``approx_bpm``,
       else the median inter-beat interval) for the grid that the most ticks
       sit on, within ±0.06 beat. Skipped beats, extra ticks, gaps and
       wandering sections simply don't land on it.
    2. Refine that grid by least squares over its inliers, which also
       averages away the analysis-hop quantisation of the ticks — so the BPM
       is not limited to onset-lag bins.

    ``inlier_ratio`` is the share of all ticks on the final grid. Returns
    ``None`` with fewer than ``MIN_BEATS_FOR_FIT`` beats on the grid.
    """
    t = np.asarray(beat_times, dtype=np.float64).ravel()
    t = np.sort(t[np.isfinite(t)])
    if t.size < MIN_BEATS_FOR_FIT:
        return None
    if approx_bpm is not None and np.isfinite(approx_bpm) and approx_bpm > 0:
        p0 = 60.0 / float(approx_bpm)
    else:
        intervals = np.diff(t)
        intervals = intervals[intervals > 0]
        if intervals.size == 0:
            return None
        p0 = float(np.median(intervals))

    rel = t - t[0]
    span_beats = max(float(rel[-1]) / p0, 1.0)
    # Fine enough that the grid drifts by under a quarter of the tolerance
    # across the whole track between neighbouring candidates.
    step = 0.25 * _GRID_TOL_BEATS * p0 / span_beats
    periods = np.arange(p0 * (1 - _PERIOD_SEARCH_REL), p0 * (1 + _PERIOD_SEARCH_REL), step)
    period, phase = _densest_grid(rel, periods)
    offset = float(t[0]) + phase * period

    # Least-squares refinement over the grid's inliers (re-selected twice as
    # the grid sharpens).
    for _ in range(3):
        k = np.round((t - offset) / period)
        on_grid = np.abs(t - (offset + k * period)) <= _GRID_TOL_BEATS * period
        if int(on_grid.sum()) < MIN_BEATS_FOR_FIT or np.unique(k[on_grid]).size < 2:
            return None
        slope, intercept = np.polyfit(k[on_grid], t[on_grid], 1)
        if not np.isfinite(slope) or slope <= 0:
            return None
        period, offset = float(slope), float(intercept)

    k = np.round((t - offset) / period)
    on_grid = np.abs(t - (offset + k * period)) <= _GRID_TOL_BEATS * period
    return BeatGridFit(
        bpm=60.0 / period,
        period_s=period,
        offset_s=offset,
        n_beats=int(on_grid.sum()),
        inlier_ratio=float(on_grid.mean()),
    )


def _densest_grid(rel_times: np.ndarray, periods: np.ndarray) -> tuple[float, float]:
    """Among candidate ``periods``, the (period, phase) whose grid holds the
    most beats. ``phase`` is a fraction of the period, relative to the first
    beat. Phases are histogrammed per period and summed over a circular
    window of ±``_GRID_TOL_BEATS``."""
    bins = _PHASE_BINS
    half = max(1, int(round(_GRID_TOL_BEATS * bins)))
    width = 2 * half + 1
    best = (-1, float(periods[0]), 0.0)
    for start in range(0, periods.size, _PERIOD_CHUNK):
        chunk = periods[start : start + _PERIOD_CHUNK]
        phase = np.mod(rel_times[None, :] / chunk[:, None], 1.0)
        idx = np.minimum((phase * bins).astype(np.int64), bins - 1)
        idx += (np.arange(chunk.size) * bins)[:, None]
        hist = np.bincount(idx.ravel(), minlength=chunk.size * bins).reshape(chunk.size, bins)
        wrapped = np.concatenate([hist[:, -half:], hist, hist[:, :half]], axis=1)
        csum = np.concatenate(
            [np.zeros((chunk.size, 1), dtype=np.int64), np.cumsum(wrapped, axis=1)], axis=1
        )
        window = csum[:, width:] - csum[:, :-width]  # (chunk, bins): centred on each bin
        row = int(np.argmax(window.max(axis=1)))
        count = int(window[row].max())
        if count > best[0]:
            col = int(np.argmax(window[row]))
            best = (count, float(chunk[row]), (col + 0.5) / bins)
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Track tempo
# ---------------------------------------------------------------------------


def essentia_confidence(raw: float) -> float:
    """Map Essentia's multifeature confidence (0..5.32) onto ``[0, 1]``."""
    value = float(raw)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value / ESSENTIA_MAX_CONFIDENCE, 0.0, 1.0))


def estimate_bpm(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate one tempo for ``audio`` (mono). Returns ``(bpm, confidence)``.

    ``bpm`` is folded into the house/techno range by
    :func:`~dance.pipeline.utils.audio.normalize_bpm`; ``confidence`` is in
    ``[0, 1]``. Raises ``RuntimeError`` when no tempo can be found at all
    (e.g. silence) — a zero BPM would poison every downstream consumer.
    """
    samples = np.asarray(audio)
    if samples.size == 0 or not np.any(np.abs(samples) > _SILENCE_PEAK):
        # Essentia happily reports ~369 BPM at 0.88 confidence on digital
        # silence, so this has to be checked up front.
        raise RuntimeError("audio is silent — no tempo to detect")
    if ESSENTIA_AVAILABLE:
        try:
            return _essentia_bpm(audio, sr)
        except Exception as exc:  # noqa: BLE001 — any essentia failure → fallback
            logger.warning(
                "Essentia tempo detection failed (%s: %s); falling back to a "
                "beat-grid fit on librosa beats",
                type(exc).__name__,
                exc,
            )
    if LIBROSA_AVAILABLE:
        return _librosa_bpm(audio, sr)
    raise RuntimeError("No tempo detector available (need essentia or librosa)")


def _essentia_bpm(audio: np.ndarray, sr: int) -> tuple[float, float]:
    signal = np.ascontiguousarray(audio, dtype=np.float32)
    if sr != ESSENTIA_SAMPLE_RATE:
        if not LIBROSA_AVAILABLE:
            raise RuntimeError(f"essentia needs {ESSENTIA_SAMPLE_RATE} Hz audio, got {sr}")
        signal = np.ascontiguousarray(
            librosa.resample(signal, orig_sr=sr, target_sr=ESSENTIA_SAMPLE_RATE),
            dtype=np.float32,
        )
    extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, ticks, confidence, _estimates, _intervals = extractor(signal)
    bpm = float(bpm)
    if not np.isfinite(bpm) or not _PLAUSIBLE_BPM[0] <= bpm <= _PLAUSIBLE_BPM[1]:
        raise RuntimeError(f"essentia returned an implausible tempo (bpm={bpm})")

    fit = fit_beat_grid(ticks, approx_bpm=bpm)
    if (
        fit is not None
        and fit.inlier_ratio >= _MIN_FIT_INLIER_RATIO
        and abs(fit.bpm - bpm) <= _FIT_AGREEMENT_REL * bpm
    ):
        logger.debug(
            "essentia bpm %.4f refined by tick-grid fit to %.4f (%.0f%% of ticks on grid)",
            bpm,
            fit.bpm,
            100 * fit.inlier_ratio,
        )
        bpm = fit.bpm
    else:
        logger.debug("essentia bpm %.4f kept; tick-grid fit rejected: %s", bpm, fit)
    return normalize_bpm(bpm), essentia_confidence(confidence)


def _librosa_bpm(audio: np.ndarray, sr: int) -> tuple[float, float]:
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    fit = fit_beat_grid(beat_times)
    if fit is not None and _PLAUSIBLE_BPM[0] <= fit.bpm <= _PLAUSIBLE_BPM[1]:
        confidence = LIBROSA_FIT_MAX_CONFIDENCE * fit.inlier_ratio
        return normalize_bpm(fit.bpm), float(confidence)

    coarse = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if not np.isfinite(coarse) or not _PLAUSIBLE_BPM[0] <= coarse <= _PLAUSIBLE_BPM[1]:
        raise RuntimeError("no tempo detected (no usable beats)")
    logger.warning(
        "Only %d beats found; using librosa's quantised tempo %.2f BPM with very low confidence",
        len(beat_times),
        coarse,
    )
    return normalize_bpm(coarse), LIBROSA_TEMPO_ONLY_CONFIDENCE


# ---------------------------------------------------------------------------
# Per-stem tempo coherence
# ---------------------------------------------------------------------------


def tempo_coherence(audio: np.ndarray, sr: int, bpm: float | None) -> float:
    """How strongly ``audio`` pulses on a ``bpm`` grid, in ``[0, 1]``.

    The onset-strength envelope is lightly smoothed (so ±1-frame onset jitter
    doesn't split a peak), detrended with a half-beat moving average (so slow
    swells and fades don't read as periodicity), then autocorrelated. The
    score is the mean normalised autocorrelation peak at 1, 2 and 4 beats —
    a four-on-the-floor kick scores ~0.95, a half-time groove ~0.6, a
    sustained pad or noise well under 0.3. Silence scores 0.
    """
    if bpm is None or not np.isfinite(bpm) or bpm <= 0:
        return 0.0
    if audio.size == 0 or not np.any(audio):
        return 0.0
    if not LIBROSA_AVAILABLE:  # pragma: no cover — librosa is a hard dependency
        raise RuntimeError("librosa is required for tempo coherence")

    env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=COHERENCE_HOP)
    fps = sr / COHERENCE_HOP
    beat_lag = 60.0 / float(bpm) * fps

    kernel = np.array([0.5, 1.0, 0.5])
    env = np.convolve(env, kernel / kernel.sum(), mode="same")
    win = max(3, int(round(_COHERENCE_DETREND_BEATS * beat_lag)))
    x = env - np.convolve(env, np.ones(win) / win, mode="same")
    x = x - x.mean()
    n = x.size
    if n < 4 or not np.any(x):
        return 0.0

    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    acf = acf / (n - np.arange(n))  # unbiased: long lags aren't penalised
    if acf[0] <= 0:
        return 0.0
    acf = acf / acf[0]

    peaks: list[float] = []
    for k in COHERENCE_BEAT_MULTIPLES:
        lag = k * beat_lag
        lo = int(np.floor(lag * (1.0 - _COHERENCE_LAG_TOL)))
        hi = int(np.ceil(lag * (1.0 + _COHERENCE_LAG_TOL)))
        if lo < 1 or hi >= n:
            continue
        peaks.append(float(acf[lo : hi + 1].max()))
    if not peaks:
        return 0.0
    return float(np.clip(np.mean(peaks), 0.0, 1.0))
