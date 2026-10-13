"""Beat-grid differential metrics — the measurement gate for grid work.

Every beat-grid change (a new tempo estimator, a downbeat detector, a
constant-tempo snap) is judged here against a REFERENCE grid, and Ableton is
the reference (CLAUDE.md rule 3: what Live plays is what the DJ hears). The
method follows djbooth's analyzer differential:

1. **Align** the two beat sequences with a monotonic dynamic program. Each
   unmatched beat costs ``0.45 x`` the reference's median beat period; each
   matched pair costs its absolute time difference. A pair is therefore only
   matched when that beats leaving both beats unmatched — i.e. when the two
   beats are within ``0.9`` periods of each other — and half/double-time
   grids degrade gracefully into "every other beat unmatched".
2. **Score** the aligned pairs: F1 at 70 ms and 100 ms collars, median / p95
   absolute timing error, signed phase offset, linear drift per 100 beats,
   and — when downbeats are known on both sides — bar-phase accuracy.
3. **Gate**: :func:`find_regressions` compares a run against a saved
   baseline and names every track whose metrics got worse. The discipline is
   djbooth's: a grid change ships only if no previously-passing track
   regresses.

Everything in this module is pure (numbers in, numbers out). Reading grids
from the DB, from ``.als`` files or from Live over OSC lives elsewhere —
see ``scripts/grid_diff.py``.

Times are seconds unless a name says ``_ms``. "Estimate" is the grid under
test (dance's ``beats`` table); "reference" is the grid it is judged against.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Gap cost as a fraction of the reference's median beat period (djbooth).
GAP_COST_FRACTION = 0.45
# Timing collars for the F1 scores, in milliseconds.
F1_TOLERANCES_MS: tuple[float, float] = (70.0, 100.0)
# Median period assumed when a grid has fewer than two beats.
_FALLBACK_PERIOD_S = 0.5
# Octave relations recognised by :func:`tempo_octave`.
_OCTAVE_FACTORS: tuple[tuple[str, float], ...] = (
    ("1x", 1.0),
    ("2x", 2.0),
    ("0.5x", 0.5),
    ("1.5x", 1.5),
    ("0.67x", 2.0 / 3.0),
    ("1.33x", 4.0 / 3.0),
    ("0.75x", 0.75),
)

# A track "passes" when its grid is usable for beatmatching against Live.
PASS_THRESHOLDS: dict[str, float] = {
    "min_f1_70": 0.90,
    "max_abs_bpm_error_pct": 0.5,
}

# How much a metric may worsen before the gate calls it a regression. Small
# slack absorbs ms rounding in the DB (``beats.position_ms`` is an int) so a
# re-run of identical code never trips the gate.
REGRESSION_TOLERANCES: dict[str, float] = {
    "f1_70": 0.005,
    "f1_100": 0.005,
    "median_abs_error_ms": 1.0,
    "p95_abs_error_ms": 2.0,
    "abs_bpm_error_pct": 0.05,
    "abs_grid_bpm_error_pct": 0.05,
    "abs_drift_ms_per_100_beats": 2.0,
    "bar_phase_accuracy": 0.01,
    "downbeat_f1_70": 0.01,
}
# Metrics where a bigger number is better. Everything else: smaller is better.
_HIGHER_IS_BETTER = frozenset({"f1_70", "f1_100", "bar_phase_accuracy", "downbeat_f1_70"})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clean(times: Iterable[float]) -> list[float]:
    """Finite values, sorted, exact duplicates dropped."""
    return sorted({float(t) for t in times if math.isfinite(float(t))})


def median_period(times: Sequence[float]) -> float:
    """Median inter-beat interval in seconds (``0.5`` for < 2 beats)."""
    if len(times) < 2:
        return _FALLBACK_PERIOD_S
    diffs = np.diff(np.asarray(times, dtype=float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return _FALLBACK_PERIOD_S
    return float(np.median(diffs))


def grid_bpm(times: Sequence[float]) -> float | None:
    """Average tempo of a beat grid (None for < 2 beats).

    Not ``60 / median IOI``: a frame-quantised tracker (librosa at hop 512)
    alternates 41- and 42-frame beats, and the median picks one of them —
    a 0.5 % tempo error that compounds to half a beat per 100 beats. Instead
    each beat gets an integer index (a gap of ~2 median periods counts as a
    dropped beat, not a slow one) and the period is the least-squares slope
    of time against index.
    """
    if len(times) < 2:
        return None
    t = np.asarray(times, dtype=float)
    p = median_period(times)
    steps = np.maximum(1, np.rint(np.diff(t) / p))
    k = np.concatenate([[0.0], np.cumsum(steps)])
    kc = k - k.mean()
    slope = float(np.dot(kc, t - t.mean()) / np.dot(kc, kc))
    return 60.0 / slope if slope > 0 else None


def bpm_error_pct(estimate_bpm: float | None, reference_bpm: float | None) -> float | None:
    """Signed tempo error ``(est - ref) / ref`` in percent."""
    if estimate_bpm is None or reference_bpm is None or reference_bpm <= 0:
        return None
    return (float(estimate_bpm) - float(reference_bpm)) / float(reference_bpm) * 100.0


def tempo_octave(
    estimate_bpm: float | None, reference_bpm: float | None, tolerance_pct: float = 4.0
) -> str | None:
    """Name the metrical relation between two tempi, or ``"other"``.

    ``"2x"`` means the estimate is double-time, ``"0.5x"`` half-time. The
    common DJ-software failure is an octave error, which a plain percentage
    hides behind a scary 100%; this label says which kind it is.
    """
    if estimate_bpm is None or reference_bpm is None or reference_bpm <= 0:
        return None
    for label, factor in _OCTAVE_FACTORS:
        err = abs(float(estimate_bpm) / (float(reference_bpm) * factor) - 1.0) * 100.0
        if err <= tolerance_pct:
            return label
    return "other"


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """Monotonic matching between a reference and an estimated grid."""

    matches: list[tuple[int, int]]  # (reference_index, estimate_index), increasing
    reference_unmatched: int
    estimate_unmatched: int
    gap_cost_s: float


def align_beats(
    reference: Sequence[float],
    estimate: Sequence[float],
    *,
    gap_cost_s: float | None = None,
) -> Alignment:
    """Monotonic minimum-cost alignment of two sorted beat-time sequences.

    Cost model (djbooth's ``align_beats``): matching reference beat ``i`` to
    estimate beat ``j`` costs ``|r_i - e_j|``; leaving any beat unmatched
    costs ``gap``. Total cost is ``gap * (n + m) - sum(2 * gap - |r_i - e_j|)``
    over matched pairs, so minimising cost is the same as finding the
    maximum-weight monotone matching with pair weight ``2 * gap - |d|`` —
    and only pairs closer than ``2 * gap`` can have positive weight. That
    turns the full ``n x m`` table into a sparse chain problem over the few
    pairs inside that window, solved with a Fenwick prefix-max in
    ``O(P log m)``. The optimum is identical to the dense DP.

    Both inputs must be sorted ascending (use the values you got back from
    :func:`compare_grids` / the readers, which sort).
    """
    ref = [float(t) for t in reference]
    est = [float(t) for t in estimate]
    gap = float(gap_cost_s) if gap_cost_s is not None else GAP_COST_FRACTION * median_period(ref)
    if not ref or not est or gap <= 0:
        return Alignment([], len(ref), len(est), gap)

    window = 2.0 * gap
    # Candidate pairs, ordered by reference index ascending and estimate
    # index DESCENDING within one reference beat — so a prefix query over
    # estimate indices < j never sees a pair that shares this reference beat.
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    pairs_w: list[float] = []
    for i, t in enumerate(ref):
        lo = bisect.bisect_left(est, t - window)
        hi = bisect.bisect_right(est, t + window)
        for j in range(hi - 1, lo - 1, -1):
            w = window - abs(est[j] - t)
            if w > 0:
                pairs_i.append(i)
                pairs_j.append(j)
                pairs_w.append(w)

    m = len(est)
    tree_val = [0.0] * (m + 1)
    tree_idx = [-1] * (m + 1)
    best_val = [0.0] * len(pairs_w)
    prev = [-1] * len(pairs_w)

    def query(j: int) -> tuple[float, int]:
        """Best chain value over pairs whose estimate index is < j."""
        val, idx = 0.0, -1
        k = j  # 1-based prefix [1, j] == 0-based estimate indices [0, j-1]
        while k > 0:
            if tree_val[k] > val:
                val, idx = tree_val[k], tree_idx[k]
            k -= k & -k
        return val, idx

    def update(j: int, val: float, idx: int) -> None:
        k = j + 1
        while k <= m:
            if val > tree_val[k]:
                tree_val[k] = val
                tree_idx[k] = idx
            k += k & -k

    overall_val, overall_idx = 0.0, -1
    for p in range(len(pairs_w)):
        base, base_idx = query(pairs_j[p])
        best_val[p] = base + pairs_w[p]
        prev[p] = base_idx
        update(pairs_j[p], best_val[p], p)
        if best_val[p] > overall_val:
            overall_val, overall_idx = best_val[p], p

    matches: list[tuple[int, int]] = []
    p = overall_idx
    while p >= 0:
        matches.append((pairs_i[p], pairs_j[p]))
        p = prev[p]
    matches.reverse()
    return Alignment(
        matches=matches,
        reference_unmatched=len(ref) - len(matches),
        estimate_unmatched=len(est) - len(matches),
        gap_cost_s=gap,
    )


# ---------------------------------------------------------------------------
# Scores over an alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRate:
    tolerance_ms: float
    matches: int
    precision: float
    recall: float
    f1: float


def match_rate(
    abs_errors_s: Sequence[float],
    reference_count: int,
    estimate_count: int,
    tolerance_ms: float,
) -> MatchRate:
    """Precision / recall / F1 of aligned pairs inside a timing collar."""
    tol = tolerance_ms / 1000.0 + 1e-9
    hits = sum(1 for e in abs_errors_s if e <= tol)
    precision = hits / max(estimate_count, 1)
    recall = hits / max(reference_count, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return MatchRate(tolerance_ms, hits, precision, recall, f1)


def drift_ms_per_100_beats(
    reference_indices: Sequence[int], signed_errors_ms: Sequence[float]
) -> float | None:
    """Least-squares slope of signed error vs reference beat index, x100.

    A constant offset has zero drift; a tempo that is slightly wrong grows
    its error linearly with beat count and shows up here. Positive means the
    estimate falls progressively later than the reference.

    Caveat: once the accumulated error passes half a beat the alignment
    re-pairs each estimate beat with the NEXT reference beat, the error turns
    into a sawtooth and this slope collapses towards zero. For grids that far
    apart, ``grid_bpm_error_pct`` is the honest tempo witness.
    """
    if len(reference_indices) < 2 or len(reference_indices) != len(signed_errors_ms):
        return None
    x = np.asarray(reference_indices, dtype=float)
    y = np.asarray(signed_errors_ms, dtype=float)
    xc = x - x.mean()
    var = float(np.dot(xc, xc))
    if var <= 1e-12:
        return None
    return float(np.dot(xc, y - y.mean()) / var * 100.0)


def bar_positions(count: int, downbeats: Sequence[int], beats_per_bar: int = 4) -> list[int]:
    """Beat-in-bar position (0 = downbeat) for each of ``count`` beats.

    ``downbeats`` are indices into the beat list. Each beat counts from the
    most recent downbeat at or before it; beats before the first downbeat
    count backwards from it. Irregular bars (a 3-beat bar, a missed beat)
    resynchronise at the next flagged downbeat.
    """
    if count <= 0:
        return []
    ds = sorted({d for d in downbeats if 0 <= d < count})
    if not ds:
        return []
    out: list[int] = []
    k = 0
    for i in range(count):
        while k + 1 < len(ds) and ds[k + 1] <= i:
            k += 1
        out.append((i - ds[k]) % beats_per_bar)
    return out


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------


@dataclass
class GridComparison:
    """All differential metrics for one (reference, estimate) pair."""

    reference_beats: int
    estimate_beats: int
    matched_beats: int
    reference_unmatched: int
    estimate_unmatched: int
    f1_70: float
    precision_70: float
    recall_70: float
    f1_100: float
    precision_100: float
    recall_100: float
    median_abs_error_ms: float | None
    p95_abs_error_ms: float | None
    mean_abs_error_ms: float | None
    max_abs_error_ms: float | None
    phase_offset_ms: float | None  # median signed (estimate - reference)
    phase_offset_beats: float | None  # same, in reference beat periods
    drift_ms_per_100_beats: float | None
    first_beat_error_ms: float | None  # signed, estimate[0] - reference[0]
    reference_bpm: float | None
    estimate_bpm: float | None
    bpm_error_pct: float | None
    tempo_octave: str | None
    reference_grid_bpm: float | None
    estimate_grid_bpm: float | None
    grid_bpm_error_pct: float | None
    bar_phase_accuracy: float | None = None
    bar_phase_shift: int | None = None  # dominant (estimate - reference) mod bar
    downbeat_f1_70: float | None = None
    downbeat_median_abs_error_ms: float | None = None

    @property
    def abs_bpm_error_pct(self) -> float | None:
        return None if self.bpm_error_pct is None else abs(self.bpm_error_pct)

    @property
    def abs_grid_bpm_error_pct(self) -> float | None:
        e = self.grid_bpm_error_pct
        return None if e is None else abs(e)

    @property
    def abs_drift_ms_per_100_beats(self) -> float | None:
        d = self.drift_ms_per_100_beats
        return None if d is None else abs(d)

    def passes(self, thresholds: Mapping[str, float] = PASS_THRESHOLDS) -> bool:
        if self.f1_70 < thresholds["min_f1_70"]:
            return False
        err = self.abs_bpm_error_pct
        return err is None or err <= thresholds["max_abs_bpm_error_pct"]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["abs_bpm_error_pct"] = self.abs_bpm_error_pct
        d["abs_grid_bpm_error_pct"] = self.abs_grid_bpm_error_pct
        d["abs_drift_ms_per_100_beats"] = self.abs_drift_ms_per_100_beats
        d["passes"] = self.passes()
        return d


def compare_grids(
    reference: Sequence[float],
    estimate: Sequence[float],
    *,
    reference_bpm: float | None = None,
    estimate_bpm: float | None = None,
    reference_downbeats: Sequence[int] | None = None,
    estimate_downbeats: Sequence[int] | None = None,
    beats_per_bar: int = 4,
) -> GridComparison:
    """Compare an estimated beat grid against a reference grid.

    ``reference_bpm`` / ``estimate_bpm`` are the headline tempi (e.g. Live's
    warp tempo vs ``audio_analysis.bpm``); when omitted they fall back to the
    grids' own average tempo (:func:`grid_bpm`). ``*_downbeats`` index the
    (sorted) beat lists; bar-phase metrics are only computed when both are
    given. Unsorted input is sorted; non-finite values and exact duplicates
    are dropped — pass already-clean lists when you also pass downbeat
    indices so the indices still line up.
    """
    ref = _clean(reference)
    est = _clean(estimate)
    if reference_downbeats is not None and ref != [float(t) for t in reference]:
        raise ValueError("reference must be sorted/unique/finite when downbeat indices are given")
    if estimate_downbeats is not None and est != [float(t) for t in estimate]:
        raise ValueError("estimate must be sorted/unique/finite when downbeat indices are given")

    alignment = align_beats(ref, est)
    signed_s = [est[j] - ref[i] for i, j in alignment.matches]
    abs_s = [abs(v) for v in signed_s]
    abs_ms = [v * 1000.0 for v in abs_s]
    signed_ms = [v * 1000.0 for v in signed_s]

    r70 = match_rate(abs_s, len(ref), len(est), F1_TOLERANCES_MS[0])
    r100 = match_rate(abs_s, len(ref), len(est), F1_TOLERANCES_MS[1])

    ref_period = median_period(ref)
    phase_ms = float(np.median(signed_ms)) if signed_ms else None
    ref_grid_bpm = grid_bpm(ref)
    est_grid_bpm = grid_bpm(est)
    ref_bpm = reference_bpm if reference_bpm is not None else ref_grid_bpm
    est_bpm = estimate_bpm if estimate_bpm is not None else est_grid_bpm

    result = GridComparison(
        reference_beats=len(ref),
        estimate_beats=len(est),
        matched_beats=len(alignment.matches),
        reference_unmatched=alignment.reference_unmatched,
        estimate_unmatched=alignment.estimate_unmatched,
        f1_70=r70.f1,
        precision_70=r70.precision,
        recall_70=r70.recall,
        f1_100=r100.f1,
        precision_100=r100.precision,
        recall_100=r100.recall,
        median_abs_error_ms=_percentile(abs_ms, 50),
        p95_abs_error_ms=_percentile(abs_ms, 95),
        mean_abs_error_ms=float(np.mean(abs_ms)) if abs_ms else None,
        max_abs_error_ms=max(abs_ms) if abs_ms else None,
        phase_offset_ms=phase_ms,
        phase_offset_beats=(phase_ms / 1000.0 / ref_period) if phase_ms is not None else None,
        drift_ms_per_100_beats=drift_ms_per_100_beats([i for i, _ in alignment.matches], signed_ms),
        first_beat_error_ms=(est[0] - ref[0]) * 1000.0 if ref and est else None,
        reference_bpm=ref_bpm,
        estimate_bpm=est_bpm,
        bpm_error_pct=bpm_error_pct(est_bpm, ref_bpm),
        tempo_octave=tempo_octave(est_bpm, ref_bpm),
        reference_grid_bpm=ref_grid_bpm,
        estimate_grid_bpm=est_grid_bpm,
        grid_bpm_error_pct=bpm_error_pct(est_grid_bpm, ref_grid_bpm),
    )

    if reference_downbeats is not None and estimate_downbeats is not None:
        ref_pos = bar_positions(len(ref), reference_downbeats, beats_per_bar)
        est_pos = bar_positions(len(est), estimate_downbeats, beats_per_bar)
        if ref_pos and est_pos and alignment.matches:
            shifts = [0] * beats_per_bar
            for i, j in alignment.matches:
                shifts[(est_pos[j] - ref_pos[i]) % beats_per_bar] += 1
            result.bar_phase_accuracy = shifts[0] / len(alignment.matches)
            result.bar_phase_shift = max(range(beats_per_bar), key=lambda s: (shifts[s], -s))
        ref_db = [ref[d] for d in sorted(set(reference_downbeats)) if 0 <= d < len(ref)]
        est_db = [est[d] for d in sorted(set(estimate_downbeats)) if 0 <= d < len(est)]
        if ref_db and est_db:
            # Downbeats are aligned with the BEAT-period gap cost, not the bar
            # period — a downbeat one beat late should not count as matched.
            db_align = align_beats(ref_db, est_db, gap_cost_s=GAP_COST_FRACTION * ref_period)
            db_abs = [abs(est_db[j] - ref_db[i]) for i, j in db_align.matches]
            result.downbeat_f1_70 = match_rate(
                db_abs, len(ref_db), len(est_db), F1_TOLERANCES_MS[0]
            ).f1
            result.downbeat_median_abs_error_ms = (
                float(np.median(db_abs)) * 1000.0 if db_abs else None
            )
    return result


# ---------------------------------------------------------------------------
# Reference grids from Ableton warp markers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarpMarker:
    """One Live warp marker: ``sec`` into the sample is pinned to ``beat``."""

    sec: float
    beat: float


@dataclass(frozen=True)
class ReferenceGrid:
    """Integer-beat grid generated from warp markers."""

    times: list[float]  # seconds into the sample, ascending
    beat_numbers: list[int]  # Live beat index of each time (0 = clip's 1.1.1)
    bpm: float  # tempo of the warped span (beats / seconds between end markers)
    beats_per_bar: int = 4

    @property
    def downbeats(self) -> list[int]:
        """Indices of beats on a Live bar line (beat number divisible by bar)."""
        return [k for k, b in enumerate(self.beat_numbers) if b % self.beats_per_bar == 0]


def clean_warp_markers(markers: Iterable[WarpMarker]) -> list[WarpMarker]:
    """Sort by beat and drop markers that break strict monotonicity."""
    out: list[WarpMarker] = []
    for m in sorted(markers, key=lambda w: (w.beat, w.sec)):
        if not (math.isfinite(m.sec) and math.isfinite(m.beat)):
            continue
        if out and (m.beat <= out[-1].beat or m.sec <= out[-1].sec):
            continue
        out.append(m)
    return out


def beats_from_warp_markers(
    markers: Iterable[WarpMarker],
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    beats_per_bar: int = 4,
) -> ReferenceGrid:
    """Turn a clip's warp markers into the integer-beat grid Live plays.

    Live maps sample time to beat time piecewise-linearly between markers,
    and continues the first / last segment's tempo outside them. Every
    integer beat inside ``[start_sec, end_sec]`` (default: the sample span
    the markers cover, extended down to ``start_sec``) becomes a reference
    beat. Beat numbers divisible by ``beats_per_bar`` are Live's bar lines.
    """
    ms = clean_warp_markers(markers)
    if len(ms) < 2:
        raise ValueError("need at least two monotonic warp markers to define a tempo")
    secs = np.asarray([m.sec for m in ms], dtype=float)
    beats = np.asarray([m.beat for m in ms], dtype=float)
    if end_sec is None:
        end_sec = float(secs[-1])

    def beat_at(sec: float) -> float:
        k = int(np.clip(np.searchsorted(secs, sec) - 1, 0, len(secs) - 2))
        slope = (beats[k + 1] - beats[k]) / (secs[k + 1] - secs[k])
        return float(beats[k] + (sec - secs[k]) * slope)

    eps = 1e-9
    first = math.ceil(beat_at(start_sec) - eps)
    last = math.floor(beat_at(end_sec) + eps)
    numbers = list(range(first, last + 1))
    b = np.asarray(numbers, dtype=float)
    k = np.clip(np.searchsorted(beats, b) - 1, 0, len(beats) - 2)
    slope = (secs[k + 1] - secs[k]) / (beats[k + 1] - beats[k])
    times = secs[k] + (b - beats[k]) * slope
    span_bpm = float((beats[-1] - beats[0]) / (secs[-1] - secs[0]) * 60.0)
    return ReferenceGrid(
        times=[float(t) for t in times],
        beat_numbers=numbers,
        bpm=span_bpm,
        beats_per_bar=beats_per_bar,
    )


# ---------------------------------------------------------------------------
# Non-regression gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regression:
    track_id: str
    metric: str
    baseline: float | bool | None
    current: float | bool | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _worse(metric: str, base: float, cur: float, tol: float) -> bool:
    if metric in _HIGHER_IS_BETTER:
        return cur < base - tol
    return cur > base + tol


def find_regressions(
    baseline: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    *,
    tolerances: Mapping[str, float] = REGRESSION_TOLERANCES,
) -> tuple[list[Regression], list[str]]:
    """Compare per-track metric dicts against a baseline run.

    Both arguments map ``track_id -> result dict`` as written by
    ``scripts/grid_diff.py`` (a result is "ok" when it has metrics, i.e.
    ``result["metrics"]`` is present). Returns ``(regressions, not_compared)``:

    - a track that passed in the baseline and fails now → ``passes``;
    - any tolerance-exceeding worsening of a gated metric → that metric;
    - a track that HAD metrics in the baseline but now has none (its grid
      vanished, its reference stopped matching) → ``grid_missing``;
    - baseline tracks absent from ``current`` entirely are not regressions
      (the run simply didn't select them) — they come back in
      ``not_compared`` so the caller can say so.
    """
    regressions: list[Regression] = []
    not_compared: list[str] = []
    for track_id, base in baseline.items():
        base_m = base.get("metrics")
        if not base_m:
            continue
        if track_id not in current:
            not_compared.append(track_id)
            continue
        cur_m = current[track_id].get("metrics")
        if not cur_m:
            regressions.append(
                Regression(
                    track_id,
                    "grid_missing",
                    True,
                    False,
                    f"had metrics in baseline; now: {current[track_id].get('status', '?')}",
                )
            )
            continue
        if base_m.get("passes") and not cur_m.get("passes"):
            regressions.append(
                Regression(track_id, "passes", True, False, "previously-passing track now fails")
            )
        for metric, tol in tolerances.items():
            b, c = base_m.get(metric), cur_m.get(metric)
            if b is None or c is None:
                continue
            if _worse(metric, float(b), float(c), tol):
                direction = "dropped" if metric in _HIGHER_IS_BETTER else "rose"
                regressions.append(
                    Regression(
                        track_id,
                        metric,
                        float(b),
                        float(c),
                        f"{metric} {direction} {float(b):.4g} -> {float(c):.4g} (tol {tol:g})",
                    )
                )
    return regressions, not_compared


__all__ = [
    "Alignment",
    "F1_TOLERANCES_MS",
    "GAP_COST_FRACTION",
    "GridComparison",
    "MatchRate",
    "PASS_THRESHOLDS",
    "REGRESSION_TOLERANCES",
    "ReferenceGrid",
    "Regression",
    "WarpMarker",
    "align_beats",
    "bar_positions",
    "beats_from_warp_markers",
    "bpm_error_pct",
    "clean_warp_markers",
    "compare_grids",
    "drift_ms_per_100_beats",
    "find_regressions",
    "grid_bpm",
    "match_rate",
    "median_period",
    "tempo_octave",
]
