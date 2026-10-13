"""Beat-grid differential harness: metrics, .als warp reading, the gate.

Covers ``dance.pipeline.utils.grid_metrics`` (pure metrics + baseline gate),
``dance.als.reader`` (warp markers out of a Live Set, round-tripped through
dance's own writer), ``dance.osc.warp_grid`` (the shared OSC reader, against
a fake AbletonOSC on loopback) and ``scripts/grid_diff.py`` end to end on a
temp DB. Everything is synthetic — no real audio, no real ~/.dance.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server, udp_client

from dance.als import AlsGenerator
from dance.als.reader import read_als_clips
from dance.als.writer import LiveSetSpec, StemEntry, build_live_set_xml
from dance.config import Settings
from dance.core.database import (
    AudioAnalysis,
    Beat,
    StemFile,
    Track,
    TrackState,
    now_utc,
)
from dance.pipeline.utils.grid_metrics import (
    GAP_COST_FRACTION,
    STRICT_GATED,
    WarpMarker,
    align_beats,
    bar_positions,
    beats_from_warp_markers,
    clean_warp_markers,
    compare_grids,
    drift_ms_per_100_beats,
    find_regressions,
    grid_bpm,
    median_period,
    tempo_octave,
)

REPO = Path(__file__).resolve().parents[1]


def grid(n: int, period: float = 0.5, offset: float = 0.0) -> list[float]:
    return [offset + k * period for k in range(n)]


def every_fourth(n: int, start: int = 0) -> list[int]:
    return list(range(start, n, 4))


# ---------------------------------------------------------------------------
# Metrics on synthetic grids
# ---------------------------------------------------------------------------


def test_identical_grids_are_perfect() -> None:
    ref = grid(200)
    c = compare_grids(
        ref,
        list(ref),
        reference_bpm=120.0,
        estimate_bpm=120.0,
        reference_downbeats=every_fourth(200),
        estimate_downbeats=every_fourth(200),
    )
    assert c.matched_beats == 200
    assert c.f1_70 == c.f1_100 == 1.0
    assert c.median_abs_error_ms == pytest.approx(0.0)
    assert c.p95_abs_error_ms == pytest.approx(0.0)
    assert c.phase_offset_ms == pytest.approx(0.0)
    assert c.drift_ms_per_100_beats == pytest.approx(0.0)
    assert c.bpm_error_pct == pytest.approx(0.0)
    assert c.tempo_octave == "1x"
    assert c.bar_phase_accuracy == 1.0
    assert c.bar_phase_shift == 0
    assert c.downbeat_f1_70 == 1.0
    assert c.passes()


@pytest.mark.parametrize(
    ("offset_ms", "f1_70", "f1_100"),
    [(30.0, 1.0, 1.0), (-45.0, 1.0, 1.0), (85.0, 0.0, 1.0), (150.0, 0.0, 0.0)],
)
def test_constant_offset(offset_ms: float, f1_70: float, f1_100: float) -> None:
    ref = grid(160)
    est = grid(160, offset=offset_ms / 1000.0)
    c = compare_grids(ref, est)
    assert c.f1_70 == pytest.approx(f1_70)
    assert c.f1_100 == pytest.approx(f1_100)
    # Every beat still pairs with its twin — the gap window is 0.9 periods.
    assert c.matched_beats == 160
    assert c.median_abs_error_ms == pytest.approx(abs(offset_ms), abs=1e-6)
    assert c.phase_offset_ms == pytest.approx(offset_ms, abs=1e-6)
    assert c.phase_offset_beats == pytest.approx(offset_ms / 500.0, abs=1e-9)
    assert c.drift_ms_per_100_beats == pytest.approx(0.0, abs=1e-6)
    assert c.first_beat_error_ms == pytest.approx(offset_ms, abs=1e-6)


def test_tempo_error_shows_as_drift_and_bpm_error() -> None:
    ref = grid(200, period=0.5)  # 120 BPM
    est = grid(200, period=0.5 / 1.001)  # 0.1 % fast → 0.5 ms earlier per beat
    c = compare_grids(ref, est, reference_bpm=120.0, estimate_bpm=120.12)
    assert c.drift_ms_per_100_beats == pytest.approx(-0.5 / 1.001 * 100, rel=1e-3)
    assert c.bpm_error_pct == pytest.approx(0.1, abs=1e-9)
    assert c.grid_bpm_error_pct == pytest.approx(0.1, abs=1e-6)
    assert c.p95_abs_error_ms is not None and c.p95_abs_error_ms > c.median_abs_error_ms
    # 70 ms is reached after ~140 beats: the tail falls out of the collar.
    assert 0.6 < c.f1_70 < 0.8
    assert c.f1_100 == pytest.approx(1.0)
    assert c.tempo_octave == "1x"


@pytest.mark.parametrize("est_bpm", [125.41, 125.92, 126.30, 124.0])
def test_drift_keeps_growing_past_half_a_beat(est_bpm: float) -> None:
    # Five minutes at 126.048 BPM. Every one of these grids slips by more
    # than half a beat before the end, so the aligner re-pairs beats and the
    # raw signed error is a sawtooth. Unwrapped, drift is the per-beat slip.
    ref_bpm = 126.048
    ref = grid(660, period=60 / ref_bpm)
    est = [round(t * 1000) / 1000 for t in grid(655, period=60 / est_bpm)]
    c = compare_grids(ref, est, reference_bpm=ref_bpm, estimate_bpm=est_bpm)
    expected = (60 / est_bpm - 60 / ref_bpm) * 1000 * 100  # ms per 100 beats
    assert c.drift_ms_per_100_beats == pytest.approx(expected, rel=0.03)
    # Without unwrapping the sawtooth's slope is meaningless — too small, or
    # even the wrong sign (125.92 reads -33 for a true +48). That is the bug
    # that made a 5x better tempo read as 10x worse drift.
    a = align_beats(ref, est)
    raw = drift_ms_per_100_beats(
        [i for i, _ in a.matches], [(est[j] - ref[i]) * 1000 for i, j in a.matches]
    )
    assert raw is not None and abs(raw - expected) > 0.5 * abs(expected)


def test_unknown_estimate_tempo_fails_without_fallback() -> None:
    ref = grid(200)
    c = compare_grids(ref, list(ref), reference_bpm=120.0, estimate_bpm=None, bpm_fallback=False)
    assert c.f1_70 == 1.0
    assert c.estimate_bpm is None and c.bpm_error_pct is None and c.tempo_octave is None
    assert not c.passes()  # a NULL analysis BPM is not a pass
    # With the fallback (library default) the beats' own tempo stands in.
    assert compare_grids(ref, list(ref), reference_bpm=120.0).passes()


def test_reference_clipped_to_the_estimate_span() -> None:
    # Live grids 10 s of beatless intro the tracker (rightly) left empty.
    ref = grid(140)  # 0..69.5 s
    est = [t for t in ref if t >= 10.0]
    c = compare_grids(ref, est, clip_reference_to_estimate=True)
    assert c.reference_beats_total == 140 and c.reference_beats == 120
    assert c.reference_coverage == pytest.approx(120 / 140)
    assert c.f1_70 == 1.0  # the intro is coverage, not missed beats
    unclipped = compare_grids(ref, est)
    assert unclipped.recall_70 == pytest.approx(120 / 140) and unclipped.reference_coverage is None


def test_grid_bpm_sees_through_frame_quantised_jitter() -> None:
    # librosa at hop 512 / 44.1 kHz: beats land on whole frames, so a
    # ~125.4 BPM track comes out as a 41/41/42-frame mix. The MEDIAN interval
    # says 126.05; the grid's real tempo is the average.
    frame = 512 / 44100
    steps = [41, 41, 42] * 100
    times = [0.0]
    for s in steps:
        times.append(times[-1] + s * frame)
    assert 60 / median_period(times) == pytest.approx(126.048, abs=1e-3)
    assert grid_bpm(times) == pytest.approx(60 / (frame * 124 / 3), rel=1e-4)
    # A dropped beat is a gap, not a slow beat.
    dropped = grid(100)[:50] + grid(100)[51:]
    assert grid_bpm(dropped) == pytest.approx(120.0)
    assert grid_bpm([1.0]) is None


def test_dropped_beats_cost_recall_only() -> None:
    ref = grid(200)
    est = [t for k, t in enumerate(ref) if k % 10 != 5]  # drop 20 beats
    c = compare_grids(ref, est)
    assert c.matched_beats == 180
    assert c.reference_unmatched == 20
    assert c.estimate_unmatched == 0
    assert c.precision_70 == pytest.approx(1.0)
    assert c.recall_70 == pytest.approx(0.9)
    assert c.f1_70 == pytest.approx(2 * 0.9 / 1.9)


def test_extra_beats_cost_precision_only() -> None:
    ref = grid(200)
    extras = [ref[k] + 0.25 for k in range(0, 200, 20)]  # 10 off-beat insertions
    c = compare_grids(ref, sorted(ref + extras))
    assert c.matched_beats == 200
    assert c.estimate_unmatched == 10
    assert c.recall_70 == pytest.approx(1.0)
    assert c.precision_70 == pytest.approx(200 / 210)
    assert c.median_abs_error_ms == pytest.approx(0.0)


def test_half_time_estimate() -> None:
    ref = grid(200)
    est = ref[::2]
    c = compare_grids(ref, est, reference_bpm=120.0, estimate_bpm=60.0)
    assert c.matched_beats == 100
    assert c.precision_70 == pytest.approx(1.0)
    assert c.recall_70 == pytest.approx(0.5)
    assert c.f1_70 == pytest.approx(2 / 3)
    assert c.tempo_octave == "0.5x"
    assert c.bpm_error_pct == pytest.approx(-50.0)
    assert not c.passes()


def test_double_time_estimate() -> None:
    ref = grid(200)
    est = grid(400, period=0.25)
    c = compare_grids(ref, est, reference_bpm=120.0, estimate_bpm=240.0)
    assert c.matched_beats == 200
    assert c.recall_70 == pytest.approx(1.0)
    assert c.precision_70 == pytest.approx(0.5)
    assert c.f1_70 == pytest.approx(2 / 3)
    assert c.tempo_octave == "2x"


def test_bar_phase_one_beat_late() -> None:
    ref = grid(64)
    c = compare_grids(
        ref,
        list(ref),
        reference_downbeats=every_fourth(64),
        estimate_downbeats=every_fourth(64, start=1),
    )
    assert c.f1_70 == 1.0  # beats are perfect …
    assert c.bar_phase_accuracy == 0.0  # … the bar is not
    assert c.bar_phase_shift == 3  # estimate counts 1 where Live counts 2
    assert c.downbeat_f1_70 == 0.0  # a downbeat one beat late is not a match


def test_bar_phase_needs_both_sides() -> None:
    ref = grid(32)
    c = compare_grids(ref, list(ref), reference_downbeats=every_fourth(32))
    assert c.bar_phase_accuracy is None
    assert c.downbeat_f1_70 is None


def test_bar_positions_resync_on_irregular_bar() -> None:
    # Downbeats at 1, 5, 8 (a 3-beat bar); beat 0 precedes the first one.
    assert bar_positions(11, [1, 5, 8]) == [3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
    assert bar_positions(4, []) == []


def test_empty_estimate_does_not_crash() -> None:
    c = compare_grids(grid(10), [])
    assert c.f1_70 == 0.0
    assert c.matched_beats == 0
    assert c.median_abs_error_ms is None
    assert c.drift_ms_per_100_beats is None
    assert not c.passes()


def test_downbeat_indices_require_clean_input() -> None:
    with pytest.raises(ValueError):
        compare_grids(
            [0.5, 0.0, 1.0], [0.0, 0.5, 1.0], reference_downbeats=[0], estimate_downbeats=[0]
        )


def test_tempo_octave_labels() -> None:
    assert tempo_octave(128.0, 128.5) == "1x"
    assert tempo_octave(64.0, 128.0) == "0.5x"
    assert tempo_octave(256.0, 128.0) == "2x"
    assert tempo_octave(96.0, 128.0) == "0.75x"  # 3:4 — a triplet-feel misread
    assert tempo_octave(85.33, 128.0) == "0.67x"
    assert tempo_octave(113.71, 124.98) == "other"  # warp-guard's mis-warped bass
    assert tempo_octave(None, 128.0) is None


def _dense_alignment_cost(ref: list[float], est: list[float], gap: float) -> float:
    """The textbook O(n*m) DP (djbooth's), as the oracle for the sparse one."""
    prev = [gap * j for j in range(len(est) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [gap * i] + [0.0] * len(est)
        for j in range(1, len(est) + 1):
            cur[j] = min(
                prev[j - 1] + abs(ref[i - 1] - est[j - 1]),
                prev[j] + gap,
                cur[j - 1] + gap,
            )
        prev = cur
    return prev[-1]


@pytest.mark.parametrize("seed", range(12))
def test_sparse_alignment_matches_dense_dp(seed: int) -> None:
    rng = random.Random(seed)
    ref = sorted(rng.uniform(0, 0.03) + k * 0.5 for k in range(40))
    est = [t + rng.gauss(0, 0.08) for t in ref if rng.random() > 0.15]
    est += [rng.uniform(0, 20) for _ in range(rng.randint(0, 8))]
    est = sorted(set(est))
    a = align_beats(ref, est)
    gap = GAP_COST_FRACTION * median_period(ref)
    cost = sum(abs(ref[i] - est[j]) for i, j in a.matches) + gap * (
        a.reference_unmatched + a.estimate_unmatched
    )
    assert cost == pytest.approx(_dense_alignment_cost(ref, est, gap), abs=1e-9)
    # Monotone, one-to-one.
    assert all(i2 > i1 and j2 > j1 for (i1, j1), (i2, j2) in zip(a.matches, a.matches[1:]))


def test_alignment_scales_to_a_long_track() -> None:
    ref = grid(2000, period=60 / 174)  # 11.5 minutes of drum & bass
    est = grid(2000, period=60 / 174, offset=0.012)
    a = align_beats(ref, est)
    assert len(a.matches) == 2000


# ---------------------------------------------------------------------------
# Warp markers → reference grid
# ---------------------------------------------------------------------------


def test_two_markers_give_a_constant_grid() -> None:
    g = beats_from_warp_markers([WarpMarker(0.0, 0.0), WarpMarker(10.0, 20.0)])
    assert g.bpm == pytest.approx(120.0)
    assert g.times == pytest.approx(grid(21))
    assert g.beat_numbers == list(range(21))
    assert g.downbeats == [0, 4, 8, 12, 16, 20]


def test_tempo_change_between_markers() -> None:
    g = beats_from_warp_markers([WarpMarker(0.0, 0.0), WarpMarker(4.0, 8.0), WarpMarker(9.0, 20.0)])
    assert g.times[8] == pytest.approx(4.0)
    assert g.times[9] == pytest.approx(4.0 + 5.0 / 12.0)  # 144 BPM after the marker
    assert g.times[-1] == pytest.approx(9.0)


def test_markers_extrapolate_to_the_sample_edges() -> None:
    # Beat 0 sits 1 s into the sample (a pickup); the grid still covers 0 s,
    # and continues past the last marker at the last segment's tempo.
    g = beats_from_warp_markers(
        [WarpMarker(1.0, 0.0), WarpMarker(11.0, 20.0)], start_sec=0.0, end_sec=12.0
    )
    assert g.beat_numbers[0] == -2
    assert g.times[0] == pytest.approx(0.0)
    assert g.beat_numbers[-1] == 22
    assert g.times[-1] == pytest.approx(12.0)
    assert g.downbeats[0] == 2  # beat number 0 = Live's 1.1.1


def test_live_autowarp_marker_pair() -> None:
    # The pair Live writes for an auto-warped sample (warp-guard.md fixture):
    # (0, 0) and (0.016092 s, 1/32 beat) → 116.5158 BPM, ~8 beats in 4.1196 s.
    g = beats_from_warp_markers(
        [WarpMarker(0.0, 0.0), WarpMarker(0.016092, 0.03125)], end_sec=4.1196
    )
    assert g.bpm == pytest.approx(116.5158, abs=0.01)
    assert g.times[1] == pytest.approx(60 / 116.5158, abs=1e-4)


def test_clean_warp_markers_drops_non_monotonic() -> None:
    ms = clean_warp_markers(
        [WarpMarker(2.0, 4.0), WarpMarker(0.0, 0.0), WarpMarker(1.5, 5.0), WarpMarker(1.0, 2.0)]
    )
    assert ms == [WarpMarker(0.0, 0.0), WarpMarker(1.0, 2.0), WarpMarker(2.0, 4.0)]
    with pytest.raises(ValueError):
        beats_from_warp_markers([WarpMarker(0.0, 0.0)])


# ---------------------------------------------------------------------------
# .als round trip through dance's own writer
# ---------------------------------------------------------------------------


def test_als_round_trip_through_writer(tmp_path: Path) -> None:
    stems = [
        StemEntry(kind=k, path=tmp_path / f"{k}.wav", duration_seconds=60.0)
        for k in ("drums", "bass", "vocals", "other")
    ] + [StemEntry(kind="mix", path=tmp_path / "song.mp3", duration_seconds=60.0)]
    xml = build_live_set_xml(LiveSetSpec(name="t", bpm=128.0, stems=stems, locators=[]))
    als = tmp_path / "t.als"
    als.write_bytes(gzip.compress(xml))

    clips = read_als_clips(als)
    by_track = {c.track_name: c for c in clips}
    assert set(by_track) == {"Drums A", "Bass A", "Vocals A", "Other A", "Mix A"}
    mix = by_track["Mix A"]
    assert mix.sample_path == str((tmp_path / "song.mp3").resolve())
    assert mix.sample_name == "song.mp3"
    assert mix.location == "session"
    assert mix.warped is True
    assert mix.sample_duration_s == pytest.approx(60.0, abs=1e-4)
    assert mix.markers == [WarpMarker(0.0, 0.0), WarpMarker(60.0, 128.0)]

    g = beats_from_warp_markers(mix.markers, end_sec=60.0)
    assert g.bpm == pytest.approx(128.0)
    assert len(g.times) == 129  # beats 0..128
    assert g.times[1] == pytest.approx(60 / 128)

    # Plain (un-gzipped) XML parses too.
    plain = tmp_path / "t.xml"
    plain.write_bytes(xml)
    assert len(read_als_clips(plain)) == 5


def test_als_reader_honours_is_warped(tmp_path: Path) -> None:
    doc = b"""<?xml version="1.0"?><Ableton><LiveSet><Tracks><AudioTrack Id="1">
      <Name><EffectiveName Value="Kick" /></Name>
      <DeviceChain><MainSequencer><Sample><ArrangerAutomation><Events>
        <AudioClip Id="0" Time="0"><Name Value="k" />
          <SampleRef><FileRef><Name Value="k.wav" /><Path Value="/x/k.wav" /></FileRef></SampleRef>
          <WarpMarkers><WarpMarker Id="0" SecTime="0" BeatTime="0" /></WarpMarkers>
          <IsWarped Value="false" />
        </AudioClip>
      </Events></ArrangerAutomation></Sample></MainSequencer></DeviceChain>
    </AudioTrack></Tracks></LiveSet></Ableton>"""
    p = tmp_path / "u.als"
    p.write_bytes(gzip.compress(doc))
    (clip,) = read_als_clips(p)
    assert clip.warped is False
    assert clip.location == "arrangement"
    assert clip.track_name == "Kick"
    assert clip.sample_duration_s is None


def _audio_clip(path: str, end_beat: float) -> str:
    return (
        f'<AudioClip Id="0" Time="0"><Name Value="c" />'
        f'<SampleRef><FileRef><Path Value="{path}" /></FileRef>'
        f'<DefaultDuration Value="2646000" /><DefaultSampleRate Value="44100" /></SampleRef>'
        f'<WarpMarkers><WarpMarker Id="0" SecTime="0" BeatTime="0" />'
        f'<WarpMarker Id="1" SecTime="60" BeatTime="{end_beat}" /></WarpMarkers>'
        f'<IsWarped Value="true" /></AudioClip>'
    )


def test_als_reader_skips_take_lane_and_frozen_clips(tmp_path: Path) -> None:
    # Element paths as Live 12.4 writes them (checked against the demo Sets
    # bundled in the app): take-lane clips sit under
    # TakeLanes/TakeLanes/TakeLane/ClipAutomation/Events, frozen audio under
    # DeviceChain/FreezeSequencer (which has its own ClipSlotList).
    doc = f"""<?xml version="1.0"?><Ableton><LiveSet><Tracks><AudioTrack Id="1">
      <Name><EffectiveName Value="Mix A" /></Name>
      <TakeLanes><TakeLanes><TakeLane Id="0"><ClipAutomation><Events>
        {_audio_clip("/lib/song.mp3", 125.0)}
      </Events></ClipAutomation></TakeLane></TakeLanes></TakeLanes>
      <DeviceChain>
        <MainSequencer><ClipSlotList><ClipSlot Id="0"><ClipSlot><Value>
          {_audio_clip("/lib/song.mp3", 126.0)}
        </Value></ClipSlot></ClipSlot></ClipSlotList></MainSequencer>
        <FreezeSequencer><ClipSlotList><ClipSlot Id="0"><ClipSlot><Value>
          {_audio_clip("/freeze/Freeze Mix A.wav", 126.0)}
        </Value></ClipSlot></ClipSlot></ClipSlotList>
        <Sample><ArrangerAutomation><Events>
          {_audio_clip("/freeze/Freeze Mix A.wav", 126.0)}
        </Events></ArrangerAutomation></Sample></FreezeSequencer>
      </DeviceChain>
    </AudioTrack></Tracks></LiveSet></Ableton>"""
    p = tmp_path / "comp.als"
    p.write_bytes(gzip.compress(doc.encode()))
    (clip,) = read_als_clips(p)
    assert clip.location == "session"
    assert clip.markers[-1] == WarpMarker(60.0, 126.0)  # the playing clip, not the take
    assert clip.sample_duration_s == pytest.approx(60.0)
    hidden = read_als_clips(p, include_hidden=True)
    assert sorted(c.location for c in hidden) == ["freeze", "freeze", "session", "take_lane"]
    assert all(c.track_name == "Mix A" for c in hidden)


# ---------------------------------------------------------------------------
# Non-regression gate
# ---------------------------------------------------------------------------


def _m(**kw: float | bool | None) -> dict[str, object]:
    base: dict[str, object] = {
        "f1_70": 0.95,
        "f1_100": 0.97,
        "median_abs_error_ms": 12.0,
        "p95_abs_error_ms": 40.0,
        "abs_bpm_error_pct": 0.05,
        "abs_drift_ms_per_100_beats": 3.0,
        "bar_phase_accuracy": 1.0,
        "downbeat_f1_70": 0.9,
        "passes": True,
    }
    base.update(kw)
    return {"status": "ok", "metrics": base}


def test_gate_quiet_on_identical_and_improved_runs() -> None:
    base = {"1": _m(), "2": _m(f1_70=0.5, passes=False)}
    cur = {"1": _m(f1_70=0.99, median_abs_error_ms=5.0), "2": _m(f1_70=0.97)}
    for after in (base, cur):
        report = find_regressions(base, after)
        assert report.regressions == [] and report.warnings == []
        assert report.not_compared == []


def test_gate_ignores_noise_inside_tolerance() -> None:
    report = find_regressions(
        {"1": _m()}, {"1": _m(f1_70=0.947, median_abs_error_ms=12.9, p95_abs_error_ms=41.5)}
    )
    assert report.regressions == []


def test_gate_flags_each_worsened_metric() -> None:
    report = find_regressions(
        {"1": _m()},
        {"1": _m(f1_70=0.90, p95_abs_error_ms=60.0, abs_bpm_error_pct=0.3)},
    )
    regs = report.regressions
    assert {r.metric for r in regs} == {"f1_70", "p95_abs_error_ms", "abs_bpm_error_pct"}
    assert all(r.track_id == "1" for r in regs)


def test_gate_flags_lost_pass_and_vanished_grid() -> None:
    base = {"1": _m(), "2": _m(), "3": _m()}
    cur = {
        "1": _m(passes=False),
        "2": {"status": "no_db_beats"},
    }
    report = find_regressions(base, cur)
    kinds = {(r.track_id, r.metric) for r in report.regressions}
    assert ("1", "passes") in kinds
    assert ("2", "grid_missing") in kinds
    assert report.not_compared == ["3"]


def test_gate_flags_a_metric_that_became_unknown() -> None:
    # The analysis BPM went NULL and every downbeat flag vanished, while the
    # beats themselves stayed put. None is not "no worse" — each is flagged.
    base = {"1": _m(downbeat_f1_70=1.0)}
    cur = {"1": _m(abs_bpm_error_pct=None, bar_phase_accuracy=None, downbeat_f1_70=None)}
    report = find_regressions(base, cur)
    assert {r.metric for r in report.regressions} == {
        "abs_bpm_error_pct_missing",
        "bar_phase_accuracy_missing",
        "downbeat_f1_70_missing",
    }
    (bpm,) = [r for r in report.regressions if r.metric == "abs_bpm_error_pct_missing"]
    assert bpm.baseline == 0.05 and bpm.current is None
    # A metric that was already unknown in the baseline is not a regression.
    assert find_regressions(cur, cur).regressions == []


def test_failing_track_is_gated_on_tempo_only() -> None:
    base = {"9": _m(f1_70=0.4, median_abs_error_ms=80.0, abs_bpm_error_pct=0.6, passes=False)}
    # F1 and timing swing with the phase sawtooth of a grid that is already
    # off: a warning, not a regression …
    worse_f1 = find_regressions(base, {"9": _m(f1_70=0.3, median_abs_error_ms=120.0, passes=False)})
    assert worse_f1.regressions == []
    assert {r.metric for r in worse_f1.warnings} == {"f1_70", "median_abs_error_ms"}
    # … but a worse tempo error is a regression, and so is losing the tempo.
    worse_tempo = find_regressions(base, {"9": _m(f1_70=0.4, abs_bpm_error_pct=1.2, passes=False)})
    assert [r.metric for r in worse_tempo.regressions] == ["abs_bpm_error_pct"]
    lost_tempo = find_regressions(base, {"9": _m(f1_70=0.4, abs_bpm_error_pct=None, passes=False)})
    assert [r.metric for r in lost_tempo.regressions] == ["abs_bpm_error_pct_missing"]


def _tempo_run(est_bpm: float, ref_bpm: float = 126.048) -> dict[str, object]:
    """One track's grid_diff-shaped result: a constant-tempo DB grid at
    ``est_bpm`` (ms-rounded like ``beats.position_ms``) against Live at
    ``ref_bpm``, both from 0 s — about five minutes of beats."""
    ref = grid(660, period=60 / ref_bpm)
    est = [round(t * 1000) / 1000 for t in grid(655, period=60 / est_bpm)]
    c = compare_grids(
        ref, est, reference_bpm=ref_bpm, estimate_bpm=est_bpm, clip_reference_to_estimate=True
    )
    return {"status": "ok", "metrics": c.to_dict()}


def test_gate_passes_a_real_tempo_improvement() -> None:
    # The essentia BPM fix in miniature: today's beats run 0.5 % slow, the
    # fix gets them to 0.1 %. Both still fail the pass bar, and F1 / median
    # error happen to land WORSE after the fix (the phase sawtooth), yet the
    # change is an improvement and the gate must say so.
    before = {"1": _tempo_run(125.41)}
    after = {"1": _tempo_run(125.92)}
    mb, ma = before["1"]["metrics"], after["1"]["metrics"]
    assert not mb["passes"] and not ma["passes"]
    assert ma["f1_70"] < mb["f1_70"]  # the trap the old gate fell into
    assert ma["abs_grid_bpm_error_pct"] < mb["abs_grid_bpm_error_pct"]
    # Drift no longer collapses past half a beat: it tracks the tempo error.
    assert ma["abs_drift_ms_per_100_beats"] < mb["abs_drift_ms_per_100_beats"]
    report = find_regressions(before, after)
    assert report.regressions == []
    assert {r.metric for r in report.warnings} >= {"f1_70"}
    # And the reverse change — 0.1 % back to 0.5 % — is caught, on tempo
    # error and on drift.
    back = find_regressions(after, before)
    assert {r.metric for r in back.regressions} == {
        "abs_bpm_error_pct",
        "abs_grid_bpm_error_pct",
        "abs_drift_ms_per_100_beats",
    }


def test_strict_gate_holds_failing_tracks_to_every_metric() -> None:
    # --strict is the literal "any worsened metric on any track" reading: it
    # rejects the tempo improvement above on F1 / timing. Documented trap.
    before = {"1": _tempo_run(125.41)}
    after = {"1": _tempo_run(125.92)}
    strict = find_regressions(before, after, failing_track_gated=STRICT_GATED)
    assert strict.warnings == []
    assert "f1_70" in {r.metric for r in strict.regressions}
    assert "abs_bpm_error_pct" not in {r.metric for r in strict.regressions}


def test_failing_track_still_gated_on_coverage() -> None:
    # Coverage does not swing with phase: a failing track whose grid now
    # stops halfway is a regression, not a warning.
    base = {"9": _m(f1_70=0.4, reference_coverage=0.98, passes=False)}
    cur = {"9": _m(f1_70=0.4, reference_coverage=0.55, passes=False)}
    report = find_regressions(base, cur)
    assert [r.metric for r in report.regressions] == ["reference_coverage"]


# ---------------------------------------------------------------------------
# scripts/grid_diff.py end to end on a temp DB
# ---------------------------------------------------------------------------


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", REPO / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def grid_diff() -> ModuleType:
    return _load_script("grid_diff")


@pytest.fixture
def graded_library(session, make_track, tmp_path: Path, db_url: str):
    """Three COMPLETE tracks whose DB grids differ from their .als exports.

    1: perfect (beats on k * 0.5 s at 120 BPM)
    2: 40 ms late everywhere (still inside the 70 ms collar)
    3: half-time (analysis says 60 BPM, beats every second)
    Returns ``(db_path, als_paths)``.
    """
    plans = {
        1: (120.0, 0.5, 0.0),
        2: (120.0, 0.5, 0.040),
        3: (60.0, 1.0, 0.0),
    }
    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
    )
    tracks = []
    for n, (bpm, period, offset) in plans.items():
        t = make_track(
            title=f"Song {n}",
            artist="Synth",
            file_path=str(tmp_path / "library" / f"song{n}.mp3"),
            duration_seconds=60.0,
            state=TrackState.COMPLETE.value,
        )
        session.add(AudioAnalysis(track_id=t.id, bpm=bpm, analyzed_at=now_utc()))
        for kind in ("drums", "bass", "vocals", "other"):
            session.add(
                StemFile(track_id=t.id, kind=kind, path=str(tmp_path / f"s{n}" / f"{kind}.wav"))
            )
        k = 0
        while offset + k * period <= 60.0:
            session.add(
                Beat(
                    track_id=t.id,
                    position_ms=round((offset + k * period) * 1000),
                    beat_number=k % 4 + 1,
                    bar_number=k // 4 + 1,
                    downbeat=k % 4 == 0,
                )
            )
            k += 1
        tracks.append(t)
    session.commit()
    gen = AlsGenerator(session, settings)
    # The .als tempo comes from the DB analysis — set the half-time track's
    # export to the true 120 BPM, as Live would after a correction by ear.
    als = []
    for t in tracks:
        if t.id == 3:
            a = session.query(AudioAnalysis).filter_by(track_id=t.id).one()
            a.bpm = 120.0
            session.flush()
            als.append(gen.write(t))
            a.bpm = 60.0
            session.commit()
        else:
            als.append(gen.write(t))
    db_path = Path(db_url.removeprefix("sqlite:///"))
    return db_path, als


def test_grid_diff_end_to_end_and_gate(
    grid_diff: ModuleType, graded_library, session, tmp_path: Path, capsys
) -> None:
    db_path, als = graded_library
    out = tmp_path / "run1.json"
    rc = grid_diff.main(["--db", str(db_path), "--als", *map(str, als), "--json", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "dance.grid_diff/1"
    assert data["db"].endswith("?immutable=1")
    t1, t2, t3 = (data["tracks"][k] for k in ("1", "2", "3"))
    for t in (t1, t2, t3):
        assert t["status"] == "ok"
        assert t["reference"]["kind"] == "mix"  # the full mix wins over stems
    m1, m2, m3 = t1["metrics"], t2["metrics"], t3["metrics"]
    assert m1["f1_70"] == 1.0 and m1["passes"]
    assert m1["bar_phase_accuracy"] == 1.0
    assert m2["phase_offset_ms"] == pytest.approx(40.0)
    assert m2["precision_70"] == 1.0  # every DB beat is inside the 70 ms collar …
    # … and Live's beat at 60.0 s, past the last DB beat (59.54 s), is outside
    # the span being scored: it lowers coverage, not recall.
    assert m2["recall_70"] == 1.0
    assert m2["reference_beats_total"] == 121 and m2["reference_beats"] == 120
    assert m2["reference_coverage"] == pytest.approx(120 / 121)
    assert t1["estimate_bpm_kind"] == "mix" and m1["estimate_bpm"] == 120.0
    assert m3["tempo_octave"] == "0.5x" and not m3["passes"]
    assert data["summary"]["compared"] == 3 and data["summary"]["passing"] == 2
    printed = capsys.readouterr().out
    assert "SUMMARY" in printed and "Synth - Song 1" in printed

    # Re-run against itself: the gate passes.
    rc = grid_diff.main(
        [
            "--db",
            str(db_path),
            "--als",
            *map(str, als),
            "--baseline",
            str(out),
            "--json",
            str(tmp_path / "run2.json"),
        ]
    )
    assert rc == 0
    assert json.loads((tmp_path / "run2.json").read_text())["gate"]["passed"] is True

    # Now "change the beat tracker": track 1's grid lands 90 ms late.
    for b in session.query(Beat).filter_by(track_id=1):
        b.position_ms += 90
    session.commit()
    rc = grid_diff.main(
        [
            "--db",
            str(db_path),
            "--als",
            *map(str, als),
            "--baseline",
            str(out),
            "--json",
            str(tmp_path / "run3.json"),
        ]
    )
    assert rc == 1
    gate = json.loads((tmp_path / "run3.json").read_text())["gate"]
    assert gate["passed"] is False
    flagged = {(r["track_id"], r["metric"]) for r in gate["regressions"]}
    assert ("1", "passes") in flagged and ("1", "f1_70") in flagged
    assert not any(tid != "1" for tid, _ in flagged)
    assert "FAIL" in capsys.readouterr().out


def test_grid_diff_gate_with_nothing_to_judge_fails(
    grid_diff: ModuleType, graded_library, tmp_path: Path
) -> None:
    db_path, als = graded_library
    base = tmp_path / "base.json"
    assert grid_diff.main(["--db", str(db_path), "--als", str(als[0]), "--json", str(base)]) == 0
    # Only track 2 this time: track 1 is "not compared", nothing was judged.
    rc = grid_diff.main(["--db", str(db_path), "--als", str(als[1]), "--baseline", str(base)])
    assert rc == 1


def test_grid_diff_reports_tracks_without_grid_or_reference(
    grid_diff: ModuleType, graded_library, session, tmp_path: Path
) -> None:
    db_path, als = graded_library
    session.query(Beat).filter_by(track_id=2).delete()
    session.commit()
    out = tmp_path / "r.json"
    rc = grid_diff.main(
        [
            "--db",
            str(db_path),
            "--als",
            str(als[1]),
            "--track",
            "1",
            "--track",
            "2",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    tracks = json.loads(out.read_text())["tracks"]
    assert tracks["1"]["status"] == "no_reference"
    assert tracks["2"]["status"] == "no_db_beats"


def test_grid_diff_never_writes_the_db(grid_diff: ModuleType, graded_library) -> None:
    db_path, als = graded_library
    before = db_path.read_bytes()
    assert grid_diff.main(["--db", str(db_path), "--als", str(als[0])]) == 0
    assert db_path.read_bytes() == before


def test_grid_diff_path_matching_survives_a_moved_library(
    grid_diff: ModuleType, graded_library
) -> None:
    db_path, _ = graded_library
    idx = grid_diff.DbIndex.open(grid_diff.db_uri(str(db_path)))
    t1 = idx.tracks[1]
    moved = "/Volumes/Elsewhere/library/" + Path(t1.file_path).name
    assert idx.match_path(moved) == (1, "mix")
    stem = idx.con.execute(
        "SELECT path FROM stem_files WHERE track_id=1 AND kind='bass'"
    ).fetchone()[0]
    # Stem basenames repeat across tracks ('bass.wav'), so the parent dir decides.
    assert idx.match_path("/elsewhere/" + "/".join(Path(stem).parts[-2:])) == (1, "bass")
    assert idx.match_path("/elsewhere/bass.wav") is None
    # A bare name (the .als FileRef Name fallback) is never an "exact" hit.
    assert idx.match_path("bass.wav") is None
    assert idx.match_path("song1.mp3") == (1, "mix")  # … unless it is unique


def test_grid_diff_shared_file_binds_to_the_canonical_track(
    grid_diff: ModuleType, graded_library, session, make_track
) -> None:
    db_path, _ = graded_library
    t1_path = session.get(Track, 1).file_path
    # Track 4 is a duplicate of track 1 that shares its file.
    make_track(title="Song 1 (dup)", artist="Synth", file_path=t1_path, duplicate_of=1)
    session.commit()
    idx = grid_diff.DbIndex.open(grid_diff.db_uri(str(db_path)))
    assert idx.match_path(t1_path) == (1, "mix")
    assert idx.match_path("/elsewhere/" + Path(t1_path).name) is None  # now ambiguous


def test_grid_diff_null_bpm_fails_and_trips_the_gate(
    grid_diff: ModuleType, graded_library, session, tmp_path: Path, capsys
) -> None:
    db_path, als = graded_library
    base = tmp_path / "base.json"
    argv = ["--db", str(db_path), "--als", *map(str, als)]
    assert grid_diff.main([*argv, "--json", str(base)]) == 0
    assert json.loads(base.read_text())["tracks"]["2"]["metrics"]["passes"] is True

    # A grid change NULLs track 2's analysis BPM; its beats are untouched.
    session.query(AudioAnalysis).filter_by(track_id=2, stem_file_id=None).one().bpm = None
    session.commit()
    out = tmp_path / "after.json"
    rc = grid_diff.main([*argv, "--json", str(out), "--baseline", str(base)])
    assert rc == 1
    data = json.loads(out.read_text())
    t2 = data["tracks"]["2"]
    m2 = t2["metrics"]
    # No silent fallback to the beats table's tempo.
    assert m2["estimate_bpm"] is None and m2["bpm_error_pct"] is None
    assert m2["estimate_grid_bpm"] == pytest.approx(120.0, rel=1e-3)
    assert m2["f1_70"] == 1.0 and m2["passes"] is False
    assert t2["note"] == "no audio_analysis.bpm for mix"
    flagged = {(r["track_id"], r["metric"]) for r in data["gate"]["regressions"]}
    assert flagged == {("2", "passes"), ("2", "abs_bpm_error_pct_missing")}
    assert "no audio_analysis.bpm for mix" in capsys.readouterr().out


def test_grid_diff_stem_reference_uses_that_stems_bpm(
    grid_diff: ModuleType, graded_library, session, tmp_path: Path
) -> None:
    db_path, als = graded_library
    drums = session.query(StemFile).filter_by(track_id=1, kind="drums").one()
    session.add(AudioAnalysis(track_id=1, stem_file_id=drums.id, bpm=119.5, analyzed_at=now_utc()))
    session.commit()
    out = tmp_path / "drums.json"
    argv = ["--db", str(db_path), "--als", str(als[0]), "--track", "1", "--json", str(out)]
    assert grid_diff.main([*argv, "--ref-kind", "drums"]) == 0
    t1 = json.loads(out.read_text())["tracks"]["1"]
    assert t1["reference"]["kind"] == "drums"
    assert t1["estimate_bpm_kind"] == "drums"
    assert t1["db_bpm_by_kind"] == {"mix": 120.0, "drums": 119.5}
    m = t1["metrics"]
    assert m["estimate_bpm"] == 119.5  # the drums stem's BPM, not the mix's 120
    assert m["bpm_error_pct"] == pytest.approx((119.5 - 120.0) / 120.0 * 100)
    # A stem with no analysis row is reported, not judged against the mix.
    assert grid_diff.main([*argv, "--ref-kind", "bass"]) == 0
    t1 = json.loads(out.read_text())["tracks"]["1"]
    assert t1["estimate_bpm_kind"] == "bass" and t1["metrics"]["estimate_bpm"] is None
    assert t1["metrics"]["passes"] is False


def test_grid_diff_reports_conflicting_clips(
    grid_diff: ModuleType, graded_library, tmp_path: Path, capsys
) -> None:
    db_path, als = graded_library
    # A second Set holding the same mix with a different warp: Live at 121.
    root = gzip.decompress(als[0].read_bytes()).decode()
    assert 'BeatTime="120' in root
    other = tmp_path / "other.als"
    other.write_bytes(gzip.compress(root.replace('BeatTime="120', 'BeatTime="121').encode()))
    out = tmp_path / "c.json"
    argv = ["--db", str(db_path), "--track", "1", "--json", str(out)]
    # The same Set twice is one clip, not a conflict.
    assert grid_diff.main([*argv, "--als", str(als[0]), str(als[0])]) == 0
    assert json.loads(out.read_text())["tracks"]["1"]["reference"]["conflicting_clips"] == 0
    capsys.readouterr()
    assert grid_diff.main([*argv, "--als", str(als[0]), str(other)]) == 0
    ref = json.loads(out.read_text())["tracks"]["1"]["reference"]
    assert ref["conflicting_clips"] == 1
    assert ref["source"].startswith(als[0].name)  # first in document order wins
    printed = capsys.readouterr().out
    assert "1 other mix clip(s)" in printed and f"used {ref['source']}" in printed


def test_grid_diff_strict_flag_gates_failing_tracks_on_everything(
    grid_diff: ModuleType, tmp_path: Path
) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"schema": grid_diff.SCHEMA, "tracks": {"1": _tempo_run(125.41)}}))
    improved = {"1": _tempo_run(125.92)}
    default = grid_diff.run_gate(base, improved)
    assert default["passed"] is True and default["policy"] == "failing-tracks-on-tempo"
    assert default["warnings"]
    strict = grid_diff.run_gate(base, improved, strict=True)
    assert strict["passed"] is False and strict["policy"] == "strict"
    assert "f1_70" in {r["metric"] for r in strict["regressions"]}


# ---------------------------------------------------------------------------
# Live over OSC — against a fake AbletonOSC on loopback
# ---------------------------------------------------------------------------


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeLive:
    """A fake patched AbletonOSC: the getters grid_diff --live and
    warp_probe ask for, plus the fork's add/remove_warp_marker setters.

    ``clips`` maps ``(track, slot) -> (sample path, flat [beat, sec, ...])``.
    A clip's length in beats is its last marker's beat (the sample end is
    pinned, as after a probe repair). Setter calls are recorded in ``writes``.
    """

    def __init__(self, reply_port: int, clips: dict[tuple[int, int], tuple[str, list[float]]]):
        self.reply = udp_client.SimpleUDPClient("127.0.0.1", reply_port)
        self.paths = {k: v[0] for k, v in clips.items()}
        self.markers: dict[tuple[int, int], list[tuple[float, float]]] = {
            k: list(zip(v[1][0::2], v[1][1::2])) for k, v in clips.items()
        }
        self.writes: list[tuple[str, tuple[object, ...]]] = []
        d = osc_dispatcher.Dispatcher()
        d.map("/live/clip/get/warp_marker_times", self._get_markers)
        d.map("/live/clip/get/file_path", self._get_path)
        d.map("/live/clip/get/length", self._get_length)
        d.map("/live/clip/add_warp_marker", self._add)
        d.map("/live/clip/remove_warp_marker", self._remove)
        self.server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 0), d)
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _get_markers(self, addr: str, track: int, slot: int) -> None:
        ms = self.markers.get((track, slot))
        if ms:
            self.reply.send_message(addr, [track, slot, *(v for m in ms for v in m)])

    def _get_path(self, addr: str, track: int, slot: int) -> None:
        if (track, slot) in self.paths:
            self.reply.send_message(addr, [track, slot, self.paths[(track, slot)]])

    def _get_length(self, addr: str, track: int, slot: int) -> None:
        ms = self.markers.get((track, slot))
        if ms:
            self.reply.send_message(addr, [track, slot, float(ms[-1][0])])

    def _add(self, addr: str, track: int, slot: int, beat: float, sec: float) -> None:
        self.writes.append((addr, (track, slot, beat, sec)))
        ms = self.markers.setdefault((track, slot), [])
        ms.append((beat, sec))
        ms.sort()

    def _remove(self, addr: str, track: int, slot: int, beat: float) -> None:
        self.writes.append((addr, (track, slot, beat)))
        ms = self.markers.get((track, slot), [])
        self.markers[(track, slot)] = [m for m in ms if abs(m[0] - beat) > 1e-3]

    def __enter__(self) -> FakeLive:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_parse_warp_marker_times() -> None:
    from dance.osc.warp_grid import DECK_COLUMNS, parse_warp_marker_times

    assert parse_warp_marker_times((8, 0, 0.0, 0.0, 128.0, 60.0)) == [
        WarpMarker(sec=0.0, beat=0.0),
        WarpMarker(sec=60.0, beat=128.0),
    ]
    assert DECK_COLUMNS["drums_a"] == 0 and DECK_COLUMNS["mix_a"] == 8  # warp_probe's map


def test_grid_diff_live_reads_the_playing_grid(
    grid_diff: ModuleType, graded_library, tmp_path: Path
) -> None:
    db_path, _ = graded_library
    reply_port = _free_udp_port()
    mix_a = 8
    song1 = str(tmp_path / "library" / "song1.mp3")
    # Live put scene 1's mix at 119.9 BPM — 0.08 % slower than the DB's 120.
    clips = {(mix_a, 0): (song1, [0.0, 0.0, 119.9, 60.0])}
    out = tmp_path / "live.json"
    with FakeLive(reply_port, clips) as live:
        rc = grid_diff.main(
            [
                "--db",
                str(db_path),
                "--live",
                "--scene",
                "1",
                "--scene",
                "2",
                "--osc-port",
                str(live.port),
                "--osc-reply-port",
                str(reply_port),
                "--wait",
                "0.25",
                "--json",
                str(out),
            ]
        )
    assert rc == 0
    data = json.loads(out.read_text())
    assert list(data["tracks"]) == ["1"]  # scene 2 is empty: skipped, not fatal
    t1 = data["tracks"]["1"]
    assert t1["reference"]["source"] == "live:scene1:mix_a"
    m = t1["metrics"]
    assert m["reference_bpm"] == pytest.approx(119.9, rel=1e-6)  # OSC floats are f32
    assert m["bpm_error_pct"] == pytest.approx((120 - 119.9) / 119.9 * 100, rel=1e-3)
    # The DB grid runs ahead of Live's by 0.417 ms per beat.
    assert m["drift_ms_per_100_beats"] == pytest.approx(-(60 / 119.9 - 0.5) * 1e5, rel=1e-2)
    assert m["f1_70"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# scripts/warp_probe.py — report and repair against the fake AbletonOSC
# ---------------------------------------------------------------------------


def _probe_clips() -> dict[tuple[int, int], tuple[str, list[float]]]:
    """Scene 1, deck A: a 60 s sample Live warped three different ways."""
    from dance.osc.warp_grid import DECK_COLUMNS

    return {
        (DECK_COLUMNS["drums_a"], 0): ("/s/drums.wav", [0.0, 0.0, 125.0, 60.0]),
        # Mis-warped bass: two segments, 113.71 BPM overall.
        (DECK_COLUMNS["bass_a"], 0): ("/s/bass.wav", [0.0, 0.0, 56.0, 29.5, 113.71, 60.0]),
        # vocals_a: no clip → the probe prints (empty)
        (DECK_COLUMNS["other_a"], 0): ("/s/other.wav", [0.0, 0.0, 62.5, 60.0]),  # half-time
    }


def _run_probe(probe: ModuleType, live: FakeLive, reply_port: int, *extra: str) -> int:
    return int(
        probe.main(
            [
                "--scene",
                "1",
                "--osc-port",
                str(live.port),
                "--osc-reply-port",
                str(reply_port),
                "--wait",
                "0.2",
                *extra,
            ]
        )
    )


def test_warp_probe_report_mode_reads_every_stem_and_writes_nothing(capsys) -> None:
    probe = _load_script("warp_probe")
    reply_port = _free_udp_port()
    with FakeLive(reply_port, _probe_clips()) as live:
        rc = _run_probe(probe, live, reply_port)
    assert rc == 0
    assert live.writes == []  # nothing is written without --repair
    out = capsys.readouterr().out
    assert "Patched handlers are ACTIVE." in out
    rows = {line.split()[0]: line.split()[1:] for line in out.splitlines() if "_a " in line}
    assert rows["drums_a"] == ["2", "125.00", "125.00"]
    assert rows["bass_a"] == ["3", "113.71", "113.71"]
    assert rows["vocals_a"] == ["(empty)"]
    assert rows["other_a"] == ["2", "62.50", "62.50"]
    assert "Report only" in out


def test_warp_probe_says_when_the_fork_is_not_loaded(capsys) -> None:
    probe = _load_script("warp_probe")
    reply_port = _free_udp_port()
    with FakeLive(reply_port, {}) as live:
        assert _run_probe(probe, live, reply_port) == 1
    assert "did NOT answer" in capsys.readouterr().out


def test_warp_probe_repair_rewrites_each_grid_to_the_target_bpm(capsys) -> None:
    from dance.osc.warp_grid import DECK_COLUMNS

    probe = _load_script("warp_probe")
    reply_port = _free_udp_port()
    with FakeLive(reply_port, _probe_clips()) as live:
        assert _run_probe(probe, live, reply_port, "--repair") == 2  # needs --bpm
        assert live.writes == []
        capsys.readouterr()
        rc = _run_probe(probe, live, reply_port, "--repair", "--bpm", "126")
    assert rc == 0
    bass = DECK_COLUMNS["bass_a"]
    # Every marker but the first removed (last first), then the end pinned
    # so the whole 60 s spans 126 beats.
    bass_writes = [w for w in live.writes if w[1][0] == bass]
    assert [w[0] for w in bass_writes] == [
        "/live/clip/remove_warp_marker",
        "/live/clip/remove_warp_marker",
        "/live/clip/add_warp_marker",
    ]
    assert bass_writes[0][1][2] == pytest.approx(113.71, abs=1e-3)
    assert bass_writes[1][1][2] == pytest.approx(56.0)
    assert bass_writes[2][1][2:] == pytest.approx((126.0, 60.0))
    for kind in ("drums_a", "bass_a", "other_a"):
        assert live.markers[(DECK_COLUMNS[kind], 0)] == pytest.approx([(0.0, 0.0), (126.0, 60.0)])
    out = capsys.readouterr().out
    after = out.split("After repair:")[1]
    assert after.count("126.00 BPM  OK") == 3 and "STILL WRONG" not in after
