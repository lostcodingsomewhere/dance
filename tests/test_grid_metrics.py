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
    TrackState,
    now_utc,
)
from dance.pipeline.utils.grid_metrics import (
    GAP_COST_FRACTION,
    WarpMarker,
    align_beats,
    bar_positions,
    beats_from_warp_markers,
    clean_warp_markers,
    compare_grids,
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
    assert find_regressions(base, base) == ([], [])
    assert find_regressions(base, cur) == ([], [])


def test_gate_ignores_noise_inside_tolerance() -> None:
    regs, _ = find_regressions(
        {"1": _m()}, {"1": _m(f1_70=0.947, median_abs_error_ms=12.9, p95_abs_error_ms=41.5)}
    )
    assert regs == []


def test_gate_flags_each_worsened_metric() -> None:
    regs, _ = find_regressions(
        {"1": _m()},
        {"1": _m(f1_70=0.90, p95_abs_error_ms=60.0, abs_bpm_error_pct=0.3)},
    )
    assert {r.metric for r in regs} == {"f1_70", "p95_abs_error_ms", "abs_bpm_error_pct"}
    assert all(r.track_id == "1" for r in regs)


def test_gate_flags_lost_pass_and_vanished_grid() -> None:
    base = {"1": _m(), "2": _m(), "3": _m()}
    cur = {
        "1": _m(passes=False),
        "2": {"status": "no_db_beats"},
    }
    regs, not_compared = find_regressions(base, cur)
    kinds = {(r.track_id, r.metric) for r in regs}
    assert ("1", "passes") in kinds
    assert ("2", "grid_missing") in kinds
    assert not_compared == ["3"]


def test_gate_failing_tracks_still_must_not_get_worse() -> None:
    regs, _ = find_regressions(
        {"9": _m(f1_70=0.4, passes=False)}, {"9": _m(f1_70=0.3, passes=False)}
    )
    assert [r.metric for r in regs] == ["f1_70"]


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
    assert m2["recall_70"] == pytest.approx(120 / 121)  # … but Live has a beat at 60.0 s
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


# ---------------------------------------------------------------------------
# Live over OSC — against a fake AbletonOSC on loopback
# ---------------------------------------------------------------------------


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeLive:
    """Answers the two getters grid_diff --live asks for, like the fork does."""

    def __init__(self, reply_port: int, clips: dict[tuple[int, int], tuple[str, list[float]]]):
        self.reply = udp_client.SimpleUDPClient("127.0.0.1", reply_port)
        self.clips = clips
        d = osc_dispatcher.Dispatcher()
        d.map("/live/clip/get/warp_marker_times", self._markers)
        d.map("/live/clip/get/file_path", self._path)
        self.server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 0), d)
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _markers(self, addr: str, track: int, slot: int) -> None:
        if (track, slot) in self.clips:
            self.reply.send_message(addr, [track, slot, *self.clips[(track, slot)][1]])

    def _path(self, addr: str, track: int, slot: int) -> None:
        if (track, slot) in self.clips:
            self.reply.send_message(addr, [track, slot, self.clips[(track, slot)][0]])

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
