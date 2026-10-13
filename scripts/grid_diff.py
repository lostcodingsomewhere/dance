#!/usr/bin/env python3
"""Beat-grid differential: dance's DB grid vs Ableton's warp grid.

The measurement gate for beat-grid work. For each selected track it takes
dance's grid (``beats`` rows + the full-mix ``audio_analysis.bpm``) and the
grid Live plays (a clip's warp markers, turned into integer beats), aligns
them, and scores the estimate against Live — Ableton is the reference
(CLAUDE.md rule 3). Metrics come from ``dance.pipeline.utils.grid_metrics``:
F1 at 70/100 ms, median/p95 timing error, phase offset, drift per 100 beats,
BPM error, bar-phase accuracy.

REFERENCES
    --als SET.als [...]   Live Sets (gzipped XML). Clips are matched to DB
                          tracks by sample path (exact, then by the longest
                          unique path suffix). Best: a Set you saved after
                          fixing warps by ear. dance's own exports also work
                          — their grid comes from the DB, so they are a
                          plumbing check, not ground truth.
    --live --scene N      Read the grid Live is playing right now over
                          AbletonOSC (patched fork, see
                          docs/proposals/warp-guard.md). Stop the backend
                          first — it owns reply port 11001 — and wait ~20 s
                          after a load; a fresh clip reports a placeholder.

THE GATE
    python scripts/grid_diff.py --als fixed.als --json base.json        # once
    ... change the beat-grid code, reprocess ...
    python scripts/grid_diff.py --als fixed.als --json new.json --baseline base.json

    Exit 1 when any track regressed versus the baseline (a previously-passing
    track now fails, any gated metric worsened past its tolerance, or a grid
    that existed vanished). Ship a grid change only on exit 0.

The DB is opened read-only: ``--db`` takes a path (opened as
``file:<path>?immutable=1``) or a full ``file:`` URI. Default:
``~/.dance/dance.db`` via the immutable URI. Immutable reads ignore an
un-checkpointed WAL, so stop the pipeline before measuring.

Exit codes: 0 ok · 1 regression (or nothing to compare against the baseline)
· 2 usage / input error.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dance.als.reader import read_als_clips
from dance.pipeline.utils.grid_metrics import (
    PASS_THRESHOLDS,
    REGRESSION_TOLERANCES,
    WarpMarker,
    beats_from_warp_markers,
    compare_grids,
    find_regressions,
)

SCHEMA = "dance.grid_diff/1"
STEM_KINDS = ("drums", "bass", "vocals", "other")
KIND_PREFERENCE = ("mix", "drums", "bass", "other", "vocals")


# ---------------------------------------------------------------------------
# DB (read-only)
# ---------------------------------------------------------------------------


def db_uri(arg: str | None) -> str:
    """Normalise ``--db`` to a read-only SQLite URI."""
    if arg and arg.startswith("file:"):
        return arg
    path = Path(arg).expanduser() if arg else Path.home() / ".dance" / "dance.db"
    return f"{path.resolve().as_uri()}?immutable=1"


def _norm(path: str) -> str:
    return unicodedata.normalize("NFC", os.path.normpath(path))


@dataclass
class DbTrack:
    id: int
    title: str | None
    artist: str | None
    file_path: str
    duration_s: float | None
    state: str
    duplicate_of: int | None
    bpm: float | None = None


@dataclass
class DbIndex:
    con: sqlite3.Connection
    tracks: dict[int, DbTrack] = field(default_factory=dict)
    # normalised path -> [(track_id, kind)]; also keyed by trailing components
    by_suffix: dict[tuple[str, ...], list[tuple[int, str]]] = field(default_factory=dict)

    @classmethod
    def open(cls, uri: str) -> DbIndex:
        con = sqlite3.connect(uri, uri=True)
        con.execute("PRAGMA query_only = ON")
        idx = cls(con)
        for row in con.execute(
            "SELECT id, title, artist, file_path, duration_seconds, state, duplicate_of FROM tracks"
        ):
            idx.tracks[row[0]] = DbTrack(*row)
        for tid, bpm in con.execute(
            "SELECT track_id, bpm FROM audio_analysis WHERE stem_file_id IS NULL"
        ):
            if tid in idx.tracks:
                idx.tracks[tid].bpm = bpm
        for t in idx.tracks.values():
            idx._index_path(t.file_path, t.id, "mix")
        for tid, kind, path in con.execute("SELECT track_id, kind, path FROM stem_files"):
            idx._index_path(path, tid, kind)
        return idx

    def _index_path(self, path: str | None, track_id: int, kind: str) -> None:
        if not path:
            return
        parts = Path(_norm(path)).parts
        for n in (len(parts), 3, 2, 1):
            key = tuple(parts[-n:])
            self.by_suffix.setdefault(key, [])
            if (track_id, kind) not in self.by_suffix[key]:
                self.by_suffix[key].append((track_id, kind))

    def match_path(self, path: str | None) -> tuple[int, str] | None:
        """Track + kind for a sample path: exact, else longest unique suffix."""
        if not path:
            return None
        parts = Path(_norm(path)).parts
        for n in (len(parts), 3, 2, 1):
            hits = self.by_suffix.get(tuple(parts[-n:]), [])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1 and n == len(parts):
                return hits[0]  # same file registered twice: take the first
        return None

    def canonical_track_ids(self, include_duplicates: bool) -> list[int]:
        q = (
            "SELECT DISTINCT t.id FROM tracks t JOIN beats b ON b.track_id = t.id "
            "WHERE t.state = 'complete'"
        )
        if not include_duplicates:
            q += " AND t.duplicate_of IS NULL"
        return [r[0] for r in self.con.execute(q + " ORDER BY t.id")]

    def beats(self, track_id: int) -> tuple[list[float], list[int]]:
        """(times_s, downbeat_indices) from ``beats``, de-duplicated by ms."""
        times: list[float] = []
        downbeats: list[int] = []
        last_ms: int | None = None
        for ms, down in self.con.execute(
            "SELECT position_ms, downbeat FROM beats WHERE track_id = ? ORDER BY position_ms, id",
            (track_id,),
        ):
            if ms is None or ms == last_ms:
                continue
            last_ms = ms
            if down:
                downbeats.append(len(times))
            times.append(ms / 1000.0)
        return times, downbeats


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    track_id: int
    kind: str
    source: str
    markers: list[WarpMarker]
    warped: bool = True
    duration_s: float | None = None


def references_from_als(paths: Sequence[Path], db: DbIndex) -> tuple[list[Reference], int]:
    """References from Live Sets, plus the count of clips that matched no track."""
    refs: list[Reference] = []
    unmatched = 0
    for p in paths:
        for clip in read_als_clips(p):
            hit = db.match_path(clip.sample_path or clip.sample_name)
            if hit is None:
                unmatched += 1
                continue
            refs.append(
                Reference(
                    track_id=hit[0],
                    kind=hit[1],
                    source=f"{p.name}:{clip.track_name or '?'}",
                    markers=clip.markers,
                    warped=clip.warped,
                    duration_s=clip.sample_duration_s,
                )
            )
    return refs, unmatched


def references_from_live(args: argparse.Namespace, db: DbIndex) -> list[Reference] | None:
    """References read from the running Live set; None when OSC is unusable."""
    from dance.osc.warp_grid import (
        DECK_COLUMNS,
        OscAsker,
        read_clip_file_path,
        read_warp_markers,
    )

    asker = OscAsker(host=args.osc_host, port=args.osc_port, reply_port=args.osc_reply_port)
    try:
        asker.start()
    except OSError:
        print(
            f"ERROR: OSC reply port {args.osc_reply_port} is busy — stop the backend first.",
            file=sys.stderr,
        )
        return None
    refs: list[Reference] = []
    try:
        column = DECK_COLUMNS[f"{args.kind}_{args.side}"]
        for scene in args.scene:
            slot = scene - 1
            markers = read_warp_markers(asker, column, slot, wait=args.wait)
            if markers is None:
                print(
                    f"scene {scene}: warp_marker_times did not answer (fork not "
                    "loaded, or the cell is empty) — see docs/proposals/warp-guard.md",
                    file=sys.stderr,
                )
                continue
            path = read_clip_file_path(asker, column, slot, wait=args.wait)
            hit = db.match_path(path)
            if hit is None and args.track and len(args.track) == 1 and len(args.scene) == 1:
                hit = (args.track[0], args.kind)
            if hit is None:
                print(f"scene {scene}: clip {path!r} matches no DB track", file=sys.stderr)
                continue
            refs.append(
                Reference(
                    track_id=hit[0],
                    kind=hit[1],
                    source=f"live:scene{scene}:{args.kind}_{args.side}",
                    markers=markers,
                )
            )
    finally:
        asker.stop()
    return refs


def pick_reference(refs: list[Reference], ref_kind: str) -> Reference | None:
    if not refs:
        return None
    order = KIND_PREFERENCE if ref_kind == "auto" else (ref_kind,)
    for kind in order:
        for r in refs:
            if r.kind == kind:
                return r
    return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def evaluate_track(db: DbIndex, track_id: int, ref: Reference | None) -> dict[str, Any]:
    t = db.tracks.get(track_id)
    out: dict[str, Any] = {
        "track_id": track_id,
        "title": t.title if t else None,
        "artist": t.artist if t else None,
        "db_bpm": t.bpm if t else None,
    }
    if t is None:
        return {**out, "status": "no_such_track"}
    if ref is None:
        return {**out, "status": "no_reference"}
    out["reference"] = {
        "source": ref.source,
        "kind": ref.kind,
        "warp_markers": len(ref.markers),
    }
    if not ref.warped:
        return {**out, "status": "reference_unwarped"}
    est, est_down = db.beats(track_id)
    if not est:
        return {**out, "status": "no_db_beats"}
    # Live's grid spans the whole sample; Live's own auto-warp markers span
    # only the first 1/32 beat, so never fall back to "the last marker".
    end = t.duration_s or ref.duration_s or est[-1]
    try:
        grid = beats_from_warp_markers(ref.markers, start_sec=0.0, end_sec=end)
    except ValueError as exc:
        return {**out, "status": "bad_reference", "error": str(exc)}
    out["reference"]["bpm"] = grid.bpm
    cmp = compare_grids(
        grid.times,
        est,
        reference_bpm=grid.bpm,
        estimate_bpm=t.bpm,
        reference_downbeats=grid.downbeats,
        estimate_downbeats=est_down,
    )
    return {**out, "status": "ok", "metrics": cmp.to_dict()}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _f(v: Any, fmt: str, width: int) -> str:
    if v is None:
        return "-".rjust(width)
    return format(v, fmt).rjust(width)


def print_table(results: dict[str, dict[str, Any]]) -> None:
    head = (
        f"{'id':>5} {'track':28} {'ref':6} {'nRef':>5} {'nEst':>5} {'F1@70':>6} "
        f"{'F1@100':>6} {'med':>6} {'p95':>6} {'phase':>7} {'drift':>7} "
        f"{'bpmDB':>7} {'bpmRef':>7} {'err%':>7} {'oct':>5} {'bpmGrd':>7} {'gErr%':>7} "
        f"{'bar':>5}  pass"
    )
    print(head)
    print("-" * len(head))
    for tid, r in results.items():
        name = f"{r.get('artist') or '?'} - {r.get('title') or '?'}"[:28]
        m = r.get("metrics")
        if not m:
            print(f"{tid:>5} {name:28} {r['status']}")
            continue
        print(
            f"{tid:>5} {name:28} {r['reference']['kind'][:6]:6} "
            f"{m['reference_beats']:>5} {m['estimate_beats']:>5} "
            f"{_f(m['f1_70'], '.3f', 6)} {_f(m['f1_100'], '.3f', 6)} "
            f"{_f(m['median_abs_error_ms'], '.1f', 6)} {_f(m['p95_abs_error_ms'], '.1f', 6)} "
            f"{_f(m['phase_offset_ms'], '+.1f', 7)} {_f(m['drift_ms_per_100_beats'], '+.1f', 7)} "
            f"{_f(m['estimate_bpm'], '.2f', 7)} {_f(m['reference_bpm'], '.2f', 7)} "
            f"{_f(m['bpm_error_pct'], '+.3f', 7)} {m['tempo_octave'] or '-':>5} "
            f"{_f(m['estimate_grid_bpm'], '.2f', 7)} {_f(m['grid_bpm_error_pct'], '+.3f', 7)} "
            f"{_f(m['bar_phase_accuracy'], '.2f', 5)}  {'PASS' if m['passes'] else 'fail'}"
        )
    print(
        "\n(med/p95/phase in ms; drift in ms per 100 beats; err% = audio_analysis.bpm vs "
        "Live; bpmGrd/gErr% = tempo of the beats table itself vs Live's grid)"
    )


def summarise(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = [r["metrics"] for r in results.values() if r.get("metrics")]
    statuses: dict[str, int] = {}
    for r in results.values():
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    def med(key: str) -> float | None:
        vals = [m[key] for m in ok if m.get(key) is not None]
        return statistics.median(vals) if vals else None

    return {
        "selected": len(results),
        "compared": len(ok),
        "passing": sum(1 for m in ok if m["passes"]),
        "status_counts": statuses,
        "median_f1_70": med("f1_70"),
        "median_f1_100": med("f1_100"),
        "median_median_abs_error_ms": med("median_abs_error_ms"),
        "median_p95_abs_error_ms": med("p95_abs_error_ms"),
        "median_abs_bpm_error_pct": med("abs_bpm_error_pct"),
        "median_abs_grid_bpm_error_pct": med("abs_grid_bpm_error_pct"),
        "median_abs_drift_ms_per_100_beats": med("abs_drift_ms_per_100_beats"),
        "octave_errors": sum(1 for m in ok if m.get("tempo_octave") not in (None, "1x")),
    }


def print_summary(s: dict[str, Any]) -> None:
    def fmt(v: Any, f: str) -> str:
        return "-" if v is None else format(v, f)

    print(
        f"\nSUMMARY  compared {s['compared']}/{s['selected']} tracks · "
        f"{s['passing']} pass · statuses {s['status_counts']}"
    )
    print(
        f"  median F1@70 {fmt(s['median_f1_70'], '.3f')} · F1@100 "
        f"{fmt(s['median_f1_100'], '.3f')} · median|err| "
        f"{fmt(s['median_median_abs_error_ms'], '.1f')} ms · p95 "
        f"{fmt(s['median_p95_abs_error_ms'], '.1f')} ms · |bpm err| "
        f"{fmt(s['median_abs_bpm_error_pct'], '.3f')}% · |grid bpm err| "
        f"{fmt(s['median_abs_grid_bpm_error_pct'], '.3f')}% · |drift| "
        f"{fmt(s['median_abs_drift_ms_per_100_beats'], '.1f')} ms/100 · octave errors "
        f"{s['octave_errors']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", help="DB path or file: URI (opened read-only/immutable).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--als", nargs="+", type=Path, help="Live Set(s) to read warp grids from.")
    src.add_argument("--live", action="store_true", help="Read warp grids from Live over OSC.")
    ap.add_argument("--track", type=int, action="append", help="Track id (repeatable).")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Every complete track with beats (default: tracks that have a reference).",
    )
    ap.add_argument("--include-duplicates", action="store_true")
    ap.add_argument(
        "--ref-kind",
        choices=("auto", "mix", *STEM_KINDS),
        default="auto",
        help="Which clip is the reference when a Set has several (auto: mix first).",
    )
    ap.add_argument("--json", type=Path, help="Write per-track results + summary here.")
    ap.add_argument("--baseline", type=Path, help="Previous --json output to gate against.")
    live = ap.add_argument_group("--live options")
    live.add_argument("--scene", type=int, action="append", help="1-based scene (repeatable).")
    live.add_argument("--side", choices=("a", "b"), default="a")
    live.add_argument("--kind", choices=("mix", *STEM_KINDS), default="mix")
    live.add_argument("--osc-host", default="127.0.0.1")
    live.add_argument("--osc-port", type=int, default=11000)
    live.add_argument("--osc-reply-port", type=int, default=11001)
    live.add_argument("--wait", type=float, default=0.8, help="Seconds to wait per OSC reply.")
    return ap


def run_gate(baseline_path: Path, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Judge ``results`` against a previous run. ``passed`` is the verdict."""
    data = json.loads(baseline_path.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{baseline_path} is not a {SCHEMA} file")
    regressions, not_compared = find_regressions(data["tracks"], results)
    compared = [tid for tid, b in data["tracks"].items() if b.get("metrics") and tid in results]
    print(f"\nGATE vs {baseline_path}: {len(compared)} baseline track(s) re-measured")
    if not_compared:
        print(f"  not selected this run (not judged): {', '.join(sorted(not_compared))}")
    if not compared:
        print("  FAIL: no baseline track was re-measured — the gate judged nothing")
    elif regressions:
        print(f"  FAIL: {len(regressions)} regression(s)")
        for r in regressions:
            print(f"    track {r.track_id}: {r.detail}")
    else:
        print("  PASS: no regressions")
    return {
        "baseline": str(baseline_path),
        "compared": len(compared),
        "not_compared": sorted(not_compared),
        "regressions": [r.to_dict() for r in regressions],
        "passed": bool(compared) and not regressions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.live and not args.scene:
        print("--live needs at least one --scene", file=sys.stderr)
        return 2
    uri = db_uri(args.db)
    try:
        db = DbIndex.open(uri)
    except sqlite3.Error as exc:
        print(f"cannot open DB {uri}: {exc}", file=sys.stderr)
        return 2

    if args.als:
        missing = [p for p in args.als if not p.exists()]
        if missing:
            print(f"no such .als: {', '.join(map(str, missing))}", file=sys.stderr)
            return 2
        refs, unmatched = references_from_als(args.als, db)
        if unmatched:
            print(f"({unmatched} clip(s) in the Set(s) matched no DB track)")
    else:
        live_refs = references_from_live(args, db)
        if live_refs is None:
            return 2
        refs = live_refs

    by_track: dict[int, list[Reference]] = {}
    for r in refs:
        by_track.setdefault(r.track_id, []).append(r)

    if args.track:
        selected = list(dict.fromkeys(args.track))
    elif args.all:
        selected = sorted(set(db.canonical_track_ids(args.include_duplicates)) | set(by_track))
    else:
        selected = sorted(by_track)
    if not selected:
        print("no tracks selected (no reference clip matched a DB track)", file=sys.stderr)
        return 2

    results = {
        str(tid): evaluate_track(db, tid, pick_reference(by_track.get(tid, []), args.ref_kind))
        for tid in selected
    }
    print_table(results)
    summary = summarise(results)
    print_summary(summary)

    gate: dict[str, Any] | None = None
    if args.baseline:
        try:
            gate = run_gate(args.baseline, results)
        except (OSError, ValueError, KeyError) as exc:
            print(f"cannot read baseline {args.baseline}: {exc}", file=sys.stderr)
            return 2

    if args.json:
        payload = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "db": uri,
            "reference": [str(p) for p in args.als] if args.als else "live",
            "ref_kind": args.ref_kind,
            "pass_thresholds": PASS_THRESHOLDS,
            "regression_tolerances": REGRESSION_TOLERANCES,
            "summary": summary,
            "tracks": results,
            "gate": gate,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        print(f"\nwrote {args.json}")

    return 1 if gate is not None and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
