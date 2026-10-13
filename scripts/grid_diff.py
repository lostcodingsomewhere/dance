#!/usr/bin/env python3
"""Beat-grid differential: dance's DB grid vs Ableton's warp grid.

The measurement gate for beat-grid work. For each selected track it takes
dance's grid (``beats`` rows + ``audio_analysis.bpm``) and the grid Live
plays (a clip's warp markers, turned into integer beats), aligns them, and
scores the estimate against Live — Ableton is the reference (CLAUDE.md
rule 3). Metrics come from ``dance.pipeline.utils.grid_metrics``: F1 at
70/100 ms, median/p95 timing error, phase offset, drift per 100 beats, BPM
error, bar-phase accuracy, and how much of Live's grid the estimate spans.

Stems are first-class (rule 4): when the reference clip is a stem
(``--ref-kind drums``, ``--live --kind bass`` …) the BPM under test is THAT
stem's ``audio_analysis.bpm``, not the mix's; the JSON records which
(``estimate_bpm_kind``). The beat grid is the track's ``beats`` table either
way — stems are sample-aligned with the mix, so one grid serves all five.

Live's grid covers the whole sample, beatless intro and outro included; the
reference is clipped to the span of dance's beats (half a beat of margin)
before scoring, so F1 measures grid accuracy, and the share of Live's grid
that span covers is reported separately as ``reference_coverage``.

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

    Where Live's grid ends (and so ``reference_coverage``) comes from the
    sample's real length: the Set's SampleRef, else the audio file's header.
    Only when neither is readable does it fall back to
    ``tracks.duration_seconds`` (Spotify metadata) — the track's note and the
    JSON's ``reference.end_source`` say so.

    Two clips of the same kind for one track whose warps DISAGREE (a Set
    holding a corrected and an uncorrected copy) make that track
    ``ambiguous_reference``: it is not scored until ``--ref-source`` picks
    one (a substring of ``<set>.als:<session|arrangement>:<track name>``, or
    ``live:scene<N>``).

THE GATE
    python scripts/grid_diff.py --als fixed.als --json base.json        # once
    ... change the beat-grid code, reprocess ...
    python scripts/grid_diff.py --als fixed.als --json new.json --baseline base.json

    Every track the baseline measured is judged again — a baseline track
    whose reference no longer matches (renamed file, re-import, a Set left
    off the command line) is a ``grid_missing`` regression, not silently
    skipped. ``--track`` narrows the run on purpose; then the baseline tracks
    left out are listed as not judged.

    Exit 1 when any track regressed versus the baseline: a previously-passing
    track now fails; any metric worsened past its tolerance; a metric that
    had a value became unknown; a grid that existed vanished; or the track
    was measured against a different reference (warp markers, clip kind or
    sample end changed — ``--allow-reference-change`` turns that into a
    warning). Ship a grid change only on exit 0.

    One waiver: a track that already FAILED and whose beats-table tempo got
    better in the same run (grid-BPM error or drift improved past
    tolerance). Its F1 / timing / bar-phase / downbeat changes are printed
    as warnings — once a grid slips more than half a beat those swing with
    where the phase sawtooth lands, so a real tempo fix can lower them. Its
    tempo error, drift and coverage are still gated. With no tempo change
    there is no waiver: a phase shift on a failing track fails the gate.
    --strict drops the waiver (expect it to reject tempo fixes on tracks
    that are still off by more than half a beat).

The DB is opened read-only and never created: ``--db`` takes a path
(opened as ``file:<path>?mode=ro&immutable=1``) or a ``file:`` URI (``mode``
is forced to ``ro``). Default: ``~/.dance/dance.db``. Immutable reads
ignore an un-checkpointed WAL, so stop the pipeline before measuring.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from dance.als.reader import read_als_clips
from dance.pipeline.utils.grid_metrics import (
    FAILING_TRACK_GATED,
    PASS_THRESHOLDS,
    REGRESSION_TOLERANCES,
    STRICT_GATED,
    TEMPO_FIX_METRICS,
    WarpMarker,
    beats_from_warp_markers,
    compare_grids,
    find_regressions,
    reference_fingerprint,
)

SCHEMA = "dance.grid_diff/1"
STEM_KINDS = ("drums", "bass", "vocals", "other")
KIND_PREFERENCE = ("mix", "drums", "bass", "other", "vocals")


# ---------------------------------------------------------------------------
# DB (read-only)
# ---------------------------------------------------------------------------


class DbPathError(ValueError):
    """``--db`` names a database that does not exist."""


def db_uri(arg: str | None) -> str:
    """Normalise ``--db`` to a read-only SQLite URI for an EXISTING file.

    A plain path becomes ``file:<abs path>?mode=ro&immutable=1``. A ``file:``
    URI keeps its other parameters but always gets ``mode=ro``. Without
    ``mode=ro`` SQLite opens URIs read-write-create, so a typo (or a missing
    ``~/.dance/dance.db``) would leave an empty database behind that dance
    would later initialise as a blank library. Raises :class:`DbPathError`
    when the file is missing — nothing is created.
    """
    if arg and arg.startswith("file:"):
        base, _, query = arg.partition("?")
        path = Path(unquote(urlsplit(base).path)).expanduser()
        params = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k != "mode"]
        uri = f"{base}?{urlencode([('mode', 'ro'), *params])}"
    else:
        path = Path(arg).expanduser() if arg else Path.home() / ".dance" / "dance.db"
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    if not path.is_file():
        raise DbPathError(f"no such database: {path}")
    return uri


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
    # audio_analysis.bpm per kind: "mix" (stem_file_id IS NULL) and each stem.
    bpm_by_kind: dict[str, float | None] = field(default_factory=dict)

    @property
    def bpm(self) -> float | None:
        """The full mix's analysis BPM."""
        return self.bpm_by_kind.get("mix")


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
        for tid, kind, bpm in con.execute(
            "SELECT a.track_id, "
            "CASE WHEN a.stem_file_id IS NULL THEN 'mix' ELSE s.kind END, a.bpm "
            "FROM audio_analysis a LEFT JOIN stem_files s ON s.id = a.stem_file_id"
        ):
            if tid in idx.tracks and kind is not None:
                idx.tracks[tid].bpm_by_kind[kind] = bpm
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
        """Track + kind for a sample path: exact, else longest unique suffix.

        An ambiguous suffix matches nothing. The one exception is an exact
        ABSOLUTE path registered more than once (a duplicate track sharing a
        file): that is still the same file, so the canonical track wins. A
        bare file name ("drums.wav") is never treated as exact.
        """
        if not path:
            return None
        parts = Path(_norm(path)).parts
        for n in (len(parts), 3, 2, 1):
            hits = self.by_suffix.get(tuple(parts[-n:]), [])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1 and n == len(parts) and os.path.isabs(path):
                return min(hits, key=self._canonical_first)
        return None

    def _canonical_first(self, hit: tuple[int, str]) -> tuple[bool, int]:
        t = self.tracks.get(hit[0])
        return (t is None or t.duplicate_of is not None, hit[0])

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
    # The sample's real length, and where it came from ("als_sample_ref",
    # "sample_file"); None -> evaluate_track falls back to metadata.
    duration_s: float | None = None
    end_source: str | None = None


def sample_file_duration(path: str | None) -> float | None:
    """Length in seconds from the audio file's header, or None.

    Reads only the header (frames / sample rate) — the same numbers Live's
    ``Clip.sample_length`` is counted in. None when the file is missing or
    unreadable (Live on another machine, a moved library, a codec libsndfile
    lacks).
    """
    if not path:
        return None
    try:
        import soundfile

        info = soundfile.info(path)
    except Exception:  # noqa: BLE001 — any failure means "unknown length"
        return None
    if info.frames > 0 and info.samplerate > 0:
        return float(info.frames) / float(info.samplerate)
    return None


def _real_length(
    set_duration: float | None, sample_path: str | None
) -> tuple[float | None, str | None]:
    if set_duration:
        return set_duration, "als_sample_ref"
    length = sample_file_duration(sample_path)
    return (length, "sample_file") if length else (None, None)


def references_from_als(paths: Sequence[Path], db: DbIndex) -> tuple[list[Reference], int]:
    """References from Live Sets, plus the count of clips that matched no track.

    Take-lane and frozen clips are skipped by the reader. The same sample
    with the same markers (a clip copied between session and arrangement,
    or two Sets) is kept once.
    """
    refs: list[Reference] = []
    seen: set[tuple[int, str, bool, tuple[WarpMarker, ...]]] = set()
    unmatched = 0
    for p in paths:
        for clip in read_als_clips(p):
            hit = db.match_path(clip.sample_path or clip.sample_name)
            if hit is None:
                unmatched += 1
                continue
            key = (hit[0], hit[1], clip.warped, tuple(clip.markers))
            if key in seen:
                continue
            seen.add(key)
            duration, end_source = _real_length(clip.sample_duration_s, clip.sample_path)
            refs.append(
                Reference(
                    track_id=hit[0],
                    kind=hit[1],
                    source=f"{p.name}:{clip.location}:{clip.track_name or '?'}",
                    markers=clip.markers,
                    warped=clip.warped,
                    duration_s=duration,
                    end_source=end_source,
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
            # Live's grid runs to the end of the sample it loaded: take that
            # file's real length, not tracks.duration_seconds (metadata).
            duration = sample_file_duration(path)
            refs.append(
                Reference(
                    track_id=hit[0],
                    kind=hit[1],
                    source=f"live:scene{scene}:{args.kind}_{args.side}",
                    markers=markers,
                    duration_s=duration,
                    end_source="sample_file" if duration else None,
                )
            )
    finally:
        asker.stop()
    return refs


@dataclass
class Pick:
    """The reference chosen for one track, or why there is none."""

    ref: Reference | None
    # Sources of same-kind clips whose warps disagree; non-empty means the
    # track is ambiguous and was NOT given a reference.
    ambiguous: list[str] = field(default_factory=list)


def pick_reference(refs: list[Reference], ref_kind: str, ref_source: str | None = None) -> Pick:
    """The reference clip for a track.

    ``ref_kind`` picks the clip kind (``auto``: the mix first, then stems).
    ``ref_source`` keeps only clips whose source contains that substring
    (case-insensitive). When two or more clips of the chosen kind remain and
    their warps differ, nothing is picked — the Set holds a corrected and an
    uncorrected copy and scoring against either one silently would be a
    coin toss; the caller reports the candidates so the user can choose.
    """
    if ref_source:
        needle = ref_source.lower()
        refs = [r for r in refs if needle in r.source.lower()]
    if not refs:
        return Pick(None)
    order = KIND_PREFERENCE if ref_kind == "auto" else (ref_kind,)
    for kind in order:
        same = [r for r in refs if r.kind == kind]
        if not same:
            continue
        first = same[0]
        if any((r.markers, r.warped) != (first.markers, first.warped) for r in same[1:]):
            return Pick(None, ambiguous=[r.source for r in same])
        return Pick(first)
    return Pick(None)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def evaluate_track(db: DbIndex, track_id: int, pick: Pick | None = None) -> dict[str, Any]:
    t = db.tracks.get(track_id)
    ref = pick.ref if pick else None
    out: dict[str, Any] = {
        "track_id": track_id,
        "title": t.title if t else None,
        "artist": t.artist if t else None,
        "db_bpm_by_kind": dict(t.bpm_by_kind) if t else {},
    }
    if t is None:
        return {**out, "status": "no_such_track"}
    if pick and pick.ambiguous:
        return {**out, "status": "ambiguous_reference", "candidates": pick.ambiguous}
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
    # Live's grid spans the whole sample: take the sample's real length (the
    # Set's SampleRef, else the file header). Only when neither is known,
    # fall back to tracks.duration_seconds — Spotify metadata — and say so,
    # because reference_coverage (gated) then depends on it. Never "the last
    # marker": Live's own auto-warp markers span only the first 1/32 beat.
    notes: list[str] = []
    if ref.duration_s:
        end, end_source = ref.duration_s, ref.end_source or "reference"
    elif t.duration_s:
        end, end_source = t.duration_s, "tracks.duration_seconds"
        notes.append("reference end from tracks.duration_seconds (metadata; sample unreadable)")
    else:
        end, end_source = est[-1], "last_db_beat"
        notes.append("reference end = last DB beat (no sample length or duration known)")
    try:
        grid = beats_from_warp_markers(ref.markers, start_sec=0.0, end_sec=end)
    except ValueError as exc:
        return {**out, "status": "bad_reference", "error": str(exc)}
    out["reference"]["bpm"] = grid.bpm
    out["reference"]["end_s"] = end
    out["reference"]["end_source"] = end_source
    out["reference"]["fingerprint"] = reference_fingerprint(ref.kind, ref.warped, ref.markers, end)
    # The BPM under test is the analysis of the same kind as the reference
    # clip (a drums clip is judged against the drums stem's BPM). No silent
    # fallback to the mix: a missing BPM is reported as missing and fails.
    out["estimate_bpm_kind"] = ref.kind
    estimate_bpm = t.bpm_by_kind.get(ref.kind)
    if estimate_bpm is None:
        notes.append(f"no audio_analysis.bpm for {ref.kind}")
    if notes:
        out["note"] = "; ".join(notes)
    cmp = compare_grids(
        grid.times,
        est,
        reference_bpm=grid.bpm,
        estimate_bpm=estimate_bpm,
        reference_downbeats=grid.downbeats,
        estimate_downbeats=est_down,
        bpm_fallback=False,
        clip_reference_to_estimate=True,
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
        f"{'id':>5} {'track':28} {'ref':6} {'nRef':>5} {'nEst':>5} {'cov':>5} {'F1@70':>6} "
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
            f"{_f(m.get('reference_coverage'), '.2f', 5)} "
            f"{_f(m['f1_70'], '.3f', 6)} {_f(m['f1_100'], '.3f', 6)} "
            f"{_f(m['median_abs_error_ms'], '.1f', 6)} {_f(m['p95_abs_error_ms'], '.1f', 6)} "
            f"{_f(m['phase_offset_ms'], '+.1f', 7)} {_f(m['drift_ms_per_100_beats'], '+.1f', 7)} "
            f"{_f(m['estimate_bpm'], '.2f', 7)} {_f(m['reference_bpm'], '.2f', 7)} "
            f"{_f(m['bpm_error_pct'], '+.3f', 7)} {m['tempo_octave'] or '-':>5} "
            f"{_f(m['estimate_grid_bpm'], '.2f', 7)} {_f(m['grid_bpm_error_pct'], '+.3f', 7)} "
            f"{_f(m['bar_phase_accuracy'], '.2f', 5)}  {'PASS' if m['passes'] else 'fail'}"
        )
    print(
        "\n(nRef = Live beats inside the span of dance's beats; cov = nRef / all of Live's "
        "grid; med/p95/phase in ms; drift in ms per 100 beats; bpmDB/err% = "
        "audio_analysis.bpm of the ref's kind — the mix or that stem — vs Live; "
        "bpmGrd/gErr% = tempo of the beats table itself vs Live's grid)"
    )
    for tid, r in results.items():
        if r.get("note"):
            print(f"  track {tid}: {r['note']}")
        if r.get("candidates"):
            print(
                f"  track {tid}: {len(r['candidates'])} clips of the same kind have different "
                "warps — not scored; pick one with --ref-source:"
            )
            for src in r["candidates"]:
                print(f"      {src}")


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
        "median_reference_coverage": med("reference_coverage"),
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
    ap.add_argument(
        "--ref-source",
        metavar="SUBSTR",
        help="Only use reference clips whose source contains SUBSTR (case-insensitive), e.g. "
        "'fixed.als', 'arrangement', 'scene2'. Resolves ambiguous_reference tracks.",
    )
    ap.add_argument("--json", type=Path, help="Write per-track results + summary here.")
    ap.add_argument(
        "--baseline",
        type=Path,
        help="Previous --json output to gate against. Every track it measured is judged "
        "again unless --track narrows the run.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="With --baseline: no waiver — a failing track whose tempo improved is still "
        "held to F1 / timing / bar phase.",
    )
    ap.add_argument(
        "--allow-reference-change",
        action="store_true",
        help="With --baseline: a track measured against a different reference than in the "
        "baseline is a warning, not a regression. Better: re-measure the baseline against "
        "the new reference.",
    )
    live = ap.add_argument_group("--live options")
    live.add_argument("--scene", type=int, action="append", help="1-based scene (repeatable).")
    live.add_argument("--side", choices=("a", "b"), default="a")
    live.add_argument("--kind", choices=("mix", *STEM_KINDS), default="mix")
    live.add_argument("--osc-host", default="127.0.0.1")
    live.add_argument("--osc-port", type=int, default=11000)
    live.add_argument("--osc-reply-port", type=int, default=11001)
    live.add_argument("--wait", type=float, default=0.8, help="Seconds to wait per OSC reply.")
    return ap


def load_baseline(path: Path) -> dict[str, Any]:
    """A previous ``--json`` output. Raises ``ValueError`` on a foreign file."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} file")
    if not isinstance(data.get("tracks"), dict):
        raise ValueError(f"{path} has no 'tracks' object")
    return data


def baseline_measured(data: Mapping[str, Any]) -> list[int]:
    """Track ids the baseline has metrics for (the ones the gate must judge)."""
    out: list[int] = []
    for tid, r in data["tracks"].items():
        if isinstance(r, dict) and r.get("metrics"):
            try:
                out.append(int(tid))
            except ValueError:
                continue
    return sorted(out)


def gate_policy(strict: bool) -> str:
    return "strict" if strict else "waive-phase-on-tempo-fix"


def run_gate(
    baseline_path: Path,
    results: dict[str, dict[str, Any]],
    *,
    strict: bool = False,
    allow_reference_change: bool = False,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge ``results`` against a previous run. ``passed`` is the verdict.

    ``baseline`` is the already-loaded JSON of ``baseline_path`` (loaded
    here when omitted).
    """
    data = baseline if baseline is not None else load_baseline(baseline_path)
    failing_gated = STRICT_GATED if strict else FAILING_TRACK_GATED
    report = find_regressions(
        data["tracks"],
        results,
        failing_track_gated=failing_gated,
        allow_reference_change=allow_reference_change,
    )
    regressions, not_compared = report.regressions, report.not_compared
    compared = [tid for tid, b in data["tracks"].items() if b.get("metrics") and tid in results]
    policy = gate_policy(strict)
    print(f"\nGATE vs {baseline_path} ({policy}): {len(compared)} baseline track(s) re-measured")
    if not_compared:
        print(f"  left out by --track (not judged): {', '.join(sorted(not_compared))}")
    if report.warnings:
        print(f"  {len(report.warnings)} warning(s), not gated:")
        for r in report.warnings:
            print(f"    track {r.track_id}: {r.detail}")
    if not compared:
        print("  FAIL: no baseline track was re-measured — the gate judged nothing")
    elif regressions:
        print(f"  FAIL: {len(regressions)} regression(s)")
        for r in regressions:
            print(f"    track {r.track_id}: {r.detail}")
        if any(r.metric == "reference_changed" for r in regressions):
            print(
                "  (a changed reference means the numbers are on different yardsticks: "
                "re-measure the baseline against the new reference with the old code, or "
                "pass --allow-reference-change)"
            )
    else:
        print("  PASS: no regressions")
    return {
        "baseline": str(baseline_path),
        "policy": policy,
        "failing_track_gated": sorted(failing_gated),
        "waiver_requires_improved": [] if strict else list(TEMPO_FIX_METRICS),
        "allow_reference_change": allow_reference_change,
        "compared": len(compared),
        "not_compared": sorted(not_compared),
        "regressions": [r.to_dict() for r in regressions],
        "warnings": [r.to_dict() for r in report.warnings],
        "passed": bool(compared) and not regressions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.live and not args.scene:
        print("--live needs at least one --scene", file=sys.stderr)
        return 2
    try:
        uri = db_uri(args.db)
    except DbPathError as exc:
        print(f"cannot open DB: {exc}", file=sys.stderr)
        return 2
    try:
        db = DbIndex.open(uri)
    except sqlite3.Error as exc:
        print(f"cannot open DB {uri}: {exc}", file=sys.stderr)
        return 2

    baseline: dict[str, Any] | None = None
    if args.baseline:
        try:
            baseline = load_baseline(args.baseline)
        except (OSError, ValueError) as exc:
            print(f"cannot read baseline {args.baseline}: {exc}", file=sys.stderr)
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
    else:
        pool = set(by_track)
        if args.all:
            pool |= set(db.canonical_track_ids(args.include_duplicates))
        if baseline is not None:
            # The gate judges every track the baseline measured. One whose
            # reference no longer matches comes back no_reference (or
            # no_such_track) and is flagged grid_missing — never skipped.
            dropped = set(baseline_measured(baseline)) - pool
            if dropped:
                print(
                    f"({len(dropped)} baseline track(s) have no reference this run and will be "
                    f"judged as missing: {', '.join(map(str, sorted(dropped)))})"
                )
            pool |= dropped
        selected = sorted(pool)
    if not selected:
        print("no tracks selected (no reference clip matched a DB track)", file=sys.stderr)
        return 2

    results = {
        str(tid): evaluate_track(
            db, tid, pick_reference(by_track.get(tid, []), args.ref_kind, args.ref_source)
        )
        for tid in selected
    }
    print_table(results)
    summary = summarise(results)
    print_summary(summary)

    gate: dict[str, Any] | None = None
    if args.baseline and baseline is not None:
        gate = run_gate(
            args.baseline,
            results,
            strict=args.strict,
            allow_reference_change=args.allow_reference_change,
            baseline=baseline,
        )

    if args.json:
        payload = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "db": uri,
            "reference": [str(p) for p in args.als] if args.als else "live",
            "ref_kind": args.ref_kind,
            "ref_source": args.ref_source,
            "pass_thresholds": PASS_THRESHOLDS,
            "regression_tolerances": REGRESSION_TOLERANCES,
            "gate_policy": gate_policy(args.strict),
            "failing_track_gated": sorted(STRICT_GATED if args.strict else FAILING_TRACK_GATED),
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
