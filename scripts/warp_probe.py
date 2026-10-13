#!/usr/bin/env python3
"""Probe — and optionally repair — Live's warp grid on a loaded stem.

Live auto-warps every stem it is handed, independently, and gets isolated
stems wrong often. Measured on this library (Live 12.4.2, 4 tracks / 16
stems), every track had at least one stem on the wrong tempo — one bass read
113.71 BPM against 124.98 for the same track's drums; another track's vocals
and melody both landed at half-time. Nothing errors; the stem just drifts.

Live's Clip class has public ``add_warp_marker`` / ``remove_warp_marker``
methods (LOM, Live 11+), but stock AbletonOSC never wired them up. Our fork
now does. This script answers the two questions that decide whether the fix
is real, and it is deliberately a script rather than app code: nothing in the
app should depend on an unverified capability.

    1. Are the patched handlers actually live in the running Live?
    2. Does rewriting the grid produce the tempo we asked for?

USAGE
    # 1. Activate the fork patch (once, in Live):
    #      Preferences -> Link/Tempo/MIDI -> Control Surface: AbletonOSC
    #      Set it to None, then back to AbletonOSC. (Or restart Live.)
    # 2. Load a track onto a deck in the companion app and WAIT ~20s for
    #    Live's auto-warp to settle.
    # 3. Then:
    python scripts/warp_probe.py --scene 1                  # report only
    python scripts/warp_probe.py --scene 1 --repair         # rewrite the grid

Nothing is written unless --repair is passed. The backend must be STOPPED
first — it owns the OSC reply port.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from typing import Any

from dance.osc.client import ABLETON_RECEIVE_PORT, ABLETON_SEND_PORT

# The ask/reply helper and deck-column map are shared with
# scripts/grid_diff.py --live (see dance.osc.warp_grid).
from dance.osc.warp_grid import (
    CLIP_LENGTH,
    DECK_COLUMNS,
    WARP_MARKER_TIMES,
    OscAsker,
)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene", type=int, required=True, help="Scene number as shown in the app (1-based)."
    )
    ap.add_argument("--side", choices=("a", "b"), default="a")
    ap.add_argument(
        "--repair",
        action="store_true",
        help="Rewrite each stem's grid to the track's analyzed BPM.",
    )
    ap.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Target BPM. Default: read from the DB for this track.",
    )
    # Defaults are AbletonOSC's; overridable so the probe can be driven
    # against a fake server in tests.
    ap.add_argument("--osc-host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=ABLETON_RECEIVE_PORT)
    ap.add_argument("--osc-reply-port", type=int, default=ABLETON_SEND_PORT)
    ap.add_argument("--wait", type=float, default=0.8, help="Seconds to wait per OSC reply.")
    args = ap.parse_args(argv)
    slot = args.scene - 1

    asker = OscAsker(host=args.osc_host, port=args.osc_port, reply_port=args.osc_reply_port)
    try:
        asker.start()
    except OSError:
        print(
            f"ERROR: OSC reply port {args.osc_reply_port} is busy — stop the backend first "
            "(the uvicorn process owns it).",
            file=sys.stderr,
        )
        return 2
    client = asker.client

    def ask(address: str, *a: Any) -> list[tuple[Any, ...]]:
        return asker.ask(address, *a, wait=args.wait)

    # --- 1. Is the fork patch active? -------------------------------------
    probe = ask(WARP_MARKER_TIMES, DECK_COLUMNS["drums_a"], slot)
    if not probe:
        print(
            "The patched warp-marker handlers did NOT answer.\n"
            "Either Live has not reloaded its Remote Scripts since the fork\n"
            "was patched, or nothing is loaded at that scene.\n\n"
            "  Fix: Live -> Preferences -> Link/Tempo/MIDI -> Control Surface\n"
            "       set AbletonOSC to None, then back to AbletonOSC.\n"
            "  Then re-run this script.\n\n"
            "Check ~/Music/Ableton/User Library/Remote Scripts/AbletonOSC/"
            "logs/abletonosc.log for 'Unknown OSC address'."
        )
        asker.stop()
        return 1
    print("Patched handlers are ACTIVE.\n")

    # --- 2. Report the current grid per stem ------------------------------
    kinds = [f"{k}_{args.side}" for k in ("drums", "bass", "vocals", "other")]
    print(f"{'cell':10s} {'markers':>8s} {'beats':>9s} {'implied BPM':>12s}")
    grids: dict[str, list[float]] = {}
    for kind in kinds:
        idx = DECK_COLUMNS[kind]
        flat = ask(WARP_MARKER_TIMES, idx, slot)
        if not flat:
            print(f"{kind:10s} {'(empty)':>8s}")
            continue
        vals = [float(v) for v in flat[0][2:]]
        grids[kind] = vals
        pairs = list(zip(vals[0::2], vals[1::2]))
        length = ask(CLIP_LENGTH, idx, slot)
        beats = float(length[0][2]) if length else 0.0
        secs = pairs[-1][1] if pairs else 0.0
        bpm = (beats / secs * 60.0) if secs else 0.0
        print(f"{kind:10s} {len(pairs):8d} {beats:9.2f} {bpm:12.2f}")

    if not args.repair:
        print("\nReport only. Pass --repair to rewrite the grids.")
        asker.stop()
        return 0

    # --- 3. Repair --------------------------------------------------------
    if args.bpm is None:
        print(
            "\n--repair needs --bpm (the tempo the stems SHOULD be at).\n"
            "Take it from the cluster the report above agrees on, or from\n"
            "`dance status` / the track's analysis.",
            file=sys.stderr,
        )
        asker.stop()
        return 2

    print(f"\nRewriting grids to {args.bpm:.2f} BPM …")
    for kind in kinds:
        idx = DECK_COLUMNS[kind]
        vals = grids.get(kind)
        if not vals:
            continue
        pairs = list(zip(vals[0::2], vals[1::2]))
        end_sec = pairs[-1][1]
        # Strip every marker except the first, then pin the end so the whole
        # sample spans exactly duration * bpm / 60 beats.
        for beat_time, _ in reversed(pairs[1:]):
            client.remove_warp_marker(idx, slot, beat_time)
            time.sleep(0.05)
        client.add_warp_marker(idx, slot, end_sec * args.bpm / 60.0, end_sec)
        time.sleep(0.2)

    time.sleep(1.0)
    print("\nAfter repair:")
    for kind in kinds:
        idx = DECK_COLUMNS[kind]
        if kind not in grids:
            continue
        length = ask(CLIP_LENGTH, idx, slot)
        flat = ask(WARP_MARKER_TIMES, idx, slot)
        beats = float(length[0][2]) if length else 0.0
        vals = [float(v) for v in flat[0][2:]] if flat else []
        secs = vals[-1] if vals else 0.0
        bpm = (beats / secs * 60.0) if secs else 0.0
        ok = "OK" if abs(bpm - args.bpm) < 0.5 else "STILL WRONG"
        print(f"  {kind:10s} {beats:9.2f} beats -> {bpm:7.2f} BPM  {ok}")

    print(
        "\nIf that worked: quit Live, reopen, drag the same .wav in fresh,\n"
        "and check whether the corrected grid persisted to the .asd. If it\n"
        "did, the fix is library-wide and permanent."
    )
    asker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
