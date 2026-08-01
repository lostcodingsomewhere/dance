# Warp guard — Live's auto-warp is wrong on stems

**Status:** PARTIALLY IMPLEMENTED (detection shipped; the real fix needs a decision)
**Date:** 2026-08-01
**Touches:** the OSC load contract (`bridge.push_track_to_live`), a new API endpoint

---

## The problem, measured

Every stem the app loads into Live gets **auto-warped by Live**, independently,
using Live's own tempo detection. The app never told Live what tempo the audio
actually is — `create_audio_clip` was followed only by `set_clip_name`.

Measured on 2026-08-01 against the real rig (Live 12.4.2, real library stems,
4 tracks / 16 stems), reading back each clip's warped beat-length over OSC and
dividing by the source duration to recover the tempo Live decided on:

| track | our analysis | drums | bass | vocals | other |
|---|---|---|---|---|---|
| 1 | 126.05 | 124.98 | **113.71** | 124.28 | 123.57 |
| 2 | 123.05 | 123.98 | **112.85** | 122.68 | 123.40 |
| 3 | 126.05 | 122.44 | **73.16** | 123.91 | **147.73** |
| 4 | 129.20 | 127.97 | 126.90 | **62.99** | **63.86** |
| 5 | 129.20 | 129.22 | 127.53 | 129.53 | 130.44 |

**4 of the first 4 tracks had at least one badly mis-warped stem.** Track 5 was
clean. The bass stem was wrong on 3 of 5 — unsurprising, since an isolated bass
has almost no transient content for Live's detector to lock onto.

Nothing errors. The stem simply drifts out of time, which is indistinguishable
from the DJ having fired the wrong thing. That is the worst possible failure
mode for someone learning: you cannot tell the tool from your own technique.

### Two timing facts that shape any fix

1. **A freshly created clip lies.** For ~13–15 s after `create_audio_clip`,
   Live reports `clip.length` as *duration × current project tempo* — a
   placeholder. Every stem agrees during that window. Only afterwards does the
   real analysis land. Any check that runs inline with the load reads the
   placeholder and reports all-clear on a broken scene.
2. **Stamping the project tempo before loading does not work.** Tested
   directly: set project tempo to the track's analyzed BPM → load → clips
   initially report exactly that tempo → ~13 s later Live's auto-warp
   overwrites it (track 4: vocals and other both flipped to half-time). The
   stamp is not authoritative while Live is willing to re-analyze.

## What shipped

Detection only, plus one consistency fix.

- `bridge._force_warp` — every created clip is pinned to `warping=True` +
  `warp_mode=0` (Beats), matching what the offline `.als` writer emits. The two
  load paths no longer disagree about warp. **This does not fix a wrong tempo
  guess** — `warping` was already True by default; the divergence is the
  analysis, not the flag.
- `bridge.check_warp_at(scene_index)` + `POST /ableton/warp-check/{scene}` —
  audits a loaded scene. Groups cells by source track (one scene routinely
  holds stems from four different songs, whose durations legitimately differ),
  then compares beat-lengths **within** each source. The largest mutually
  agreeing cluster is the reference; outliers are named with the fix
  (`*2` / `:2` in Live's Clip view). A near-2× disagreement between the whole
  cluster and our own analysis is reported separately and softly — our
  analyzer disagrees with itself across stems of one track by ~3 BPM, so it is
  the weaker witness (CLAUDE.md rule 3).
- `useWarpCheck` — the companion schedules the audit 18 s after any whole-song
  load and pushes findings into a sticky `LoadWarnings` banner.

Tolerance is 2%. That reliably catches the killers above (9%, 1.7×, 2×) without
crying wolf on track 5's 1.5% bass. It does **not** catch fine drift; only ears
catch that.

## What it does NOT fix

The DJ still has to correct the clip by hand in Live. With roughly one bad stem
per track that is not a workable flow for a live set. Two candidate real fixes:

### Option A — disable Live's auto-warp, then stamp the tempo

Live's **Preferences → Record/Warp/Launch → Auto-Warp Long Samples** is what
overwrites the stamp. With it off, a clip should keep the "assume project
tempo" grid, which the tempo-stamp experiment showed Live applies immediately
and consistently across all four stems.

- Cheap: we already know the analyzed BPM, and `/live/song/set/tempo` works.
- Unverified: the preference is not reachable over OSC and was not toggled
  during this investigation. **Needs one manual test before it is a plan.**
- Wrinkle: the stamp must be applied while nothing is playing, since it moves
  the master tempo. Fine for pre-set deck prep, wrong for a mid-set swap.
- Leaves the downbeat offset unhandled — the grid would start at 0:00 rather
  than the first downbeat. We have `beats` rows (187k, with downbeat flags) if
  that turns out to matter by ear.

### Option B — write `.asd` analysis sidecars

**Investigated and cracked, then set aside — Option C is strictly better.**

A multi-agent reverse-engineering pass (3 independent parsers, adversarial
verification) established the format: a self-describing container, magic
`06 49`, a schema chunk (`ab 1e 56 78`) carrying a typedef table with UTF-16
field names. `WarpMarker` is **named in the file**, with `SecTime : f64` and
`BeatTime : f64`. Blind-confirmed on a fixture: a 4.1196 s / 181675-frame WAV
decoded to markers `(0, 0)` and `(0.016092, 0.03125)` → 116.5158 BPM → exactly
**8.0 beats**, nothing fitted. Forward-walking the record lands on
`OriginalFileSize` == the sibling audio file's byte size, 26/26.

One finding overturns an assumption this doc previously made:

> **The stems' `.asd` files store no tempo at all.** The `WarpMarkers` list
> header is 0, `UnbiasedTempoEstimate` is 0.0, and bytes `D+0 … D+139` are
> **byte-identical across all 20 files** spanning 62.99–147.73 BPM. Live
> re-runs auto-warp on every load. That is a proof, not a correlation.

So the `.asd` never was a cache of Live's answer — which is why the same stem
mis-warps identically every time.

Writing one is feasible (no checksum, no global length field, no self-
referential offsets) but has never been tested against Live 12.4.2, and a
gotcha was found: auto-analysed stems carry `LoopEnd = OutMarker = 0.0`, so a
naive writer that flips the saved-clip-settings flag could produce a
zero-length clip. Confidence that a hand-written `.asd` actually changes
Live's behaviour: **~0.65**. Not good enough to bet an evening on.

### Option C — repair the grid over OSC ← **the plan**

Live's `Clip` exposes **`add_warp_marker` / `move_warp_marker` /
`remove_warp_marker`** as public, documented LOM methods (Live 11+). Stock
AbletonOSC never registered them: `add_warp_marker` takes a *dict*, and the
generic method dispatcher splats positional args. The `warp_markers`
*property* is excluded in `clip.py` for an unrelated reason (it returns a dict
Live can't serialise back) — the **methods** were simply never wired.

This is strictly better than Option B: documented interface, applies to a clip
already loaded, undoable, no binary forgery. And Live writes the `.asd` back
itself when a clip's warp changes — so a repair may persist for every future
load, giving library-wide correctness from the only writer guaranteed correct.

**Shipped (inert until activated):**

- The fork now registers `/live/clip/{add,move,remove}_warp_marker` plus
  `/live/clip/get/warp_marker_times` (the grid flattened to floats, which is
  what makes a repair *verifiable* rather than hopeful).
- `client.py` has the matching senders.
- `scripts/warp_probe.py` reports a scene's grids and, with `--repair`,
  rewrites them.

**Not shipped:** any automatic repair in the app. The patched handlers only
take effect once Live reloads its Remote Scripts, so none of this could be
verified against real Live in the session that wrote it. Per CLAUDE.md rule 2,
unverified capability does not get wired into the performance path.

## Recommendation

**For the first sessions: change nothing.** `check_warp_at` already flags
mis-warped stems; trust it. If it fires with a non-octave drift it now tells
you the target and where to type it (Live's Sample tab → Seg. BPM).

**When you have 15 minutes** (not on a night you intend to play):

1. Live → Preferences → Link/Tempo/MIDI → Control Surface: set AbletonOSC to
   None, then back. That reloads the patched script.
2. Stop the backend (it owns OSC port 11001), load a track, wait ~20 s.
3. `python scripts/warp_probe.py --scene 1` — confirms the handlers answer and
   prints each stem's grid.
4. `python scripts/warp_probe.py --scene 1 --repair --bpm <target>`.
5. Then quit Live, reopen, drag the same `.wav` in fresh. If the corrected grid
   persisted to the `.asd`, the fix is permanent and library-wide — and the
   binary writer is never needed.

If step 5 holds, wire the repair into `check_warp_at` so the audit becomes
self-healing. Note the target BPM must come from **dance's analysis, not from
doubling Live's guess**: within `29d6163a` Live produced 127.97 / 126.90 /
62.99 / 63.86, and doubling the halved pair gives 125.98 / 127.72 — still
disagreeing. And `a747bfec` bass at 113.71 vs 124.98 drums is a 9% drift with
no halving involved, so `:2` / `*2` cannot fix it at all.

Until then, the practical workaround is in [`../session-1.md`](../session-1.md):
**start on the mix cells, not the stems.** Live auto-warps a full mix reliably
(track 1's mix cell landed at 124.98, matching its drums exactly) — it is
isolated stems it cannot read.
