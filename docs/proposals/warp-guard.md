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

Live reads a `.asd` next to the audio file and skips analysis entirely. This is
the canonical fix and a one-time batch over 1,412 stems. The format is binary
and undocumented; this is a research task, not an afternoon.

Warp **markers** are not reachable over OSC in any case — AbletonOSC's clip
handler explicitly excludes `warp_markers` ("Infered arg_value type is not
supported"), and Live's LOM exposes no marker-mutation method. So there is no
third option that works purely through the existing bridge.

## Recommendation

Test Option A (one preference toggle, one load, one read-back — ~5 minutes at
the rig). If the stamp survives, implement it behind an "is anything playing?"
guard and keep the audit as the backstop. If it does not survive, Option B is
the only real fix and the audit plus manual correction is the interim.

Until then, the practical workaround is in [`../session-1.md`](../session-1.md):
**start on the mix cells, not the stems.** Live auto-warps a full mix reliably
(track 1's mix cell landed at 124.98, matching its drums exactly) — it is
isolated stems it cannot read.
