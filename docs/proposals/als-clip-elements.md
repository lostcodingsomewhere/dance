# `.als` clip elements: write what Live 12.4 actually reads

**Status:** proposal, not implemented. Needs your go-ahead (CLAUDE.md rule 1: `.als` writer emission shape).
**Date:** 2026-10-13
**Touches:** `src/dance/als/writer.py` (`_populate_audio_track`, `_build_audio_clip`, `_add_file_ref`), `tests/test_als_generator.py`, one new test file and fixture. No DB, API, OSC or Stage change.
**Related:** [`warp-guard.md`](warp-guard.md), [`../als-export.md`](../als-export.md), [`als-deck-name-prefix-mismatch.md`](als-deck-name-prefix-mismatch.md)

---

## The problem, measured

The writer builds each `<AudioClip>` by hand (`writer.py:471-574`). Some of its
element names come from older Live versions and some were made up. Live 12.4
writes none of them.

**Reference:** Live's own bundled set,
`/Applications/Ableton Live 12 Standard.app/Contents/App-Resources/Core Library/Ableton Folder Info/Sample Reference.als`
(gzip XML, `Creator="Ableton Live 12.4d1"`, `MinorVersion="12.0_12402"`, the
same schema as our template, `writer.py:55`). It holds 2679 `AudioClip`s, and
**every one has the same 50 children in the same order**. All 2679 `SampleRef`s
have the same 7 children, all 2679 `Loop`s the same 7, and all 5358 `FileRef`s
the same 9 (one under `SampleRef`, one under
`SampleRef/SourceContext/SourceContext/OriginalFileRef`). Our template agrees:
its four `FileRef`s have the same 9-child shape (first at
`blank_live12.xml:3138-3148`), and its one clip — the GroovePool groove
"Swing 16ths 66", a `MidiClip` — uses `<Color Value="7"/>` (`:5440`) and
`<IsWarped Value="true"/>` (`:5492`). Extracted read-only with a throwaway
lxml script outside the repo (gunzip, iterate, count child-tag sequences). No
Ableton content is in the repo.

| dance writes | where | Live 12.4 writes |
|---|---|---|
| clip `<Warp>` | `writer.py:533` | `<IsWarped>` on 2679/2679 clips. `<Warp>` appears **0** times |
| clip `<ColorIndex>` | `:493` | `<Color>`, one per clip. `<ColorIndex>` appears **0** times in the file |
| track `<ColorIndex>` sibling | `:414-419` | nothing. The track `<Color>` is already set at `:411-413` |
| `<Gain>` | `:528` | nothing. Clip gain is most likely `SampleVolume` (`1` on 2664/2679), which we also write (`:531`) |
| `<GranularityResolution>` | `:527` | nothing. The nearest Live element is probably `TransientResolution` (`6` on all 2679) |
| `FileRef` `Name`, `HasRelativePath`, `HasSearchHint`, `SearchHint`, `DefaultDuration`, `DefaultSampleRate` | `:565`, `:570-574` | nothing. The last two exist only one level up on `SampleRef`, where we write them too (`:513-514`) |

Elements Live writes that we omit: `IsWarped`, `Color`, `SampleRef/SamplesToAutoWarp`
(`0` on all), and `FileRef` `RelativePathType`, `LivePackName`, `LivePackId`
and `SourceHint`. We also omit 31 more `AudioClip` children (`TimeSignature`,
`Fades`, `Sync`, `HiQ`, …), which Live presumably fills with defaults.

**What Live's values say about warp.** `IsWarped` is *present* on every clip
but **`false` on 2469** and `true` on 210. The split is by content: 2437 of
the false ones live under `Samples/One Shots/`; 208 of the 238 clips under
`Samples/Loops/` are warped. Live's own convention is "musical loops warped,
one-shots not". Stems are long musical material, so `true` matches it.

Elements we both write, but with a different value or shape:

- **`LaunchQuantisation`.** We write `4` (`:502`), and the comment (`:497-498`)
  says "4 = 1 bar". That is the *Song* enum, `clip_trigger_quantization`
  (`0`=None … `4`=1 Bar). The *clip* enum is `0`=Global, `1`=None, `2`=8 Bars,
  `3`=4 Bars, **`4`=2 Bars**, `5`=1 Bar
  ([LOM `Clip.launch_quantization`](https://docs.cycling74.com/apiref/lom/clip/),
  [AbletonOSC README](https://github.com/ideoforms/AbletonOSC)). The bridge
  already learned this for OSC (`bridge.py:936-937`). Live writes `0` on all
  2679 clips, and so does the template's groove clip (`:5442`). *Inferred, not
  verified:* if the XML uses the LOM clip enum, exported clips launch on
  2-bar boundaries today.
- **`SampleRef/SourceContext`.** We write a leaf `<SourceContext Value=""/>`
  (`:511`). Live writes a class element: `SourceContext/SourceContext Id=…`
  holding `OriginalFileRef/FileRef`, `BrowserContentPath` and
  `LocalFiltersJson`. Leaf-for-class is the gotcha `troubleshooting.md:221`
  warns about, but the Set opens today, so Live tolerates it here.
- **`FileRef/Type`.** We write `1` (`:566`). Live writes `2` on all 5358
  `FileRef`s (samples) and on the template's presets.
- **`FileRef/RelativePath`.** We write the parent *directory* without its
  leading slash and no `RelativePathType` (`:558-563`). Live writes a path to
  the *file*, relative to a base that `RelativePathType` selects: `5` in the
  reference (whose clips also carry `LivePackName="Core Library"`, so probably
  pack-relative), `1` + `../../…` for the template's user presets (probably
  document-relative). What Live writes for an arbitrary external file is unknown
  until verification step 2.
- **Order.** Live puts `WarpMarkers` after `SampleVolume`; we put it straight
  after `SampleRef` (`:516`). In `FileRef`, Live writes `Path` before `Type`.

Live's skeleton (tag names only):

```
AudioClip: LomId LomIdView CurrentStart CurrentEnd Loop Name Annotation Color LaunchMode
  LaunchQuantisation TimeSignature Envelopes ScrollerTimePreserver TimeSelection Legato
  Ram GrooveSettings Disabled VelocityAmount FollowAction Grid FreezeStart FreezeEnd
  IsWarped TakeId IsInKey ScaleInformation AutomationEnvelopesListWrapper SampleRef
  Onsets WarpMode GranularityTones GranularityTexture FluctuationTexture
  TransientResolution TransientLoopMode TransientEnvelope ComplexProFormants
  ComplexProEnvelope Sync HiQ Fade Fades PitchCoarse PitchFine SampleVolume
  WarpMarkers SavedWarpMarkersForStretched MarkersGenerated IsSongTempoLeader
SampleRef: FileRef LastModDate SourceContext SampleUsageHint DefaultDuration DefaultSampleRate SamplesToAutoWarp
FileRef:   RelativePathType RelativePath Path Type LivePackName LivePackId OriginalFileSize OriginalCrc SourceHint
Loop:      LoopStart LoopEnd StartRelative LoopOn OutMarker HiddenLoopStart HiddenLoopEnd
```

Of the 50 `AudioClip` children we write 17: LomId, LomIdView, CurrentStart,
CurrentEnd, Loop, Name, Annotation, LaunchMode, LaunchQuantisation, Disabled,
SampleRef, WarpMode, PitchCoarse, PitchFine, SampleVolume, WarpMarkers,
MarkersGenerated. Our `Loop` already matches Live's 7 children in order.

### What Live does with this today

Live evidently ignores unknown names and defaults missing ones: the Set opens,
and `writer.py:415` relies on exactly that. So, today:

- **A clip's warp state is whatever Live defaults a missing `IsWarped` to.**
  Nobody has measured it. (The beat-grid branch's `.als` reader also assumes
  `true` when both are missing, `claude/grid-metrics:src/dance/als/reader.py:80-85`;
  that is a guess too.) The check CLAUDE.md prescribes ("5 tracks load + audio
  plays + tempo correct") cannot detect it. The writer uses one `bpm` for both
  the project tempo and the grid slope (`:480`), so at the exported tempo the
  stretch ratio is 1.0 and warped and unwarped clips sound identical. The
  difference shows only when the tempo moves or a second deck plays against
  the clip.
- **A clip's colour is Live's default, not `STEM_COLOR_INDEX`.** Track colours are correct.
- Gain, GranularityResolution and the extra `FileRef` fields do nothing. Live
  finds the sample by its absolute `Path`.

Four places claim the two load paths agree on warp, and the XML does not
deliver that: `warp-guard.md:52-54`, `LEARNING.md:156-157`, `bridge.py:2047-2049`
(`_force_warp` docstring) and `tests/test_osc.py:779-782`.

## Interactions

**warp-guard.** Its measurements (bass mis-warped on 3 of 5 tracks) cover the
OSC load path, where Live auto-warps a fresh clip. Nobody has measured the
`.als` path. If Live honours saved markers on a Set clip, exported stems never
go through auto-warp, which may be warp-guard's cheapest fix. Verification
step 2 tests that directly. One unknown matters here: **what `MarkersGenerated`
means.** We write `true`. In the reference it is `true` on all 2469 unwarped
one-shots and `false` on 61 of the 210 warped loops. If `true` means "Live
generated these, may regenerate", writing it could invite exactly the
re-analysis we want to avoid. Step 2 round-trips both values. warp-guard's
Option C (repair over OSC) and this writer should produce the **same grid from
the same function**, so their parity holds by construction rather than by
comment.

**The BPM fix in flight** (`claude/analysis-fixes`, not merged). The markers
encode `audio_analysis.bpm` (`generator.py:171-175`). Its commit message
confirms the stored values came from the librosa fallback: Essentia's
`RhythmExtractor2013` path raised a swallowed `TypeError`, so every track used
`librosa.beat.beat_track` (`analyze.py:180`, sr 44100, default hop 512), whose
tempo snaps to 5168/lag: 120.19, 123.05, 126.05, 129.20 ("282 tracks shared 26
BPMs"). Those are exactly the "our analysis" values in warp-guard's table. Near
125 BPM the steps are about 3 BPM, so the error can reach ±1.5 BPM (~1.2%),
about 0.4 beat of drift over 8 bars against a correctly gridded deck. Writing
`IsWarped=true` makes that wrong grid authoritative. **It still need not wait:**
for a DJ, a warped clip with a slightly wrong grid is no worse than an unwarped
one, which does not follow tempo at all. The fix also makes the analyze stage
set `duration_seconds` from the decoded audio, which the end marker uses. Once
it lands, `dance reprocess --from analyze --all` then `dance export-als --all`
re-encodes every grid with no writer change.

**Grid shape.** Today we write `(0 s, 0)` → `(duration, duration·bpm/60)`
(`:516-524`). With no duration that becomes `(1.0 s, 16 beats)`, which is
**960 BPM** (`:480`, `:523`). Live encodes tempo as the slope of the last two
markers, 1/32 beat apart in 2670/2679 clips; 2527 clips have only that pair,
`(0, 0)` and `(60/(32·bpm) s, 0.03125)`. Copying that convention removes the
dependency on duration.

**Downbeat anchoring (after the beat-grid work).** Beat 0 is currently the
file start, so launch quantisation puts the file start, not the first downbeat,
on the bar. Live's file shows how to encode "1.1.1 is later than the file
start". *Vocal Demo Hum Low* (warped) has markers `(0 s, −3.35)`,
`(1.522 s, 0)`, … and `LoopStart=0`, `StartRelative=0`: the clip launches and
loops from the downbeat, and the pre-roll sits at negative beats. For a first
downbeat at `d` s we would write `(0, −d·bpm/60)`, `(d, 0)` and the trailing
pair, set `CurrentEnd`/`LoopEnd`/`OutMarker` to `(duration − d)·bpm/60`, and
leave `LoopStart` at 0. Locators would also subtract `d` (`markers.py:93`).
`d` comes from `beats.downbeat` (`database.py:635`), once the beat-grid
harness says it can be trusted.

**Ableton-first (rule 3).** Three candidate owners of the grid:
(i) Live's auto-warp, which warp-guard measured as wrong on isolated stems;
(ii) dance's corrected analysis, written as a hint the user can re-warp in Live;
(iii) Live's own warp of the *full mix*, which it gets right, stamped onto the
sample-aligned stems. (iii) is the most Ableton-first, but capturing and
storing Live's grid is a schema change and belongs in its own proposal. Use
(ii) now; the beat-grid harness compares it against Live-saved sets (the
user's corrections by ear), and we move to (iii) if they disagree.

## Options

### A. Conformance patch ← recommended

Keep the hand-built clip and write Live 12.4's names, values and order:

- `<Warp>` becomes `<IsWarped Value="true">`; `WarpMode=0` (Beats) stays.
- Clip `<ColorIndex>` becomes `<Color Value=STEM_COLOR_INDEX[kind]>`. This is
  the same 0-69 palette index we already write to the track `<Color>`, whose
  values were picked by eye in Live 12.4 (`als-export.md:46-58`), so
  **nothing needs converting**. (Live's `Color` values in the reference span
  0-57, plus `-1` on uncoloured scenes, consistent with that palette.
  `ColorIndex` was, as far as we can tell, the pre-Live-11 name; nothing in
  Live 12.4 reads it.) Also drop the track `ColorIndex` sibling (`:414-419`)
  and fix the stale palette in the module docstring (`:26-31` lists
  1/7/13/39/25; the code uses 1/2/4/10/13).
- Drop `Gain`, `GranularityResolution`, and the six `FileRef` extras.
- Add `SamplesToAutoWarp=0`, plus `RelativePathType`, `LivePackName=""`,
  `LivePackId=""` and `SourceHint=""`, with the `RelativePathType`/
  `RelativePath`/`Type`/`SourceContext` values **copied from Live's round-trip**
  (step 2), not guessed.
- `LaunchQuantisation` and `MarkersGenerated` per decisions 3 and 4.
- Use Live's child order, and the trailing-pair grid described above.

About 40 changed lines, testable against the skeleton.

### B. Clone a Live-written AudioClip from a refreshed template

The strategy the writer already uses for tracks: save a Set in Live 12.4 with
one audio clip, and the writer only sets values on known paths.

- **For:** all 50 children are exactly Live's, and a Live upgrade needs only a
  template refresh (`troubleshooting.md:224-227`).
- **Against:** needs you in Live to regenerate `blank_live12.xml`, which every
  part of the writer depends on. It must scrub the `SourceContext` `FileRef`
  and `BrowserContentPath` (stale paths from your machine). More change than
  the bug needs. Fall back to B only if step 2 shows a defaulted element
  (`Fades`, `Sync`, `HiQ`, …) audibly matters.

### C. Stop emitting clips

The 2026-05-30 decision made static exports "columns-only"
(`als-deck-name-prefix-mismatch.md:124`), and `als-export.md:39` already calls
these clips "scaffold decoys".

- **For:** the problem disappears; fewer emitted elements to keep correct.
- **Against:** the export stops being a playable snapshot, contradicting the
  "let's DJ" runbook. The beat-grid harness loses the plumbing check it runs
  on our exports (`grid_diff.py --als`, "a plumbing check, not ground truth").
  And we lose the one load path where markers travel *with* the clip.

## Recommendation

Do **A**, in this order:

1. Measure the baseline (steps 1-2 below). No code; answers "what does Live do
   today", "what does `MarkersGenerated` do" and "which `RelativePathType`
   does Live use for an external file".
2. Land A with the conformance tests, and correct the four parity claims.
3. After the BPM fix merges: `dance reprocess --from analyze --all`, then
   re-export. No writer change.
4. Add downbeat anchoring after the beat-grid gate, as a separate small change
   under the same tests.

## Decisions for you

1. **A, B or C.** I recommend A.
2. **Warp state.** `IsWarped=true` + Beats + markers from dance's analysis
   (grid owner (ii)) — *recommended*: follows tempo changes, matches Live's own
   "loops are warped" convention and the OSC path's `_force_warp`. Alternative:
   `IsWarped=false` — stems play at native speed and stay in sync only at the
   exported tempo; no wrong grid is ever authoritative, but nudging the master
   tempo or beatmatching a second deck breaks them. Alternative: omit markers
   and let Live auto-warp — the behaviour warp-guard measured as broken.
3. **`LaunchQuantisation`:** `5` (explicit 1 Bar, the comment's intent) or `0`
   (Global, which inherits the template's `GlobalQuantisation=4`, `:5321` —
   1 Bar if that field uses the Song enum — and matches the bridge's deck
   clips, `bridge.py:945-948`). Both sound the same today; `5` survives a
   changed global setting, `0` lets you change all clips at once from the
   transport bar. I recommend `5`.
4. **`MarkersGenerated`:** keep `true` or switch to `false`. Decide from
   step 2. If step 2 is inconclusive I lean `false`, on the untested guess that
   it marks hand-placed markers (Live writes `false` on 61 warped loops, and
   only ever `true` on unwarped clips).
5. **Commit a names-only skeleton fixture** extracted from Live's bundled set:
   73 tag names, no values or paths. I recommend yes.
6. **Launch point once anchoring lands:** the first downbeat (`LoopStart=0`)
   or the file start. I recommend the downbeat.

## Migration cost

- **Code:** about 40 lines in `writer.py`. `generator.py` changes only if
  step 2 shows Live wants a document-relative `RelativePath`, because the
  writer would then need `out_path`.
- **Tests:** two assertions change from `./ColorIndex` to `./Color`
  (`tests/test_als_generator.py:291`, `:527`; both read the *track*-level
  invented sibling). One new file of about 80 lines, a JSON fixture, a
  ~40-line extractor script, and the `tests/test_osc.py:779-782` docstring.
- **Docs:** `als-export.md:43` (WarpMarkers), plus the four parity claims above.
- **Data:** none. Existing `.als` files stay as they are until re-exported.
  The beat-grid branch's reader already reads `IsWarped` first and falls back
  to `Warp`, and falls back from `FileRef/Name` to the `Path` basename
  (`reader.py:76-85` on `claude/grid-metrics`), so it works with either writer.
- **You:** one 20-30 minute Live session, as CLAUDE.md requires ("don't change
  `.als` writer emission without re-verifying in Live").

## Verification plan (Live 12.4 Standard)

**1. Baseline, before any code.** Export a track that has stems, then open it
(the full app name matters: Live 12 Lite is installed too):

```bash
dance export-als <id>                                   # prints "✓ Wrote <path>"
open -a "Ableton Live 12 Standard" "<path>"
```

In Live, select the Drums A clip in scene 1 and note: whether the Warp button
is lit, Seg. BPM, Launch → Quantization (does `4` display as "2 Bars"?), and
the clip colour. Then raise the master tempo by 8 BPM and fire scene 1. Warped
stems speed up; unwarped stems do not.

**2. Round-trip.** Wait 30 s, then *File → Save Live Set As…* to
`/tmp/als-rt/roundtrip.als`. Then:

```bash
mkdir -p /tmp/als-rt
gunzip -c "<path>" > /tmp/als-rt/ours.xml
gunzip -c /tmp/als-rt/roundtrip.als > /tmp/als-rt/live.xml
for f in ours live; do
  xmllint --xpath '(//AudioTrack)[1]//AudioClip/*[self::IsWarped or self::Warp or self::Color or self::ColorIndex or self::LaunchQuantisation or self::WarpMarkers or self::MarkersGenerated]' /tmp/als-rt/$f.xml; echo
done
xmllint --xpath '(//AudioTrack)[1]//AudioClip/SampleRef' /tmp/als-rt/live.xml
```

(Both queries checked against a writer-built Set and Live's reference file; on
ours the first prints `ColorIndex=1`, `LaunchQuantisation=4`, the two-marker
`WarpMarkers` and `Warp=true`.) The round-trip shows Live's default for a
missing `IsWarped`, whether it kept our `WarpMarkers` (unchanged means no
re-analysis), what it did with `MarkersGenerated`, and the exact
`RelativePathType`/`RelativePath`/`Type`/`SourceContext` it writes for an
external absolute file, which A then copies. Repeat once with
`MarkersGenerated=false` hand-edited into `ours.xml` (re-gzip, reopen) to
settle decision 4.

**3. Candidate.** Repeat steps 1-2 on the branch. Pass: the Set loads; all 5
clips (four A-side stems + Mix A, which is muted) play; Warp is lit with Beats;
clip colours match track colours; Launch shows the chosen quantisation; and the
round-trip `AudioClip` diff contains only elements Live *added* with defaults,
with none of ours dropped or rewritten.

**4. OSC read-back.** Stop the backend first (it owns OSC port 11001). A
report-only `scripts/als_clip_probe.py --scene 1`, built like `warp_probe.py`'s
`ask()`, queries each A-side column with
`/live/clip/get/{warping,warp_mode,color_index,launch_quantization,length}`
at t=0 s and t=20 s (past warp-guard's 13-15 s auto-warp window). Pass:
`warping=1`, `warp_mode=0`, `color_index == STEM_COLOR_INDEX[kind]`, and
`length` equal across the 4 stems and mix, equal to the export's `CurrentEnd`,
and unchanged at 20 s. With the AbletonOSC fork active, `warp_probe.py --scene 1`
also prints each grid.

## Conformance tests

New file `tests/test_als_clip_conformance.py`:

- `tests/fixtures/live12_clip_skeleton.json`: tag names and order for
  `AudioClip`, `SampleRef`, `FileRef` and `Loop`, generated by
  `python scripts/als_skeleton.py "<Sample Reference.als path>"`.
- Every child we emit under those parents is in the skeleton, in Live's
  relative order. `AudioTrack` has no `ColorIndex`. The `FileRef` shape is also
  checked against the template's own `FileRef` (`blank_live12.xml:3138`), which
  needs no fixture.
- Clip values: `IsWarped=true`, `WarpMode=0`, `Color == STEM_COLOR_INDEX[kind]`
  (equal to the track `Color`), `LaunchQuantisation` and `MarkersGenerated` at
  the chosen values.
- The grid: the trailing pair is 1/32 beat apart, its slope equals `bpm`, and
  it stays finite when `duration_seconds=None`.
- `test_skeleton_matches_installed_live` re-extracts from the installed app and
  diffs against the fixture; `skipif` when the app is absent, as in CI. It
  catches a Live upgrade the same way the template-refresh note does.

Per CLAUDE.md, implementation waits for your go-ahead.
