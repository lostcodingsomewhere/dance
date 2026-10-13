# `.als` clip elements: write what Live 12.4 actually reads

**Status:** proposal, not implemented. Needs your go-ahead (CLAUDE.md rule 1: `.als` writer emission shape).
**Date:** 2026-10-13
**Touches:** `src/dance/als/writer.py` (`_populate_audio_track`, `_build_audio_clip`, `_add_file_ref`), `tests/test_als_generator.py`, one new test file and fixture. No DB, API, OSC or Stage change.
**Related:** [`warp-guard.md`](warp-guard.md), [`../als-export.md`](../als-export.md), [`als-deck-name-prefix-mismatch.md`](als-deck-name-prefix-mismatch.md)

---

## The problem, measured

The writer builds each `<AudioClip>` by hand (`writer.py:471-574`). Some of the
element names it uses come from older Live versions, and some it made up.
Live 12.4 writes none of them.

**Reference:** Live's own bundled set,
`/Applications/Ableton Live 12 Standard.app/Contents/App-Resources/Core Library/Ableton Folder Info/Sample Reference.als`
(gzip XML, `Creator="Ableton Live 12.4d1"`, `MinorVersion="12.0_12402"`, the
same schema as our template, `writer.py:55`). It holds 2679 `AudioClip`s,
and **every one has the same 50 children in the same order**. Its 2679
`SampleRef`s all have 7 children, and its 5358 `FileRef`s all have 9 (one
under `SampleRef`, one under `SourceContext/OriginalFileRef`). Our own template
agrees. Its `MidiClip` uses `<Color Value="7"/>` and `<IsWarped Value="true"/>`
(`blank_live12.xml:5440`, `:5492`), and its four `FileRef`s have the same
9-child shape (`:3138-3148`). I extracted this read-only with a throwaway lxml
script that gunzips, iterates the elements and counts child-tag sequences. No
Ableton content is in the repo.

| dance writes | where | Live 12.4 writes |
|---|---|---|
| clip `<Warp>` | `writer.py:533` | `<IsWarped>`: 2679/2679 clips. `<Warp>` appears **0** times |
| clip `<ColorIndex>` | `:493` | `<Color>`: one per clip. `<ColorIndex>` appears **0** times in the file |
| track `<ColorIndex>` sibling | `:414-419` | nothing. The track `<Color>` is already set at `:411-413` |
| `<Gain>` | `:528` | nothing. Clip gain lives in `SampleVolume`, which we also write (`:531`) |
| `<GranularityResolution>` | `:527` | nothing. The Beats-mode resolution is `TransientResolution`, `6` on all 2679 |
| `FileRef` `Name`, `HasRelativePath`, `HasSearchHint`, `SearchHint`, `DefaultDuration`, `DefaultSampleRate` | `:565`, `:570-574` | nothing. The last two exist only one level up on `SampleRef`, where we write them too (`:513-514`) |

Elements Live writes that we omit: `IsWarped`, `Color`, `SampleRef/SamplesToAutoWarp`
(`0` on all), and `FileRef` `RelativePathType`, `LivePackName`, `LivePackId`
and `SourceHint`. We also omit 31 more `AudioClip` children (`TimeSignature`,
`Fades`, `Sync`, `HiQ`, …), which Live fills with defaults.

Elements we both write, but with different values or shape:

- **`LaunchQuantisation`.** We write `4` (`:502`), and the comment says "4 = 1 bar".
  That comment uses the *Song* enum (`0`=None … `4`=1 Bar). The *clip* enum is
  `0`=Global, `1`=None, `2`=8 Bars, `3`=4 Bars, **`4`=2 Bars**, `5`=1 Bar
  ([LOM `Clip.launch_quantization`](https://docs.cycling74.com/apiref/lom/clip/),
  [AbletonOSC README](https://github.com/ideoforms/AbletonOSC)). The bridge
  already learned this for OSC (`bridge.py:936-941`). Live writes `0` on all
  2679 clips, and so does the template's clip. If the XML uses the LOM enum
  (inferred, not verified), exported clips launch on 2-bar boundaries.
- **`FileRef/Type`.** We write `1` (`:566`). Live writes `2` for every sample and preset.
- **`FileRef/RelativePath`.** We write the parent *directory* without its leading
  slash and no `RelativePathType` (`:558-563`). Live writes a path to the *file*,
  relative to a base that `RelativePathType` selects: `5` = pack-relative in
  the reference, `1` + `../../…` in our template.
- **Order.** Live puts `WarpMarkers` after `SampleVolume`; we put it straight
  after `SampleRef` (`:516`). In `FileRef`, Live writes `Path` before `Type`.

Live's skeleton (tag names only):

```
LomId LomIdView CurrentStart CurrentEnd Loop Name Annotation Color LaunchMode
LaunchQuantisation TimeSignature Envelopes ScrollerTimePreserver TimeSelection Legato
Ram GrooveSettings Disabled VelocityAmount FollowAction Grid FreezeStart FreezeEnd
IsWarped TakeId IsInKey ScaleInformation AutomationEnvelopesListWrapper SampleRef
Onsets WarpMode GranularityTones GranularityTexture FluctuationTexture
TransientResolution TransientLoopMode TransientEnvelope ComplexProFormants
ComplexProEnvelope Sync HiQ Fade Fades PitchCoarse PitchFine SampleVolume
WarpMarkers SavedWarpMarkersForStretched MarkersGenerated IsSongTempoLeader
SampleRef: FileRef LastModDate SourceContext SampleUsageHint DefaultDuration DefaultSampleRate SamplesToAutoWarp
FileRef:   RelativePathType RelativePath Path Type LivePackName LivePackId OriginalFileSize OriginalCrc SourceHint
```

Of these 50 `AudioClip` children we write 17: LomId, LomIdView, CurrentStart,
CurrentEnd, Loop, Name, Annotation, LaunchMode, LaunchQuantisation, Disabled,
SampleRef, WarpMode, PitchCoarse, PitchFine, SampleVolume, WarpMarkers,
MarkersGenerated.

### What Live does with this today

Live evidently ignores unknown names and gives missing ones their default
values: the Set opens, and `writer.py:415` relies on exactly that. So, today:

- **A clip's warp state is whatever Live defaults a missing `IsWarped` to.**
  Nobody has measured it. The check CLAUDE.md prescribes ("5 tracks load + audio plays +
  tempo correct") cannot detect it. The writer uses one `bpm` for both the
  project tempo and the grid slope (`:480`), so at the exported tempo the
  stretch ratio is exactly 1.0, and warped and unwarped clips sound identical.
  The difference only shows when the tempo moves or a second deck plays
  against the clip.
- **A clip's colour is Live's default, not `STEM_COLOR_INDEX`.** Track colours are correct.
- Gain, GranularityResolution and the extra `FileRef` fields do nothing. Live
  finds the sample by its absolute `Path`.

Four places claim the two load paths agree on warp, and the XML does not
deliver that: `warp-guard.md:52-54`, `LEARNING.md:156-157`, `bridge.py:2048-2049`
(`_force_warp` docstring) and `tests/test_osc.py:779-782`.

## Interactions

**warp-guard.** Its measurements (bass mis-warped on 3 of 5 tracks) cover the
OSC load path, where Live auto-warps a fresh clip. Nobody has measured the
`.als` path. If Live honours saved markers on a Set clip (explicit
`IsWarped=true` plus `MarkersGenerated=true`, which we already write and
2618/2679 Live clips carry), exported stems never go through auto-warp.
Verification step 2 tests that directly. warp-guard's Option C (repair over
OSC) and this writer should produce the **same grid from the same function**,
so that their parity holds by construction instead of by comment.

**The BPM fix in flight.** The markers encode `audio_analysis.bpm`
(`generator.py:171-175`). The stored values came from the librosa fallback,
`beat_track` (`analyze.py:180`, sr 44100, hop 512), whose tempo snaps to
5168/lag: 120.19, 123.05, 126.05, 129.20. Those are exactly the "our analysis"
values in warp-guard's table. Near 125 BPM the steps are about 3 BPM, so the error can
reach ±1.5 BPM (~1.2%), which is roughly 0.4 beat of drift over 8 bars against
a correctly gridded deck. Writing `IsWarped=true` makes the wrong grid
authoritative. **It still need not wait:** for a DJ, a warped clip with a wrong
grid is never worse than an unwarped one, because an unwarped clip does not
follow tempo at all. Once the fix lands, `dance export-als --all` re-encodes
every grid with no writer change.

**Grid shape.** Today we write `(0 s, 0)` → `(duration, duration·bpm/60)`
(`:516-524`). With no duration that becomes `(1.0 s, 16 beats)`, which is
**960 BPM** (`:480`, `:523`). Live encodes tempo as the slope of a trailing
marker pair, 1/32 beat apart in 2670/2679 clips. Copying that convention
removes the dependency on duration.

**Downbeat anchoring (after the beat-grid work).** Beat 0 is currently the
file start, so launch quantisation puts the file start, not the first downbeat,
on the bar. Live's file shows how to encode "1.1.1 is later than the file
start". *Vocal Demo Hum Low* has markers `(0 s, −3.35)`, `(1.522 s, 0)`, … and
`LoopStart=0`, `StartRelative=0`. The clip launches and loops from the
downbeat, and the pre-roll exists at negative beats. For a first downbeat at
`d` s, we would write `(0, −d·bpm/60)`, `(d, 0)` and the trailing pair, set
`CurrentEnd`/`LoopEnd`/`OutMarker` to `(duration − d)·bpm/60`, and leave
`LoopStart` at 0. The locators would also need to subtract `d`
(`markers.py:93`). `d` comes from `beats.downbeat` (`database.py:635`), once
the beat-grid harness says it can be trusted.

**Ableton-first (rule 3).** There are three candidate owners of the grid:
(i) Live's auto-warp, which warp-guard measured as wrong on isolated stems;
(ii) dance's corrected analysis, written as a hint the user can re-warp in Live;
(iii) Live's own warp of the *full mix*, which it gets right, stamped onto
the sample-aligned stems. (iii) is the most Ableton-first, but capturing and
storing Live's grid is a schema change and belongs in its own proposal. Use
(ii) now. The beat-grid harness then compares it against Live-saved sets, the
user's corrections by ear. Move to (iii) if they disagree.

## Options

### A. Conformance patch ← recommended

Keep the hand-built clip and write Live 12.4's names, values and order:

- `<Warp>` becomes `<IsWarped Value="true">`, and `WarpMode=0` (Beats) stays.
- Clip `<ColorIndex>` becomes `<Color Value=STEM_COLOR_INDEX[kind]>`. This is the
  same 0-69 palette index we already write to the track `<Color>`, and the
  numbers were picked by eye against Live 12.4 (`als-export.md:46-58`), so
  **nothing needs converting**. Whatever palette the old `ColorIndex`
  addressed, nothing in Live 12.4 reads it. Also drop the track
  `ColorIndex` sibling (`:414-419`) and fix the stale palette in the docstring
  (`:24-29` lists 1/7/13/39/25, but the code uses 1/2/4/10/13).
- Drop `Gain`, `GranularityResolution`, and the six `FileRef` extras.
- Add `SamplesToAutoWarp=0`, plus `RelativePathType`, `LivePackName=""`,
  `LivePackId=""` and `SourceHint=""`, with the `RelativePathType`/
  `RelativePath`/`Type` values **copied from Live's round-trip** (step 2), not guessed.
- Set `LaunchQuantisation` to `5` (decision 3).
- Use Live's child order, and the trailing-pair grid described above.

This is about 40 changed lines and can be tested against the skeleton.

### B. Clone a Live-written AudioClip from a refreshed template

This is the strategy the writer already uses for tracks. You save a Set in
Live 12.4 with one audio clip, and the writer only sets values on known paths.

- **For:** all 50 children are exactly Live's, and a Live upgrade only needs
  a template refresh (`troubleshooting.md:224-227`).
- **Against:** it needs you in Live to regenerate `blank_live12.xml`, which
  touches the whole writer. It also has to scrub the second `FileRef` under
  `SourceContext` (a stale path from your machine). It is more change than
  the bug needs. Fall back to B only if step 2 shows that a defaulted element
  (`Fades`, `Sync`, `HiQ`, …) audibly matters.

### C. Stop emitting clips

The 2026-05-30 decision made static exports "columns-only", and
`als-export.md:39` already calls these clips "scaffold decoys".

- **For:** the problem disappears, and there are fewer emitted elements to keep correct.
- **Against:** the export stops being a playable snapshot, which contradicts
  the "let's DJ" runbook. The in-flight beat-grid harness also loses the
  plumbing check that it bases on our own exports. And we lose the one load
  path where markers travel *with* the clip, which may be warp-guard's cheapest fix.

## Recommendation

Do **A**, in this order:

1. Measure the baseline (steps 1-2 below). This needs no code and answers
   "what does Live do today" and "which `RelativePathType` does Live use".
2. Land A with the conformance tests, and correct the four parity claims.
3. Re-export after the BPM fix lands. No writer change is needed.
4. Add downbeat anchoring after the beat-grid gate, as a separate small
   change under the same tests.

## Decisions for you

1. **A, B or C.** I recommend A.
2. **`IsWarped=true` + Beats + markers from dance's analysis** (grid owner
   (ii)), instead of leaving the grid to Live's auto-warp. I recommend yes.
3. **`LaunchQuantisation`:** `5` (an explicit 1 Bar, which was the comment's
   intent) or `0` (Global, which inherits the template's `GlobalQuantisation=4`
   = 1 Bar and matches the bridge's deck clips, `bridge.py:945-948`). Both
   sound the same today. `5` survives a changed global setting. I recommend `5`.
4. **Commit a names-only skeleton fixture** extracted from Live's bundled set:
   about 70 tag names, with no values or paths. I recommend yes.
5. **Launch point once anchoring lands:** the first downbeat (`LoopStart=0`)
   or the file start. I recommend the downbeat.

## Migration cost

- **Code:** about 40 lines in `writer.py`. `generator.py` changes only if
  step 2 shows Live wants a document-relative `RelativePath`, because the
  writer would then need `out_path`.
- **Tests:** two assertions change from `./ColorIndex` to `./Color`
  (`tests/test_als_generator.py:291`, `:527`; both read the *track*-level
  invented sibling). Add one new file of about 80 lines, a JSON fixture, and
  a ~40-line extractor script.
- **Docs:** `als-export.md:43` (WarpMarkers), plus the four parity claims above.
- **Data:** none. Existing `.als` files stay as they are until re-exported.
  The in-flight `.als` reader on the beat-grid branch already reads
  `IsWarped` first and falls back to `Warp`, and falls back from `FileRef/Name`
  to the `Path` basename. It works with either writer.
- **You:** one 20-30 minute Live session for verification, as CLAUDE.md
  requires ("don't change `.als` writer emission without re-verifying in Live").

## Verification plan (Live 12.4 Standard)

**1. Baseline, before any code.** Export a track that has stems, then open it:

```bash
dance export-als <id>                                   # prints the .als path
open -a "Ableton Live 12 Standard" "<path>"
```

In Live, select the Drums A clip in slot 1 and write down: whether the Warp
button is lit, Seg. BPM, the Launch → Quantization field (does `4` display as
"2 Bars"?), and the clip colour. Then raise the master tempo by 8 BPM and fire
scene 1. Warped stems speed up; unwarped stems do not.

**2. Round-trip.** Wait 30 s, then use *File → Save Live Set As…* to save to
`/tmp/als-rt/roundtrip.als`. Then:

```bash
mkdir -p /tmp/als-rt
gunzip -c "<path>" > /tmp/als-rt/ours.xml
gunzip -c /tmp/als-rt/roundtrip.als > /tmp/als-rt/live.xml
for f in ours live; do
  xmllint --xpath '(//AudioTrack)[1]//AudioClip/*[self::IsWarped or self::Warp or self::Color or self::ColorIndex or self::LaunchQuantisation or self::WarpMarkers]' /tmp/als-rt/$f.xml; echo
done
xmllint --xpath '(//AudioTrack)[1]//AudioClip/SampleRef/FileRef' /tmp/als-rt/live.xml
```

(I checked both `xmllint` queries against a writer-built Set and Live's
reference file. The first prints `ColorIndex=1`, `LaunchQuantisation=4`,
`Warp=true` for ours.) The round-trip shows Live's default for a missing
`IsWarped`, and whether it kept our `WarpMarkers` (unchanged means no
re-analysis). It also gives the exact `RelativePathType`/`RelativePath`/`Type`
that Live writes for an external absolute file, which A then copies.

**3. Candidate.** Repeat steps 1-2 on the branch. Pass criteria: the Set loads;
all 5 clips play; the Warp button is lit with Beats mode; the clip colours
match the track colours; Launch shows the chosen quantisation; and the
round-trip `AudioClip` diff contains only elements Live *added* with defaults,
with none of ours dropped or rewritten.

**4. OSC read-back.** Stop the backend first, because it owns port 11001. A
report-only `scripts/als_clip_probe.py --scene 1`, built like
`warp_probe.py`'s `ask()`, queries every deck column with
`/live/clip/get/{warping,warp_mode,color_index,launch_quantization,length}`
at t=0 s and t=20 s. The 20 s point is past warp-guard's 13-15 s
auto-warp window. Pass criteria: `warping=1`, `warp_mode=0`,
`color_index == STEM_COLOR_INDEX[kind]`, and `length` equal across the 4
stems and mix, equal to the export's `CurrentEnd`, and unchanged at 20 s.
If the AbletonOSC fork is active, `warp_probe.py --scene 1` also prints each grid.

## Conformance tests

Add a new file, `tests/test_als_clip_conformance.py`:

- `tests/fixtures/live12_clip_skeleton.json` contains tag names and order for
  `AudioClip`, `SampleRef`, `FileRef` and `Loop`. It is generated by
  `python scripts/als_skeleton.py "<Sample Reference.als path>"`.
- Every child we emit under those parents is in the skeleton, in Live's
  relative order. `AudioTrack` has no `ColorIndex`. The `FileRef` shape is
  also checked against the template's own `FileRef` (`blank_live12.xml:3138`),
  which needs no fixture.
- Clip values: `IsWarped=true`, `WarpMode=0`, `Color == STEM_COLOR_INDEX[kind]`
  (equal to the track `Color`), and `LaunchQuantisation` set to the chosen value.
- The grid: the trailing pair is 1/32 beat apart, its slope equals `bpm`, and
  it stays finite when `duration_seconds=None`.
- `test_skeleton_matches_installed_live` re-extracts from the installed app
  and diffs the result against the fixture. It is `skipif` when the app is
  absent, as in CI. It catches a Live upgrade the same way the template
  refresh note does.

Per CLAUDE.md, implementation waits for your go-ahead.
