# `.als` clip elements: write what Live 12.4 reads

**Status:** PROPOSAL, not implemented. Waiting for your go-ahead (CLAUDE.md rule 1: it changes the `.als` writer's emission shape).
**Date:** 2026-10-13
**Touches:** `src/dance/als/writer.py` (`_populate_audio_track`, `_build_audio_clip`, `_add_file_ref`), `tests/test_als_generator.py`, a new test file and fixture, docs. No DB, API, OSC, or Stage change.
**Related:** [`warp-guard.md`](warp-guard.md), [`../live-compat.md`](../live-compat.md), [`../als-export.md`](../als-export.md), [`als-deck-name-prefix-mismatch.md`](als-deck-name-prefix-mismatch.md)

---

## The problem, measured

The writer builds each `<AudioClip>` by hand (`writer.py:471-574`). Two of the
elements that carry meaning, `<Warp>` (`:533`) and the clip's `<ColorIndex>`
(`:493`), are names that Live 12.4 never writes.

**Reference.** Live ships its own Sets inside the app. I read all 26 under
`/Applications/Ableton Live 12 Standard.app/Contents/App-Resources/` read-only, using
throwaway lxml scripts in the session scratchpad. No Ableton content is in the
repo. 25 of the 26 say `Creator="Ableton Live 12.4d1"` with
`MinorVersion="12.0_12402"`, the same schema as our template
(`writer.py:55`). The 26th is the 12.1 `DefaultLiveSet.als` (`12.0_12117`), which has
one `MidiClip` and no audio clips.
Between them they hold **2878 `AudioClip`s, and every one has the same 50 children
in the same order.** The largest file,
`Core Library/Ableton Folder Info/Sample Reference.als`, has 2679 of them. In
all 26 files `<Warp>` appears **0** times and `<ColorIndex>` **0** times.
`<IsWarped>` appears on every clip, and `<Color>` appears 3203 times in
`Sample Reference.als` alone.

| dance writes | where | Live 12.4 writes instead |
|---|---|---|
| clip `<Warp Value="true">` | `writer.py:533` | `<IsWarped>`, on 2878/2878 clips |
| clip `<ColorIndex>` | `:493` | `<Color>`, one per clip |
| track `<ColorIndex>` sibling | `:414-419` | nothing. The track `<Color>` is already set at `:411-413` |
| `<Gain Value="0">` | `:528` | nothing under any `AudioClip` (`Gain` occurs only inside devices). Clip gain is `SampleVolume`, a linear factor: 1 on 2664/2679 clips, and values like `1.99526238` (+6 dB) and `0.7943282127` (−2 dB) on the rest. We already write it (`:531`) |
| `<GranularityResolution>` | `:527` | nothing. The nearest element is `TransientResolution` (`6` on all 2679) |
| `FileRef` `Name`, `HasRelativePath`, `HasSearchHint`, `SearchHint`, `DefaultDuration`, `DefaultSampleRate` | `:565`, `:570-574` | nothing under `FileRef`. The last two belong one level up on `SampleRef`, where we also write them (`:513-514`) |

Elements Live writes that we leave out: `IsWarped`, `Color`,
`SampleRef/SamplesToAutoWarp` (`0` on all 2878 clips), and the `FileRef` children `RelativePathType`,
`LivePackName`, `LivePackId` and `SourceHint`. We also leave out 31 other
`AudioClip` children (`TimeSignature`, `Fade`, `Fades`, `Sync`, `HiQ`, …).

Elements we share with Live but write with a different value or shape:

- **`LaunchQuantisation`.** We write `4` (`:502`). The comment above it (`:497-498`) says 4 means
  "1 bar", but that is the *Song* enum (`clip_trigger_quantization`: 0 None, 1 8 Bars,
  2 4 Bars, 3 2 Bars, **4 1 Bar**). The *Clip* enum is 0 Global, 1 None, 2 8 Bars,
  3 4 Bars, **4 2 Bars**, 5 1 Bar ([LOM Clip](https://docs.cycling74.com/apiref/lom/clip/),
  [LOM Song](https://docs.cycling74.com/apiref/lom/song/)). The bridge
  already learned this the hard way (`bridge.py:936-948`). Live's own files fit the
  per-element enums (inferred: the XML enum is not documented). 2866 of the 2878
  bundled clips hold `0` and the other 12 hold `1`; `0` only makes sense
  as "Global", Live's default for a new clip. 25 of the 26 Sets hold
  `GlobalQuantisation=4`, Live's default of 1 Bar in the Song enum, and so does our
  template (`blank_live12.xml:5321`). *Inferred:* exported clips launch on 2-bar
  boundaries today, while app-loaded deck clips (Global) launch on 1-bar boundaries.
- **`FileRef/Type`.** We write `1` (`:566`). Live writes `2` on every `SampleRef/FileRef` (2878/2878), and on all four of the template's `FileRef`s, which point at device presets. So `Type` is not the "audio sample" flag our comment says it is (its meaning is undocumented); copy Live's `2`.
- **`FileRef/RelativePath`.** We write the parent *directory*, without its leading slash
  and without a `RelativePathType` (`:558-563`). Live writes a path to the *file*,
  relative to a base that `RelativePathType` picks. Across *all* `FileRef`s in the
  bundled files (devices and presets included) the values are 0 (empty, absolute `Path`
  only), 1 (relative to the Set, `../../…`), 5 (Core Library pack), 6 (User Library)
  and 7 (app built-ins). A public example shows 3 for a sample inside the Set's
  project folder
  ([ableton-under-the-hood](https://github.com/Sample-Offline/ableton-under-the-hood)).
  The meanings are inferred from the paths, not documented. **Every one of the 2878
  clip `SampleRef/FileRef`s is `5`** (Core Library samples), so the bundled clips
  give no example of an external file like our stems; step 2 of the verification
  plan has to supply it.
- **`SampleRef/SourceContext`.** We write a leaf `<SourceContext Value=""/>` (`:511`). Live writes a
  class: `SourceContext/SourceContext` holding `OriginalFileRef`,
  `BrowserContentPath` and `LocalFiltersJson`.
- **Order.** Live puts `WarpMarkers` after `SampleVolume`, and we put it right after `SampleRef`
  (`:516`). In `FileRef`, Live writes `Path` before `Type`.
- **Grid shape.** We write `(0 s, beat 0)` and `(duration, duration·bpm/60)`
  (`:516-524`). When `duration_seconds` is missing, the end marker becomes `(1.0 s, 16 beats)`,
  a **960 BPM** grid (`:480`, `:523`). Live encodes a constant tempo as a tiny first segment.
  2527 of its 2679 clips have exactly `(0, 0)` and `(60/(32·bpm) s, 0.03125)`,
  2670 end on a 1/32-beat pair, and Live extends the last segment's slope past it.

Live's skeleton (names only, Live's order). `AudioClip` children 1-28 match the
template's own `MidiClip` exactly (`blank_live12.xml:5424`), and its `FileRef`
matches the template's (`:3138-3148`). Only the audio tail and `SampleRef` exist
nowhere in the repo:

```
AudioClip  1-28 (shared Clip base, = template MidiClip):
  LomId LomIdView CurrentStart CurrentEnd Loop Name Annotation Color LaunchMode
  LaunchQuantisation TimeSignature Envelopes ScrollerTimePreserver TimeSelection Legato
  Ram GrooveSettings Disabled VelocityAmount FollowAction Grid FreezeStart FreezeEnd
  IsWarped TakeId IsInKey ScaleInformation AutomationEnvelopesListWrapper
AudioClip 29-50 (audio tail):
  SampleRef Onsets WarpMode GranularityTones GranularityTexture FluctuationTexture
  TransientResolution TransientLoopMode TransientEnvelope ComplexProFormants
  ComplexProEnvelope Sync HiQ Fade Fades PitchCoarse PitchFine SampleVolume
  WarpMarkers SavedWarpMarkersForStretched MarkersGenerated IsSongTempoLeader
SampleRef: FileRef LastModDate SourceContext SampleUsageHint DefaultDuration DefaultSampleRate SamplesToAutoWarp
FileRef:   RelativePathType RelativePath Path Type LivePackName LivePackId OriginalFileSize OriginalCrc SourceHint
Loop:      LoopStart LoopEnd StartRelative LoopOn OutMarker HiddenLoopStart HiddenLoopEnd
```

Diffing a writer-built Set against it: we emit 21 `AudioClip` children, and 17 of
them are Live's. Our `Loop` and `SampleRef` match Live's order, but `SampleRef`
is missing `SamplesToAutoWarp`. Only 5 of our 11 `FileRef` children are Live's.

## What Live does with it today

The Set opens: Live 12.4 checked it on 2026-05-10 (`live-compat.md`,
log). So Live **tolerates names it doesn't know** (presumably ignoring them) and **fills
in defaults for names that are missing**. The comment at `writer.py:415` already relies on
this. Inferred consequences, none measured:

- **A clip's warp state is whatever Live defaults a missing `IsWarped` to.** Nobody has
  measured it. The beat-grid branch's reader assumes `true` when both elements are
  missing (`claude/grid-metrics:src/dance/als/reader.py:90-95`), and that is a guess too.
  The check CLAUDE.md prescribes ("tempo correct, audio plays") cannot tell the
  difference. The writer uses one BPM for the master tempo and for the grid
  (`generator.py:171-176`), so at the exported tempo the stretch ratio is 1.0 and a
  warped clip sounds the same as an unwarped one.
- **A clip's colour is probably Live's default, not `STEM_COLOR_INDEX`.** Track colours
  are right because the track `<Color>` is set.
- **Exported clips probably launch on 2-bar boundaries**, as described above.
- `Gain`, `GranularityResolution` and the extra `FileRef` fields do nothing. Live finds
  the stem by its absolute `Path`.
- Whether Live re-runs auto-warp on our clips is **unknown**. `SamplesToAutoWarp`, which we
  omit, sounds like that very switch. Live writes `0` everywhere, but the default for a
  missing element is unknown.

Five places claim that the two load paths agree on warp, which the XML probably doesn't deliver:
`warp-guard.md:52-54`, `LEARNING.md:156-157`, the `_force_warp` docstring
(`bridge.py:2047-2050`), `tests/test_osc.py:780-782`, and the `live-compat.md:60` row
(which lists `Warp=true` as emitted, accurately, but not that Live ignores it).

## Interactions

**warp-guard.** Its measurements (a mis-warped stem on 4 of the first 4 tracks, the bass on 3 of 5)
cover the **OSC path**, where Live auto-warps every fresh `create_audio_clip`. The `.als`
path has never been measured. If Live honours a saved `IsWarped`+`WarpMarkers` grid, then
exported stems never go through auto-warp, and every stem of a track shares one grid by
construction, so warp-guard's failure can't happen on this path. It is also the only
grid control that doesn't depend on unverified fork handlers. `live-compat.md:37` (on
this branch) still says the warp-marker OSC handlers are missing from the installed
AbletonOSC; since `9ce8ac3` on `main` they are vendored as patch `0005` and were
applied on this Mac, but the commit records that their behaviour in Live is **not yet
verified**. Verification step 1 tests this
directly, on a track whose bass Live mis-warped over OSC. warp-guard's Option C and this writer
should build the grid with **one shared function**, so the two paths agree by
construction and not just by docstring.

**The BPM fix (merged in `d2ab400`).** The markers encode `audio_analysis.bpm`. Until
`5b9a65f`, every stored tempo came from librosa's fallback, quantised to `5168/lag`:
120.19, 123.05, 126.05, 129.20 ("282 tracks shared 26 BPMs"). These are warp-guard's "our analysis"
column. Near 125 BPM the steps are 3 BPM wide, so the error can reach ±1.5 BPM
(±1.2 %). The commit's real examples are 126.048 → 124.999 and 123.047 → 124.002, both
about 0.8 %. The grid-metrics harness measured the consequence: a dance export scored
F1@70 ms 0.27-0.30 against dance's own `beats` table, while a Set whose warp markers
*were* those beats scored 0.996 (`c48428f` on `claude/grid-metrics`, three real tracks).
That measures self-consistency against dance's beats, not against ground truth. The
warp flag doesn't change that error. At 0.84 % a stem
slips a full beat against a correctly gridded deck in 32 bars, warped or not. What
changes it is re-deriving the tempo. Every Set exported before the fix carries the old
grid. The fix needs no writer change: `dance reprocess --from analyze --all --skip-embeddings`
(`docs/cli.md:69-78`), then `dance build-graph` and `dance export-als --all`, the sequence
`cli.md` prescribes. That re-analysis should happen
**before** step 3 of the verification below, so the grid being verified is the
precise one.

**Who owns the grid (rule 3, Ableton-first).** There are four candidates:

1. **dance's analysis, written as `IsWarped=true` plus markers.** The user can re-warp it in
   Live, and when they do, Live's version wins, because that is the Set they save.
2. **Live's auto-warp** (omit the markers, or let `SamplesToAutoWarp` trigger it).
   warp-guard measured this as wrong on isolated stems: 9 % drift, or half or double tempo.
3. **`IsWarped=false`.** Stems play at their native speed and stay in sync only at the
   exported tempo. Nudging the master tempo, or beatmatching a second deck against them,
   breaks them.
4. **Live's warp of the full mix, stamped onto the sample-aligned stems.** Live gets mixes
   right. This is the most Ableton-first option, but capturing and storing Live's grid
   is a schema change, so it needs its own proposal.

Rule 3 says Live wins "when the audio engine and the analyzer disagree". On stems, Live's
auto-warp disagrees *with itself* across the stems of one track, so it is not the
authority there. Recommend (1) now, and let the grid-metrics harness compare it against
Live's own grids: mixes warped by Live, and Sets you correct by ear. Move to (4) if they disagree.

**Downbeat anchoring (after the beat-grid work).** Beat 0 is currently `SecTime 0`, the
file start. Launch quantisation therefore lands the *file start* on the bar, so any
lead-in puts the stems' bar lines `d` seconds late against the other deck. Live's own
files show the encoding for "1.1.1 is later than the file start". *Game Time B 101 bpm*
has `(0.4901 s, 0)`, `(0.5086 s, 0.03125)`, `LoopStart=0`, and the pre-roll is
extrapolated to negative beats. *Vocal Demo Hum Low* writes the pre-roll marker explicitly:
`(0 s, −3.35)`, `(1.522 s, 0)`, …. For a first downbeat at `d` seconds we would write
`(d, 0)`, `(d + 60/(32·bpm), 1/32)`, set
`CurrentEnd = LoopEnd = OutMarker = (duration − d)·bpm/60`, and leave `LoopStart` at 0.
Three things to watch:

- **Locators** (`markers.py`) must subtract `d`. `LocatorEntry` rejects negative beats
  (`markers.py:29-31`), and `detect_regions` always emits a cue at 0 ms
  (`emit(0, "Start")`, `detect_regions.py:231`), so those regions need clamping or dropping.
- **Pre-roll.** With `LoopStart=0` at `d`, the first `d` seconds sit before the clip
  start and don't play. Live's *Vocal Demo Hum Low* keeps its pre-roll reachable through
  a negative `HiddenLoopStart` (`−3.35`); whether stems need their lead-in is part of
  decision 6.
- **`d` isn't trustworthy yet.** `beats.downbeat` is set on every 4th librosa beat
  counted from the first detected beat (`detect_regions.py:167`), so the bar phase is a
  guess. Anchoring therefore waits for a downbeat source that passes the grid-metrics
  gate (bar-phase and downbeat accuracy, `claude/grid-metrics`, not merged).

None of this changes the element set. The conformance tests below assert "first marker is
at beat 0, slope = bpm" rather than `SecTime 0`, so anchoring can land later without
rewriting them.

**Colour values.** Live 12.4 stores a clip's colour as an integer `<Color Value=N>`. In the
reference, clip values span 0-57, tracks 1-27, and scenes `-1` (no colour), which fits an
index into the 70-entry palette (0-69) that LOM `Clip.color_index` and AbletonOSC's
`/live/clip/get/color_index` ("0-69", [README](https://github.com/ideoforms/AbletonOSC))
expose. *Inferred:* that the XML number and the LOM index are the same; nothing documents
it. LOM `Clip.color` is RGB and snaps to the nearest palette entry. `STEM_COLOR_INDEX`
(drums 1, bass 2, vocals 4, other 10, mix 13) was retuned by eye in Live 12.4
(`bb2a895`, `als-export.md:52-60`), and the only colour element of ours Live could have
been reading then was the track's `<Color>`. So those numbers are already Live 12 `Color`
values, and the clip gets the same number: **there is nothing to convert.** No Live 12
file I read and no public documentation I found uses `ColorIndex`. *Unverified
recollection:* older Live versions (≤ 10) may have used `ColorIndex` for the same
palette, in which case Live might honour it only when upgrading an old-schema Set, not a
`12.0_12402` one. Verification step 1 settles it by eye (clip colour vs track colour).
Two stale spots to fix along the way: the module docstring (`writer.py:26-31`) still
lists the pre-`bb2a895` palette 1/7/13/39/25. The OSC path colours *tracks* in RGB
(`bridge.py:850-866`) and never sets a clip colour (`client.set_clip_color` is called
only by its unit test, `tests/test_osc.py:382`). `als-export.md:62` calls that split
intentional, so it is out of scope here.

## Options

### A. Conformance patch ← recommended

Keep the hand-built clip, and write Live 12.4's names, values and order:

- `<Warp>` becomes `<IsWarped Value="true">` (decision 2). `WarpMode=0` (Beats) stays, to match `_force_warp`.
- The clip's `<ColorIndex>` becomes `<Color Value=STEM_COLOR_INDEX[kind]>`. Drop the
  track's `ColorIndex` sibling (`:414-419`) and fix the docstring.
- Drop `Gain`, `GranularityResolution` and the six extra `FileRef` children.
- Add `SamplesToAutoWarp=0`. Make `FileRef` Live's 9 children with `Type=2`,
  `LivePackName=""`, `LivePackId=""` and `SourceHint=""`. Take
  `RelativePathType`/`RelativePath`/`SourceContext` **from Live's round-trip**
  (verification step 2), not from a guess. If Live writes `1` plus a Set-relative
  path, the writer needs `out_path` from the generator. `0` plus an empty path is
  the fallback that keeps today's absolute-path contract.
- Use the grid shape Live uses (the 1/32-beat pair), which removes the 960 BPM fallback.
- Use Live's child order throughout. `LaunchQuantisation` and `MarkersGenerated` follow decisions 3 and 4.

That's about 40 changed lines, and every element is checked against the skeleton.

### B. Clone a Live-written AudioClip

This is how the writer already handles tracks. Save a Set in Live 12.4 that holds one audio
clip, ship it as the template (or as a second one), and let the writer only set values on
known paths.

- **For:** all 50 children are exactly Live's, and a Live upgrade costs only a template
  refresh (`troubleshooting.md:236-240`).
- **Against:** you have to regenerate `blank_live12.xml` in Live, and everything in the
  writer depends on that file (its SHA-256 is pinned in `live-compat.md`). The clone
  would also carry your machine's paths in `SourceContext`/`BrowserContentPath`, which
  must be scrubbed. That is more change than the bug needs. Fall back to B only if step 2
  shows a defaulted element (`Fade`, `Fades`, `Sync`, `HiQ`, …) that audibly matters.

### C. Stop emitting clips

Static exports were declared "columns-only" on 2026-05-30
(`als-deck-name-prefix-mismatch.md`), and `als-export.md:41` calls the shipped
clips "scaffold decoys".

- **For:** the problem goes away, along with every element we'd have to keep correct.
- **Against:** the CLAUDE.md "let's DJ" runbook (`export-als --all`) would no longer
  produce anything playable. The grid harness would lose its plumbing check
  (`grid_diff.py --als`). And we would give up the one load path where the grid travels
  with the clip and bypasses the auto-warp that warp-guard showed is wrong on stems.

## Recommendation

**A**, in this order:

1. **Baseline (verification steps 1-2), before any code.** This answers "what does
   Live do today?", "does Live keep a saved grid?", "what does `MarkersGenerated`
   do?" and "which `RelativePathType` does Live write for an external file?".
2. Land A with the conformance tests, and correct the five parity claims.
3. Re-analyse the library (`dance reprocess --from analyze --all --skip-embeddings`,
   then `dance build-graph`), re-export, then run step 3.
4. Downbeat anchoring, once a downbeat source passes the grid-metrics gate. It is a
   separate small change, under the same tests.

## Decisions for you

1. **A, B or C.** I recommend A.
2. **Warp state.** I recommend `IsWarped=true` with Beats and markers from dance's
   analysis (grid owner 1). It follows tempo changes, matches the OSC path's
   `_force_warp`, and matches Live's own convention: 208 of 238 clips under
   `Samples/Loops` are warped, while 2437 of 2439 one-shots are not. The alternative is
   `IsWarped=false`: no wrong grid can ever be authoritative, but the stems stop
   following the master tempo.
3. **`LaunchQuantisation`.** I recommend `0` (Global). Today that inherits 1 Bar from the
   template's `GlobalQuantisation=4`, it matches the bridge's deck clips
   (`_LAUNCH_QUANT_GLOBAL`, `bridge.py:948`), and changing the transport-bar
   setting moves exported and live-loaded decks together. The alternative is `5`
   (explicit 1 Bar), which survives a changed global setting. That is what `:499-501`
   meant to protect against, but it was written with the wrong enum.
4. **`MarkersGenerated`.** In `Sample Reference.als`, Live writes `true` on all 2469 unwarped clips and on 149 of
   210 warped ones, and `false` on 61 warped clips (across all 26 Sets: 2479 unwarped
   all `true`; 285 warped `true`, 114 `false`). 10 of the 12 clips whose first
   marker isn't at `(0 s, beat 0)` are `false`, and the other two are offset by less
   than 1 ms. I lean `false` ("these markers were placed, not detected"), but step 2
   round-trips both values first. Keep `true` if nothing differs.
5. **A names-only skeleton fixture**, extracted from Live's bundled Set: 29 tag names (the
   audio tail plus `SampleRef`), with no values or paths. The clip base, `Loop` and
   `FileRef` are checked against the template we already ship. I recommend yes. The
   alternative is a test that reads the installed app and skips when it is absent, which
   means CI checks nothing.
6. **Launch point once anchoring lands:** the first downbeat (`LoopStart=0` at `d`), or
   the file start. I recommend the first downbeat.

## Migration cost

- **Code:** about 40 lines in `writer.py`. `generator.py` changes only if step 2 shows
  Live wants a Set-relative `RelativePath`.
- **Tests:** `tests/test_als_generator.py:291` and `:527` change `./ColorIndex` to
  `./Color`. Both read the *track-level* invented sibling, which already holds the same
  value as the track `Color`. There is a new test file, a 29-name JSON fixture, a
  ~40-line `scripts/als_skeleton.py` that regenerates the fixture, and a corrected
  docstring at `tests/test_osc.py:780-782`.
- **Docs:** `als-export.md:45`, the `live-compat.md` matrix rows (`:51`, `:58`, `:60`)
  plus a log entry, and the parity claims listed above. `warp-guard.md` gets a dated
  correction note; its text stays as it is.
- **Data:** none. Existing `.als` files stay as they are until re-exported. The
  grid-metrics reader reads `IsWarped` first and falls back to `Warp`, and falls back
  from `FileRef/Name` to the `Path` basename (`reader.py:84-95` on its branch), so it
  works with either writer.
- **You:** one Live session of 20-30 minutes. CLAUDE.md requires it: "don't change `.als`
  writer emission without re-verifying in Live".

## Verification plan (Live 12.4 Standard)

Live 12 Lite is installed too and shares the bundle id, so always open Sets with
`open -a "Ableton Live 12 Standard"`. Run `dance doctor` first to confirm the build.

**1. Baseline, before any code.** Export a track that warp-guard flagged (its bass was
mis-warped over OSC), then open it:

```bash
dance export-als <id>                          # prints the path it wrote
open -a "Ableton Live 12 Standard" "<path>"
```

In Live, select the Drums A clip in scene 1 and note: whether Warp is lit; Seg. BPM;
Launch → Quantization (does it say "2 Bars"?); and the clip's colour against the track's.
Then stop the backend (it owns UDP 11001) and read every A-side clip twice, about 1 s
after opening and again after 20 s, which is past warp-guard's 13-15 s auto-warp window.
This uses only stock AbletonOSC getters (run it from the repo root with the venv active):

```bash
python - <<'EOF'
import time
from dance.osc.client import AbletonOSCClient
from dance.osc.listener import AbletonOSCListener
replies = []
lis = AbletonOSCListener(port=11001); lis.on_any(lambda a, args: replies.append((a, args))); lis.start()
cli = AbletonOSCClient(host="127.0.0.1", port=11000); time.sleep(0.3)
def ask(prop, track):
    replies.clear(); cli._send(f"/live/clip/get/{prop}", track, 0); time.sleep(0.4)
    got = [args[2] for addr, args in replies if addr == f"/live/clip/get/{prop}"]
    return got[0] if got else None
for track, kind in [(0, "drums_a"), (2, "bass_a"), (4, "vocals_a"), (6, "other_a"), (8, "mix_a")]:
    print(kind, {p: ask(p, track) for p in ("warping", "warp_mode", "color_index",
                                            "launch_quantization", "length", "sample_length")})
lis.stop()
EOF
```

(I dry-ran this against unused ports: it imports and runs, and prints `None`
for everything when nothing answers.) If `length` is equal across the five clips, equals
the export's `CurrentEnd`, and is unchanged at 20 s, then Live kept our grid. If bass
differs, Live re-warped it. If patch `0005` is applied and Live was fully quit and
reopened (`vendor/abletonosc/README.md` on `main`), `python scripts/warp_probe.py --scene 1`
(report-only without `--repair`) prints each stem's actual markers, which is direct
evidence where `length` is indirect. Finally, raise the master tempo by 8 BPM and fire
scene 1: warped stems speed up, unwarped ones don't.

**2. Round-trip.** In Live, *File → Save Live Set As…* (Live creates a
`… Project` folder), then:

```bash
RT=/tmp/als-rt; mkdir -p "$RT"
gunzip -c "<exported path>" > "$RT/ours.xml"
gunzip -c "<path Live saved>" > "$RT/live.xml"
for f in ours live; do
  echo "== $f"
  xmllint --xpath '((//AudioTrack)[1]//AudioClip)[1]/*[self::IsWarped or self::Warp or self::Color or self::ColorIndex or self::LaunchQuantisation or self::WarpMode or self::MarkersGenerated or self::WarpMarkers]' "$RT/$f.xml"; echo
  xmllint --xpath '((//AudioTrack)[1]//AudioClip)[1]/SampleRef/*[self::FileRef or self::SamplesToAutoWarp or self::SourceContext]' "$RT/$f.xml"; echo
done
```

I ran these exact queries on a writer-built Set and on `Sample Reference.als`. On ours
they print `ColorIndex=1`, `LaunchQuantisation=4`, the two markers `(0,0)`/`(240 s, 496)`,
`WarpMode=0`, `MarkersGenerated=true`, `Warp=true`, and the 11-child `FileRef`. This
step shows Live's default for a missing `IsWarped`, whether it kept our markers, what it
writes for `MarkersGenerated`, and the exact `RelativePathType`/`RelativePath`/`Type`/
`SourceContext` for an external file, which A then copies. Repeat once with
`MarkersGenerated` flipped (`sed -i '' 's/MarkersGenerated Value="true"/MarkersGenerated Value="false"/' "$RT/ours.xml"`,
`gzip -c` it back to an `.als`, reopen and save) to settle decision 4.

**3. Candidate.** Repeat steps 1-2 on the implementation branch, after re-analysis.
To pass: the Set loads with no dialog. All four A-side stems play, and `mix_a` is
deactivated. Warp is lit with Beats, and Seg. BPM equals `audio_analysis.bpm`. Clip
colours equal track colours. Quantization shows the chosen value. The probe shows
`warping=1`, `warp_mode=0`, `color_index == STEM_COLOR_INDEX[kind]`, and equal
`length` at 1 s and at 20 s. The round-trip `AudioClip` diff shows only elements
Live *added* with defaults: none of ours dropped or rewritten.

**4. Record it** in `live-compat.md`: update the matrix rows and add a log line with the date,
the build string, and what you looked at.

**5. When grid-metrics merges:** run `python scripts/grid_diff.py --als <export>` against the
DB. After the BPM fix, the export should score close to the beats-gridded 0.996, not 0.27-0.30.

## Conformance tests

A new file, `tests/test_als_clip_conformance.py`:

- **Names and order.** Every child we emit under `AudioClip` is in Live's skeleton, and
  in Live's relative order (an ordered-subsequence check). Children 1-28 are checked
  against the template's `MidiClip`, and the audio tail against
  `tests/fixtures/live12_clip_skeleton.json`. That fixture is regenerated with
  `python scripts/als_skeleton.py "<…>/Sample Reference.als"`. `SampleRef` is checked
  against the fixture, and `Loop` and `FileRef` against the template (`MidiClip/Loop`,
  `blank_live12.xml:3138`). No `AudioTrack/ColorIndex`, `Warp`, `Gain` or
  `GranularityResolution` anywhere in the output.
- **Values.** `IsWarped=true`, `WarpMode=0`, clip `Color == STEM_COLOR_INDEX[kind] ==`
  the track `Color`, `SamplesToAutoWarp=0`, `FileRef/Type=2`, and `LaunchQuantisation`
  and `MarkersGenerated` at the chosen values.
- **Grid.** The first marker is at beat 0. The last pair is 1/32 beat apart, and its slope
  equals `bpm` to within 1e-6. The grid stays finite with `duration_seconds=None`, as a
  regression test for the 960 BPM case.
- **Cross-path parity.** The writer's `(IsWarped, WarpMode)` equals what `_force_warp`
  sends (`True`, `_WARP_MODE_BEATS`). If decision 3 is Global, its `LaunchQuantisation`
  equals `_LAUNCH_QUANT_GLOBAL`. This follows the existing crossfade parity test
  (`tests/test_osc.py:1262-1279`, which compares the bridge against
  `writer._crossfade_value_for`).
- **Upgrade tripwire.** `test_skeleton_matches_installed_live` re-extracts the skeleton from the
  installed app and diffs it against the fixture. It is `skipif` when the app is absent
  (as in CI), and it catches a Live upgrade the same way the template-refresh note does.

Per CLAUDE.md, implementation waits for your go-ahead.
