# Ableton Live Set (.als) export

Generate a Live Set per track from the analyzed bundle (stems + cues + tempo). Open it in Live and you get five colored audio tracks pre-pointed at the right files, master tempo set, and timeline locators at every detected region.

## Quick start

```bash
dance export-als 42
# -> /Users/you/Music/Dance/Sets/My Song - Some Artist.als
open "/Users/you/Music/Dance/Sets/My Song - Some Artist.als"
```

Or batch:

```bash
dance export-als --all
```

Or via the API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tracks/42/als
```

The React UI's `LoadActions` component (`companion-app/src/components/LoadActions.tsx`) wraps this and reveals the file in Finder on success.

## What the generated .als contains

`src/dance/als/writer.py` (`build_live_set_xml`):

- **5 audio tracks**, in this canonical vertical order:
  - `Drums` — points at `<stems_dir>/<hash>/drums.wav`
  - `Bass` — `bass.wav`
  - `Vocals` — `vocals.wav`
  - `Other` — `other.wav`
  - `Mix` — points at the original full-mix file, muted by default
- **2 Return tracks** preserved from the template (A Reverb, B Delay) so sends are wired up.
- **Master tempo** — set to the track's detected BPM (`audio_analysis.bpm`). Written to both `MainTrack/.../Tempo/Manual` *and* the master-tempo `AutomationEnvelope` anchor (the latter overrides Manual if you forget it).
- **8 scenes**, with our AudioClip in scene 1 of each stem track. The remaining 7 slots are empty so you can drop in alternate loops / variations.
- **WarpMarkers** — two per clip (start + end-of-stem), enough for Live to play the stem at its native tempo without re-warping.
- **Locators** — one per track-level `Region` row, named from `region.name` or auto-named (`Cue 1`, `Intro`, etc. — see `src/dance/als/markers.py`). Per-stem regions are dropped (Live's master timeline can't carry them).

## Color palette

`src/dance/als/writer.py` — Live's built-in palette indices (0-69):

| Kind   | Index | Color (Live 12.4)  |
|--------|-------|--------------------|
| drums  | 1     | orange-red         |
| bass   | 2     | brown / mustard    |
| vocals | 4     | lime               |
| other  | 10    | light blue         |
| mix    | 13    | white              |

Live's 70-color palette is 7 cols × 10 rows; the indices above were chosen empirically by generating Sets and reading off Live. Edit `STEM_COLOR_INDEX` to taste.

The TrackCard in the React UI uses a different scheme (`STEM_TRACK_COLORS` in `osc/bridge.py`) — that's for the live "Push to Live" path, not the .als export. The two paths intentionally diverge so the .als looks like a Live-native Set and the OSC-pushed tracks look distinct from anything Live's own scheme assigns.

## Output location

`settings.als_output_dir`, default `~/Music/Dance/Sets`. Override:

```bash
DANCE_ALS_OUTPUT_DIR=/Volumes/Stage/Sets dance export-als 42
```

Custom `--out`:

```bash
dance export-als 42 --out ~/Music/Dance/Sets/PeakTime.als
```

`out_path` must resolve inside `als_output_dir` — otherwise `AlsOutsideDirError` (`src/dance/als/generator.py`) / 403 over the API. This is a hard safety guard so the API endpoint can't be tricked into writing anywhere on disk.

## How it's built — template injection

The `.als` format is **gzipped XML with a schema Ableton does not publish**. Writing one from scratch is not viable: a blank Live 12 Set is ~189 KB of XML with 570+ distinct element types (Transport, GroovePool, ScaleInformation, MainTrack mixer chain, AutomationEnvelopes, ContentLanes, ...). Live's loader checks hundreds of required fields and enforces invariants like "MainTrack ClipSlotList size == Scenes count".

Instead we **ship a real blank Live 12 Set as a template** (`src/dance/als/templates/blank_live12.xml`, decompressed from a user-saved `Untitled.als`) and surgically inject our content:

1. Load template via `lxml`.
2. Clone the template's sole `<AudioTrack>` as a DOM template; deepcopy it 5 times.
3. **Renumber Pointee IDs** on each clone (`AutomationTarget`, `ModulationTarget`, `Pointee`, ...) to fresh globally-unique values starting at 30000. Without this Live errors with "non-unique Pointee IDs" — the clones share the template's IDs otherwise.
4. Set per-stem `Id`, `Name`, `Color`; inject `<AudioClip>` into the first session ClipSlot pointing at the stem file.
5. Update master tempo in `MainTrack/.../Tempo/Manual` **and** the matching `AutomationEnvelope FloatEvent` anchor (the envelope overrides Manual if you only update one).
6. Replace `Locators` inner content with our entries.
7. Bump `LiveSet.NextPointeeId` above our high-water mark so Live's allocator doesn't collide on next edit.
8. Re-gzip.

Everything else — Transport state, GroovePool, view state, MainTrack mixer, ReturnTracks (A Reverb, B Delay), scenes 1-8 — comes verbatim from the template, so Live's loader sees exactly what it expects.

### What we do **not** emit

- Per-clip device chains (EQs, compressors) — drag in your own.
- Automation envelopes on stem tracks — drawing automation is a manual step.
- View state (zoom, scroll, selection) — Live opens to the default view.
- Embedded sample data — `FileRef Path` is **absolute**. Moving the stems folder breaks the Set.

### If Live rejects the .als

The schema is undocumented and Live's loader is strict. If you see a load error:

1. Capture the exact error — Live usually gives a line and column in the decompressed XML.
2. Unzip and inspect:
   ```bash
   gunzip -c bad.als > bad.xml
   ```
3. Read the indicated line; cross-check against the bundled template (`src/dance/als/templates/blank_live12.xml`) for what shape Live expects.
4. Fix in `writer.py` — typically an attribute mismatch on a class element (e.g. emitting `<TimeSignature Value=""/>` when Live wants `<TimeSignature>` with child elements).

Tests: `tests/test_als_generator.py` parses every emitted Set back through `lxml` and asserts shape (track count, locator count, tempo, etc.). They don't and can't assert "Live will accept this" — only Live can.

## Workflow

```
dance export-als 42
  -> writes ~/Music/Dance/Sets/Track Title - Artist.als

open ~/Music/Dance/Sets/...
  -> Live launches, opens the Set
  -> 5 stem tracks ready, mix track muted
  -> click a clip to play, or launch scene 1 for all stems at once
```

## Limitations

| Limitation                                | Workaround |
|-------------------------------------------|------------|
| Stems referenced by **absolute** path     | Don't move the stems folder. Regenerate the .als after a move. |
| Mix track is muted by default             | Click the mute button to unmute and use the original as a reference. |
| No device chains (EQ, compression, etc.)  | Set up a template Set with your chains; drag clips in. |
| No automation envelopes                   | Drawing automation is a manual step in Live. |
| Locators land on the master timeline      | Per-stem regions are intentionally dropped — Live's locators are master-scoped. |
| One Set per track                         | This is a stem-prep workflow, not a set-building one. For combined sets, drag clips between Sets in Live. |
| Template is a Live 12.4 Set               | If you're on an older Live, the template may not load. Save your own blank Set as `src/dance/als/templates/blank_live12.xml` and the generator picks it up automatically. |
