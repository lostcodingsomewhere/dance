# Live compatibility journal

What dance emits into Ableton Live, which Live build it was checked against, and how. Tests only prove the `.als` XML has the shape we intend; **only Live can say it accepts it** (CLAUDE.md: "Don't change `.als` writer emission without re-verifying in Live"). This page is where that verification gets written down, so "verified" always has a date, a build and a method behind it.

Run `dance doctor` first: it prints the installed Live builds, the template's `Creator`/`Revision`/SHA-256, and which AbletonOSC handlers are missing. The facts below were captured from it on 2026-10-13.

## Pinned versions

### `.als` template

| | |
|---|---|
| File | `src/dance/als/templates/blank_live12.xml` (a decompressed blank Set saved by Live) |
| SHA-256 | `05cfb2cb1ad7db34e63c410601d745dda69d6fb6a92c77bdd9e81311ff1c5ba1` |
| Root attributes | `MajorVersion="5" MinorVersion="12.0_12402" SchemaChangeCount="2" Creator="Ableton Live 12.4" Revision="d85a94ab5eb1299391a7ff9400bdb8c0dcd7cc75"` (mirrored in `writer.py` `ABLETON_ROOT_ATTRS`) |
| Captured | 2026-05-10, commit `bb2a895`; unchanged since |

### Live installs on this Mac

| App | `CFBundleShortVersionString` | Bundle id | Status |
|---|---|---|---|
| `Ableton Live 12 Standard.app` | `12.4 (2026-04-24_d85a94ab5e)` | `com.ableton.live` | **Target.** Build hash `d85a94ab5e` is the prefix of the template's `Revision` — the template was saved by exactly this build. |
| `Ableton Live 12 Lite.app` | `12.0.2 (2024-04-09_491da1ed88)` | `com.ableton.live` | **Not supported.** Older than the template and track-capped. |

Both apps register `com.ableton.live`, so `open x.als` or a Finder double-click can pick Lite. Always open Sets with `open -a "Ableton Live 12 Standard" <file>.als`.

### AbletonOSC Remote Script

Installed at `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC`: a **shallow git clone** of `https://github.com/ideoforms/AbletonOSC.git` at **`0ca68214bd62c9b5cb641ca34006cfd70ba94430`** (`master`), cloned 2026-05-10 (`docs/abletonosc_setup.md` clones upstream HEAD unpinned, so a fresh install today may get a different commit). Going by file modification times, only `abletonosc/clip_slot.py` was edited after the clone (2026-05-17). Every other source file dates from the clone.

| Patch dance relies on | Documented in | Written | In the installed copy? | Verified in Live |
|---|---|---|---|---|
| `create_audio_clip` in `clip_slot.py` `methods` | `abletonosc_setup.md` § auto-loading stems | 2026-05-17 (`cd41b9c`) | **yes** | In use since 2026-05-17, when `cd41b9c` built the load-track flow on it. |
| `clip_slot_callback` logging fix | `abletonosc_setup.md` § Fix the `clip_slot_callback` logging bug | 2026-06-14 (`b5e8d21`) | **no** (the `logger.info(track_index, …)` bug is still there) | no record |
| `/live/track/{get,set}/crossfade_assign` | `abletonosc_setup.md` § crossfader assignment | 2026-06-13 (`70e8982`) | **no** | no record |
| `/live/song/{get,set,start_listen,stop_listen}/crossfader` (master crossfader) | **nowhere.** Commit `63f6b0b` (2026-08-01) describes it, but the handler source was never committed to this repo. | 2026-08-01 | **no** | No: "inert until Live reloads Remote Scripts" (`63f6b0b`). |
| `/live/clip/{add,move,remove}_warp_marker`, `/live/clip/get/warp_marker_times` | `proposals/warp-guard.md` § Option C (describes it; **no handler source**) | 2026-08-01 (`d54c002`) | **no** | No: marked UNVERIFIED AGAINST LIVE in `client.py` |

The commits for the last four say they patched "the fork", but none of those patches are in this machine's copy. Without them Live drops the messages and the bridge silently no-ops: the crossfader does nothing on app-loaded decks, and the on-screen master crossfader never moves. The master-crossfader and warp-marker handlers have to be recovered from wherever they were applied, or rewritten, and then documented in `abletonosc_setup.md`. `dance doctor` flags all of this; after patching, fully quit and reopen Live, because the doctor reads the source on disk and can't see what Live has loaded.

## `.als` emission matrix

Each row is something `src/dance/als/writer.py` writes into a Set today. "Covering test" means `tests/test_als_generator.py` asserts it at the XML level. "Verified in Live" means someone opened a generated Set in real Live and looked. Sources: `git log -- src/dance/als` commit bodies and `docs/proposals/als-deck-name-prefix-mismatch.md`.

| Emission | Introduced | Covering test | Verified in Live | Method / evidence |
|---|---|---|---|---|
| Template-based Set; root attrs copied from the 12.4 template | `bb2a895` 2026-05-10 | `test_writer_emits_root_ableton_with_known_attrs`, `test_generator_output_is_valid_gzipped_xml` | **2026-05-10, Live 12.4** | Opened the generated Set: "opens in Live 12.4, tempo correct, audio plays, locators visible" (`bb2a895`). |
| `FileRef`: absolute `Path` plus a leaf `RelativePath Value=…` | `61432ea` / `bb2a895` 2026-05-10 | `test_writer_clip_references_actual_file_path`, `test_generator_stem_to_track_mapping` (Path only) | **2026-05-10, Live 12.4** | A real-Live dry run rejected the child-element form ("Required attribute 'Value' missing"). The fixed Set opened (`61432ea`) and played audio (`bb2a895`). |
| Master tempo in `Tempo/Manual` **and** the tempo `AutomationEnvelope` anchor | `bb2a895` 2026-05-10 | `test_writer_emits_one_main_track_with_correct_tempo` (Manual only; the envelope anchor is untested) | **2026-05-10, Live 12.4** | Found in Live: with Manual alone, clips played at the template's 120. |
| Pointee Id renumbering of cloned AudioTracks + `NextPointeeId` bump | `bb2a895` 2026-05-10 | none | **2026-05-10** (5 clones) | Live refuses Sets with non-unique Pointee IDs, so the Set opening proves it held at the time. The 10-clone case is covered only by the 2026-05-30 open. |
| Track colors (`Color` + `ColorIndex`): drums 1, bass 2, vocals 4, other 10, mix 13 | `bb2a895` 2026-05-10 | `test_writer_track_color_palette`, `test_generator_track_colors` | **2026-05-10, Live 12.4** | Palette retuned by direct observation (`bb2a895`), on the 5-track layout. Per-kind values unchanged since. |
| Locators, one per track-level region | `e1aa81c` / `bb2a895` 2026-05-10 | `test_writer_emits_one_locator_per_entry`, `test_generator_locator_positions_in_beats`, `test_regions_to_locators_*` | **2026-05-10, Live 12.4** | "locators visible" (`bb2a895`). |
| 10 deck AudioTracks in `DECK_ORDER`, named `Drums A` … `Mix B` | `66c7c5a` (9) 2026-05-25, `cbe4ba5` (10) 2026-05-26 | `test_writer_emits_ten_deck_audio_tracks`, `test_writer_track_color_palette`, `test_generator_handles_missing_stem_kinds_gracefully` | **2026-05-30**, Live build not recorded (Mac mini rig) | An exported Set was opened; the bridge read back `Drums A … Mix B` and adopted all 10 columns (`proposals/als-deck-name-prefix-mismatch.md`). |
| A-side decks + `mix_a` carry a clip in slot 0; B-side decks + `mix_b` ship empty | `66c7c5a` / `cbe4ba5` | `test_writer_b_side_decks_have_empty_clip_slots`, `test_generator_stem_to_track_mapping` | **2026-05-30**, indirect | Resync on the opened Set reported `scanned:5`, consistent with four A-side stems plus `mix_a`. |
| Bare clip names (`Drums`, `Bass`, …) | `e1aa81c` | `test_generator_stem_to_track_mapping` (by track) | **2026-05-30** | Resync reports them `unmatched`. That is accepted behaviour: static Sets are columns-only. |
| `CrossFadeState/Manual`: A-side 0, B-side 2 (mix decks included) | `66c7c5a` 2026-05-25, changed in `cbe4ba5` | `test_writer_crossfader_assignment_per_side` | **unverified** | `66c7c5a`: "Live verification still pending". The 2026-05-30 notes don't mention the crossfader. |
| Clip loop on: `LoopOn=true`, `LoopEnd` = clip end | `bcdd659` 2026-05-17 | none | **unverified** | `bcdd659`: "Real-set verification … left for manual testing". |
| Clip `LaunchQuantisation=4` (1 bar) | `55c91b6` 2026-05-26 | none | **unverified** | No Live check recorded. |
| `mix_a` clip `Disabled=true` (muted reference) | `e1aa81c` 2026-05-10 | `test_writer_mix_clip_is_muted_by_default` | **unverified** | Never confirmed separately in Live. |
| Warp: `Warp=true`, `WarpMode=0` (Beats), two `WarpMarker`s pinned to the analyzer BPM, `MarkersGenerated=true` | `e1aa81c` / `bb2a895` | none | **partly** | The Set loads and plays (2026-05-10). Whether Live keeps our two-marker grid or re-runs auto-warp is unverified; `proposals/warp-guard.md` shows Live's auto-warp misreads stems. |

**Phase 0 is still open.** "Open a generated `.als`, hit play, hear the Edifiers; solo and hear only the Bose" (`docs/session-1.md` Part 1) has been unticked since May. Nobody has listened end to end to the current 10-deck Set (LoopOn, 1-bar quantization, crossfader assignment) in Live 12.4.

## How to verify (and record) a change

1. `dance doctor`: confirm Standard 12.4 is the build you'll open and note the template SHA-256.
2. `dance export-als <id>`, then `open -a "Ableton Live 12 Standard" <file>.als`.
3. Check what you changed, plus the cheap invariants: loads with no error dialog; 10 tracks `Drums A … Mix B`; master tempo equals the track's BPM; locators present; an A-side clip plays audio; `mix_a`'s clip is deactivated; moving the crossfader A→B swaps sides; a fired clip waits for the bar; the loop wraps.
4. Add a row to the log below and update the matrix (date, Live build string, what you looked at). If you refreshed the template, update the SHA-256 and root attributes above as well.

## Verification log

| Date | Live build | What | Result |
|---|---|---|---|
| 2026-05-10 | 12.4 | Template-based writer, 5-track layout (`bb2a895`) | Opens; tempo, audio, locators, colors OK |
| 2026-05-30 | not recorded (Mac mini) | 10-deck export, column adoption | Opens; bridge adopted all 10 columns |
| 2026-10-13 | 12.4 (2026-04-24_d85a94ab5e), from Info.plist | `dance doctor` first run (read-only; Live not opened) | Template build matches the installed Standard. Lite 12.0.2 shares the bundle id. AbletonOSC is missing 9 of the 69 addresses dance sends, and the logging fix is absent. |
