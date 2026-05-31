# Proposal: reconcile `.als` track names with bridge deck-column recovery

**Status:** fix #1 SHIPPED; fix #2 WON'T FIX (decided 2026-05-30 — see below).
**Found:** 2026-05-30, during first real end-to-end DJ test on the Mac mini.

> **TL;DR (decision 2026-05-30):** A static `.als` is used **only to load the deck
> columns into Live**. Stems are **live-loaded through the app** (⌘K / rec promote →
> `push_track_to_live` over OSC), which names tracks and clips correctly. Fix #1
> (column-name recovery) shipped so columns adopt from an opened `.als`. Fix #2
> (linking `.als` clip names back to library track ids) is **out of scope** — we
> don't need static-export clips to be library-linked. Option 3 below is the
> accepted behavior.

## Symptom

Open an exported `.als` in Live (e.g. `dance export-als 1`), and the companion
app stays stuck on **"Waiting for Ableton deck columns. Open Live and load a
track to populate the grid."** BPM never leaves the default. `POST
/api/v1/ableton/decks/resync` returns `{"scanned":0,"adopted":0}` instantly and
`GET /api/v1/ableton/decks` returns `{"columns":null,"cells":[]}` — even though
Live clearly has the 13-track deck layout loaded and the OSC link is verified
working (reads + writes both confirmed).

## Root cause — a track-name contract mismatch

Two components independently name the per-stem deck tracks, and they disagree:

| Component | Source | Names emitted |
|---|---|---|
| **`.als` writer** (static export) | `src/dance/als/writer.py:134` `_display_for()` | `Drums A`, `Bass A`, `Vocals A`, `Other A`, `Mix A`, … (**no prefix**) |
| **Live bridge** (OSC, runtime) | `src/dance/osc/bridge.py:613` `_DECK_DISPLAY_NAMES` | `Deck Drums A`, `Deck Bass A`, … (**`Deck ` prefix**) |

`recover_deck_columns()` (`bridge.py:494`) adopts a column only when a Live track
name **exactly equals** a `_DECK_DISPLAY_NAMES` value, and only commits if it
finds **all 10**. The static `.als` provides none of those names, so recovery
returns `None`, no columns are cached, and `scan_live_for_cells()` (which only
scans *known* columns) has nothing to scan.

The app's own load path is internally consistent — `push_track_to_live()` →
`set_track_name(idx, _DECK_DISPLAY_NAMES[kind])` (`bridge.py:1455`) creates
`Deck …`-prefixed tracks via OSC, which recovery then adopts fine. Only the
**static-export path** is out of sync.

## Why this needs a decision (not just a patch)

The fix is one of three, and they imply different things about what an exported
`.als` *is*:

1. **Prefix the writer** — change `_display_for()` to emit `Deck Drums A`. Makes
   static exports first-class adoptable deck sets. Risk: changes the `.als`
   emission shape (CLAUDE.md "don't change writer emission without re-verifying
   in Live"); every existing exported `.als` keeps the old names.
2. **Teach recovery the bare names** — accept both `Drums A` and `Deck Drums A`
   (add a legacy/alias set like the existing `_LEGACY_DECK_DISPLAY_NAMES`).
   Lowest-risk; no `.als` change; bridge tolerates both conventions.
3. **Declare static export non-live** — document that `export-als` produces a
   standalone artifact and the live flow is "open template → load via app rec
   banner." No code change; but then the CLAUDE.md "let's DJ" runbook
   (`export-als --all` then perform) is misleading.

**Recommendation: option 2** (alias in recovery). It's the smallest, touches
neither the `.als` emission nor the OSC command contract, fixes the symptom for
all past and future exports, and keeps the `Deck ` prefix as the canonical
runtime name. A one-line addition to the match loop in `recover_deck_columns`
plus a parallel alias map.

## Immediate workaround (no code)

Either path gives a working end-to-end test today:
- **Native live flow:** open the Live **template** (empty deck set) and use the
  app's rec-grid **A/B load buttons** — the bridge creates correctly-named
  `Deck …` columns via OSC and the grid populates.
- (Avoid mixing: don't app-load on top of an opened static `.als`, or you'll get
  duplicate `Drums A` + `Deck Drums A` tracks.)

## Update 2026-05-30 — fix #1 shipped, fix #2 (clip names) still open

**Fix #1 (track-column names) — DONE & verified.** Implemented option 2:
`bridge.py` now has `_DECK_RECOVERY_NAMES` (accepts both `Deck Drums A` and the
bare `Drums A`), and `recover_deck_columns` matches against it. Verified live: the
bridge queried the loaded `.als`, got `Drums A … Mix B`, and adopted all 10
columns (`{drums_a:0 … mix_b:9}`). `GET /decks` now returns `columns` populated
on API startup. 3 regression tests added to `tests/test_osc.py` (bare names,
prefixed names, partial-layout-bails); all pass. **This clears the app's "Waiting
for Ableton deck columns" banner — the grid renders.**

**Fix #2 (clip names) — NEW finding, still broken.** With columns now adopting,
`resync` reaches the next layer and `scanned:5, adopted:0`. Root cause is a
*second* writer↔bridge contract gap, on **clip** names this time:

| Path | Clip name emitted | Source |
|---|---|---|
| Runtime load (OSC) | `"{title} ({kind} {side})"` e.g. `Bad Memories (drums a)` | `bridge.py:1394` |
| Resync parser expects | `"{title} ({kind})"` → reverses to recover title | `ableton.py:571` |
| **`.als` writer** | bare `"Drums"`, `"Bass"`, `"Vocals"`, `"Other"`, `"Mix"` | `writer.py:491` `entry.display_name()` |

The static `.als` clip names carry **no track title at all**, so resync's
`Track.title == title` lookup can't resolve them → `adopted:0`. The cells stay
un-linked to library track ids (so they won't light as "loaded" / be scoped for
recs), even though the audio is correctly in Live and playable.

Also note the runtime format `(kind side)` vs the resync parser's `(kind)` is
itself inconsistent — resync may not even adopt app-loaded clips after a restart.

**Options for fix #2:**
1. **Writer embeds the title** in clip names (`"{title} ({kind} {side})"`, matching
   the runtime format) so resync can reverse it. Changes `.als` emission → must
   re-verify in Live; bloats clip names in the Live UI.
2. **Resync falls back to scene/column position** when the clip name has no title
   — e.g. adopt the cell as "occupied" without a track_id, or map all cells in a
   freshly-opened single-track `.als` to that track. Less precise but no `.als`
   change.
3. **Static `.als` opens are "play-only"** — columns mirror (grid renders, you can
   fire clips) but cells aren't library-linked; library-scoped features (rec
   re-scoring against the combo) only light up for app-loaded tracks. Document
   and accept.
4. **Fix the runtime `(kind side)` vs parser `(kind)` inconsistency** regardless,
   so app-loaded clips survive a restart.

Recommendation: decide #1 vs #3 based on whether static-export sets need full
library-linking or just playback. (#4 is a separate, clear bug worth fixing
either way.) Needs your call before I touch the writer emission or the parser.

### DECISION (2026-05-30): Option 3 — static `.als` is "columns-only", WON'T FIX

The user confirmed: *"we dont care about sync with als file except to load the
columns; otherwise we are live loading stems."* So:

- **Accepted behavior:** opening a static `.als` mirrors the deck **columns** (grid
  renders, clips are playable in Live) but its cells are **not** linked to library
  track ids. Library-scoped features (per-combo rec re-scoring, cell "loaded"
  state) light up only for stems **live-loaded through the app**, which already
  name everything correctly.
- **No change** to the `.als` writer's clip-name emission (option 1 rejected) and
  **no change** to the resync title-parser for the `.als` case.
- The `.als` writer keeps emitting bare clip names (`Drums`, `Bass`, …); resync
  reporting them as `unmatched` is expected and harmless for this workflow.
- **Option 4 — FIXED 2026-05-30.** The runtime loader writes clip names as
  `"{title} ({source} {side})"` (e.g. `"Anthem (drums A)"`; mix → `"(mix A)"`),
  but the resync parser only reversed the legacy `"{title} ({kind})"` form, so
  app-loaded clips never re-adopted after a backend restart (`adopted:0`). The
  parser in `resync_decks` now reconstructs the deck kind from the
  `"{source} {side}"` parenthetical (`"drums a"` → `"drums_a"`) and still accepts
  the legacy form. Tests: `test_resync_adopts_runtime_format_clip_names` +
  `test_resync_adopts_legacy_kind_clip_names` (`tests/test_api.py`).

## Verification when fully implemented

1. `dance export-als <id>`; open in Live.
2. `POST /api/v1/ableton/decks/resync` → expect `scanned:N, adopted:N` (fix #2).
3. App grid mirrors the 5 columns (Drums/Bass/Vocals/Other/Mix) × A/B; BPM 126.
4. Fire a stem from the grid; confirm audio + UI playhead.
