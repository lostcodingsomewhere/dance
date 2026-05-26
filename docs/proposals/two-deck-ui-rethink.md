# Two-Deck UI Rethink (Side-by-Side A/B Layout)

Status: **proposal — not implemented**
Supersedes: `stem-deck-pair.md` §3 "SceneGrid layout — Option C" (compact A/B stacked half-cells). The bridge + writer changes from that proposal **stay** — they're correct. Only the frontend layout is being reconsidered.

Trigger: living with Option C for an hour confirmed your read — A/B stacked half-cells don't match how DJs actually think. We want **two decks side-by-side, like every DJ rig ever built**.

## The thesis (and is it normal?)

**Yes, this is exactly how stem-DJing-on-APC40 was meant to feel.** Three pieces of evidence:

1. **APC40 mk2 hardware.** Confirmed via Akai's Communication Protocol doc: each of the 8 channel strips has dedicated **A** and **B** buttons (right above the fader, next to Solo / Activator / Record-Arm) that assign that strip to crossfader group A, B, or neither. 8 strips × {A, B, none} = the hardware is *literally designed* for the 8-column-stem-deck layout we just put in Live (drums_a/drums_b/bass_a/bass_b/vocals_a/vocals_b/other_a/other_b). Every fader maps 1:1 to a stem deck; every A/B button on the strip lights to match.
2. **DJ-software convention.** Traktor, Serato, Rekordbox all do the same thing: **waveforms stacked horizontally across the top (Deck A above Deck B, time-aligned at a shared playhead), per-deck transport mirrored beneath, crossfader at the bottom-center, library/browser collapsible at the bottom.** Decades of muscle memory live in this layout. Our stacked-half-cell Option C fought all of it.
3. **Your own intuition.** "Almost like 2 decks basically." That's the right mental model. Stems = 4 channels per deck, not 8 stacked rows.

The only novel piece — and the reason this app exists — is that **each "deck" has 4 stem channels** instead of one full-track channel. So our 2-deck UI looks like Traktor's 2-deck UI, but the *deck* is itself a 4-stem mini-mixer.

## Proposed layout (top → bottom)

```
┌─────────────────────────────────────────────────────────────┐
│ MasterStrip (unchanged): BPM, transport, KEY, ARC, RESYNC   │
├─────────────────────────────────────────────────────────────┤
│ ╔═══ DECK A ═══════════════════════════════════════════════╗ │
│ ║  Lioness — Argy Remix      129 · 11A · 5:12             ║ │
│ ║  ┌──────────────────────────────────────────────────────┐║ │
│ ║  │ stacked-stem waveform (4 stems × A side)             │║ │
│ ║  │  ▶ [drums] ───────────────────────────────────────   │║ │
│ ║  │  ▶ [bass]  ────────────────────────────────────────  │║ │
│ ║  │  ▶ [vocals]──────────────────────────────────────    │║ │
│ ║  │  ▶ [other] ──────────────────────────────────────    │║ │
│ ║  └──────────────────────────────────────────────────────┘║ │
│ ║  [PFL A]  vol ▓▓▓▓▓░░  ◢◤ scrub                          ║ │
│ ╚══════════════════════════════════════════════════════════╝ │
│ ╔═══ DECK B ═══════════════════════════════════════════════╗ │
│ ║  (mirror of A)                                            ║ │
│ ╚══════════════════════════════════════════════════════════╝ │
├─────────────────────────────────────────────────────────────┤
│ Column headers:                                              │
│  DRUMS A | DRUMS B | BASS A | BASS B | VOC A | VOC B | OT A | OT B │
│ SceneGrid: 8 columns interleaved by role, A then B per role  │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                  │
│ 1│ d_A│ d_B│ b_A│ b_B│ v_A│ v_B│ o_A│ o_B│                  │
│ 2│ ...│    │    │    │    │    │    │    │                  │
│ 3│ ...│    │    │    │    │    │    │    │                  │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                  │
├─────────────────────────────────────────────────────────────┤
│ Rec banners (4 source roles, each spanning A+B columns):    │
│  [Drums recs]  [Bass recs]  [Vocals recs]  [Other recs]     │
│  [Song recs spans full width — anchor candidates]            │
└─────────────────────────────────────────────────────────────┘
```

### Two-deck waveform strip (replacing ComboStrip)

- One row per deck, **A on top, B on the middle** — matches DJ-software convention exactly.
- Each row shows the song currently anchored on that side (track title, BPM, key, remaining time, beat-aligned playhead).
- Waveform: **stacked 4 stems** (drums on top, bass, vocals, melody) for that side. User picked STACKED in the survey — this is where it lives.
- Per-deck controls inline:
  - **PFL** toggle (headphones monitor this deck — see Cueing section for the catch)
  - Master volume mini-fader (informational; APC40 fader is the real input)
  - Scrub bar (the existing click-to-snap-to-section seeking)

### 8-column scene grid (replacing the half-cell split)

- 8 visible columns, **interleaved by role** so A and B for the same role sit next to each other: `Drums A | Drums B | Bass A | Bass B | Vocals A | Vocals B | Other A | Other B`.
- One cell per (scene, deck) intersection. Full row-height again — same as pre-half-cell layout. Drag-and-drop column reorder still works.
- This maps **exactly** to APC40 mk2's 8 visible strips: strip 1 → drums A, strip 2 → drums B, etc. The A/B button on each strip auto-lights from our `.als` template's crossfade assignments.
- **No SONG column in the main grid.** Replaced by a per-deck "song" indicator inside each deck waveform header (the muted Mix reference). One less column = more horizontal real estate for the 8 stems.

### Rec banners (unchanged plumbing, regrouped layout)

- Still **4 source-role rec feeds** (drums, bass, vocals, other) + 1 song feed.
- Each role banner spans **both** A and B columns above it (2-column-wide blocks). Visual grouping reinforces "Drums recs apply to either side."
- Load button still passes source-role; backend's `_pick_side` auto-routes (already implemented). Shift-click forces the other side.
- Song-rec banner spans full width at the bottom.

## Cueing — the real problem

You called this out and you're right: **with both A and B on master via the crossfader, where do previews go?**

### Research finding

**AbletonOSC does NOT expose Live's per-track Cue switch** (the small headphone icon in the session-view mixer that routes a track to the Cue output instead of Main). I checked `ideoforms/AbletonOSC/track.py` and the README — they wrap `mute`, `solo`, `arm`, `volume`, `panning`, etc., but `Track.solo_cue` (the LOM property for PFL) is unexposed.

So "PFL deck A to headphones" isn't a 1-line OSC call. Options:

### Option A — Defer per-deck PFL. Use clip state as audibility.

A real DJ rig's PFL exists because vinyl/CDJ decks ALWAYS output audio when playing. In Live, a clip in a slot doesn't make a sound until you fire it. So:

- Load incoming song into B-side stems → clips sit unfired, silent.
- Beatmatch via the analyzer's tempo + key (we already show these per cell). Visual matching beats audible matching for our use case.
- When ready, fire B-side scene → crossfade A→B on the APC40.

**This is what most stem-DJs actually do.** Pre-listening the audio is overrated when the analyzer tells you BPM/key/energy upfront. Defer PFL to a follow-up.

**Keep the existing Cue track** (outs 3/4) for **out-of-grid auditioning** only — the rec card ▶ preview button is still useful for "is this candidate worth loading at all?"

### Option B — Fork AbletonOSC (5 lines), wire per-deck PFL properly.

If we want the full DJ workflow, add a handler to AbletonOSC that exposes `track.solo_cue`. Then add a UI toggle per deck (PFL A / PFL B) that flips `solo_cue` on every track in that side simultaneously. Master Cue Mode set to "Cue" (not "Solo") in Live's master strip routes those to outs 3/4.

Cost: ~5 lines in our AbletonOSC fork + a small UI toggle. We already ship a patched AbletonOSC (per `docs/abletonosc_setup.md`) so adding one more handler isn't a big deal.

### Option C — Send-bus PFL (no OSC patch needed).

Add a "Cue Bus" return track routed to outs 3/4. Each A-side stem track has a default Send-X to Cue Bus of 0. PFL-A button → set Send-X to 100% for all A-side tracks; PFL-B → flip to B-side. Same effect, all driven by AbletonOSC primitives that exist today (`/live/track/set/send`).

Costs: 1 extra return track in the `.als` template, ~10 lines of bridge code, no fork required. Slight CPU overhead from active sends.

### Recommendation

- **Ship Option A first.** Get the layout right, prove the 2-deck UX is correct, learn what we actually miss.
- **If we miss PFL after a few sets, do Option C** (send-bus). No fork, contained complexity.
- **Don't fork AbletonOSC** unless C feels insufficient — forks are maintenance burden.

## What we keep from the deck-pair work

All three of yesterday's commits stay valid:

1. **Bridge `_DECK_KINDS` shape (9 tracks, mix-last).** Unchanged — backend doesn't care about UI layout.
2. **`.als` writer with crossfader assignment.** Unchanged — the per-side A=0/B=2 routing is exactly what the 2-deck UI needs.
3. **Persistence migration.** Unchanged — covers the upgrade path either way.

What we **throw out**: the `SplitCell` / `HalfCell` half-cell layout in SceneGrid. ~150 lines, contained replacement.

## What changes

| Surface | Today (Option C, half-cells) | Proposed (2-deck) |
|---|---|---|
| ComboStrip | One card per role (5), playing-row mini-waveforms | **Two deck waveform strips** (A on top, B below) with stacked 4-stem waveforms each |
| SceneGrid columns | 5 (4 stems + Song), each non-mix split A/B vertically | **8 columns interleaved by role** (no Song column in grid) |
| Mix reference | Single shared SONG cell per row | **Per-deck mix indicator in the deck strip header** (muted reference, click to unmute) |
| Rec banners | 5 columns (one per source role + Song) | **4 source-role banners spanning 2 cols each, + a full-width Song banner below** |
| BoothColumnHeaders | 5 headers (DRUMS/BASS/VOCALS/MELODY/SONG) | 8 headers (DRUMS A / DRUMS B / ...) — keeps the per-role color, A-side bright / B-side muted |
| Cueing | One Cue track for previews | Same — Cue track stays. **No per-deck PFL in v1** (Option A above). |
| APC40 mapping | Strip 1=Drums A, 2=Drums B, ... (already correct) | Same — the layout already matched the hardware, we're just making the *screen* match too |

## Visualizer detail (your "stacked" pick)

Each deck waveform shows 4 stacked stem waveforms (drums/bass/vocals/other) for that side's anchored track. Same colors as the column headers. A single shared playhead scrolls across.

When only one stem is loaded on a side (single-stem-load case), only that lane has bars; the other 3 are dashed/empty. When 4 stems share one source track (anchor), it's a full stacked waveform.

The currently-firing stem(s) glow brighter; muted/stopped stems are dim. This gives a per-deck "I see what's playing" at a glance.

## Open questions for you

1. **Per-deck Mix reference in the deck strip header** — show as a tiny ◇ chip with the original track title, click to unmute the Mix track? Or omit entirely until you need the "fall back to original" parachute?
2. **8 columns + Song banner full-width** vs. **9 columns with Song still as the rightmost column**? My pick: **8 + full-width Song banner**, but I can see arguments for keeping Song in the grid.
3. **Crossfader visualization on screen** — should the UI show the current crossfader position (a horizontal bar between Deck A and Deck B that animates as APC40's fader moves)? AbletonOSC exposes the master crossfader value. ~15 LOC.
4. **PFL strategy: A or C?** I lean A (defer), but if you've tried sets and miss pre-listening, C (send-bus) is the right escalation.

If you sign off on this — even rough yes/no per question — I'll do this in **two gated commits**:

1. **Layout swap**: SceneGrid 8-column interleaved, BoothColumnHeaders for 8 headers, rec banner regroup. (Pure UI, no backend.)
2. **Deck strips**: New TwoDeckStrip component replacing ComboStrip — stacked 4-stem waveforms per deck, shared playhead, deck-level metadata header. Wire to existing useAbletonState.

Then we live with it for a session and revisit PFL.

## Risks

- **Horizontal cramping at 1512px logical** (your 14" MBP): 8 stem columns + row label + rec rail = ~118px/column at default scaling, tight but workable. Track titles will truncate harder. Could be a real problem during a set if cells get unreadable.
- **The 2 deck waveform strips take vertical space** that today's ComboStrip didn't need. Means fewer SceneGrid rows visible without scrolling. Mitigation: collapse one or both deck strips when nothing's loaded that side.
- **Backend assumptions** (`_pick_side`, anchor detection per side) were designed for the deck-pair model and stay valid — no churn there.
