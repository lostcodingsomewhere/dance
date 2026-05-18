# DJ UX flow

What the user does during a set and what information they need at each moment. Reference doc for UX decisions.

**Read [`vision.md`](vision.md) first.** This doc assumes the live-remixing style declared there. Scope: solo, APC40 mk2 + Ableton Live 12.4, 30–90 min living-room sets.

## The loop, in stem terms

Performance is a continuous loop with three nested clocks. Same shape as any DJ flow, but **the unit at each clock is a stem, not a song.**

```
                  ┌─────────────────────────────────────────┐
                  │       SET ARC  (10–60 min horizon)      │
                  │   energy curve, vibe drift, fatigue     │
                  │                                         │
                  │   ┌─────────────────────────────────┐   │
                  │   │   NEXT SWAP  (5–60 s)           │   │
                  │   │   pick column, pick cell, fire  │   │
                  │   │                                 │   │
                  │   │   ┌─────────────────────────┐   │   │
                  │   │   │  NOW  (0–8 bars)        │   │   │
                  │   │   │  the current combo,     │   │   │
                  │   │   │  ride the groove        │   │   │
                  │   │   └─────────────────────────┘   │   │
                  │   └─────────────────────────────────┘   │
                  └─────────────────────────────────────────┘
```

Each clock has different attentional demands and different "what do I need to see right now" answers.

### NOW — 0–8 bars

What's happening: a combination of stems is playing (anywhere from 1 stem to all 5). The user is **listening** and reacting — riding stem faders on the APC40, pushing the kick, pulling vocals out for a breakdown, sweeping a filter. They are not reading.

| Need | Source |
|---|---|
| What stems are currently active? | Scene grid: which cells are lit |
| Where am I in the loop? | Beat / bar indicator on the playing cells |
| Will anything end soon? | "Loop ends in N bars" callout on non-looped clips |
| What's the master BPM / key of the combo? | Master strip |

**Eyes-on demand: low.** Ambient, peripheral-vision glanceable. Hands on faders, ears on speakers.

### NEXT SWAP — 5–60 s

What's happening: the user decides to swap one stem (or layer in a new one). They're **shopping a single column.**

Decision: "I want different/new [drums | bass | vocals | other | a full anchor song]." Then: scan that column's recs, pick a candidate, fire it. Live's clip-launch quantize handles the timing — the swap happens on the next phrase boundary (typically bar or beat-quantized).

This is the highest-frequency decision in the loop. In song-mode DJing the picker fires every 3–5 minutes; here, **swaps fire every 30–90 seconds**. Cognitive load per decision is lower (one stem, not a whole song) but the cadence is higher.

| Need | Source |
|---|---|
| Per-column rec stream | Live-rescoring banner above each column |
| Compat with current combo | Per-cell chips: key / BPM / energy fit vs combo |
| Filter to a mood | Per-column vibe chips ("denser", "darker", "more vocal") |
| Search escape hatch | ⌘K vibe search, auto-scoped to focused column |
| Cue before firing | Per-rec ▶ preview button → cue bus (Scarlett 4i4 outs 3/4 → headphones, master unaffected) |
| Fire it | Tap cell in grid OR launch from APC40 |

**Eyes-on demand: medium-high.** Scanning rec banners, reading compat chips. Faster cycle than song-mode picking.

### SET ARC — 10–60 min

What's happening: the user is shaping the trajectory across many swaps. Decisions here are at the level of "the past 5 minutes have been heavy, time to pull back".

| Need | Source |
|---|---|
| Energy curve over time | Compact sparkline inline in MasterStrip (always visible) |
| Recent stem swaps | Horizontally-scrolling PlayedStrip at the bottom |
| Have I leaned hard on one source? | "Used N× already" badge on tracks heavily mined (v2+) |
| Set shape vs intent | Optional set-arc template overlay (v2+) |

**Eyes-on demand: occasional.** Glanced at every 5–10 minutes during lulls — but because both signals live in slim strips at top/bottom, no surface switch is needed.

## Surfaces in the live-remixing Booth

| Surface | Where | What it answers |
|---|---|---|
| MasterStrip | Top bar | BPM · KEY anchor · energy arc · OSC heartbeat · view tabs · vibe search |
| ComboStrip | Below MasterStrip | "What's playing right now?" — one card per stem role, surfaces the source track of each playing stem; flags anchor mode when present |
| Per-column rec banners | Above the grid, 5 across | "What should I swap into each column next?" — live-rescored against the active combo |
| SceneGrid (8×5) | Center, full width | Canonical APC40 mirror — tap a cell to fire one stem, tap a row to anchor a whole song |
| PlayedStrip | Bottom footer | Set name, play count, recent plays history, end-set |

There are no sidebars. Glanceable signals (energy arc, played history) live in slim strips so the grid + banners own the center.

## What information must be co-located

Some things must be visible in one glance, no tab-switching:

- **Per-column rec banner + the column itself in the grid.** Picking a vocal means seeing both candidates and the cell it will land in.
- **Current playing combo + all rec streams.** Recs are scored against the combo, so the combo has to be readable while shopping.
- **Master BPM/key + per-cell compat chips.** Compatibility is a relationship; both sides visible.

Things that can be separated by a tab/keystroke:

- Pre-set Crate work (library browse, stage tracks).
- Pipeline / Ops / Settings (administrative).
- Session history beyond the current set.

## Attention pattern

| Phase | Primary | Secondary | Background |
|---|---|---|---|
| NOW (combo grooving) | Ears + APC40 hands | Glance at grid for playing cells | — |
| SHOPPING (deciding swap) | Focused column's rec banner + the column in grid | Current combo summary | Set arc / energy sparkline |
| SWAP MOMENT (firing the cell) | Grid (tap target) or APC40 | Beat indicator | — |
| LULL (combo locked in) | Set arc / SetRail / "what's working" | Library browse for next-but-one | — |

A well-tuned single-surface UI should support this pattern without forcing the user to remember where to look.

## Hardware constraints

Non-negotiable for this user (see [`../HARDWARE.md`](../HARDWARE.md)):

- **APC40 mk2 — 8×5 clip grid.** Hardware-fixed. Each row = scene, each column = stem-role track. The companion app's grid mirrors this orientation.
- **Ableton Live Standard 12.4.** Audio engine, mixer, clock, clip-launch quantize, master out. Not negotiable.
- **MacBook Pro M2 Pro 16 GB.** The screen the companion app runs on, primary surface during dev.
- **No Bluetooth audio path.** Wired-only.

## Glossary

- **Combo** — the current set of playing stem cells. The thing the listener hears.
- **Swap** — firing a new cell in a column where one cell is already playing, replacing it.
- **Layer** — firing a cell in a column that's currently silent.
- **Anchor** — firing a whole row to play the original song combo. Recovery / fallback move.
- **Stem role** — drums / bass / vocals / other / mix. Maps to one of the 5 columns of the APC40 grid.
- **Cell** — the intersection of a row (track) and column (stem role). One stem of one track.
- **Rec stream** — the live-rescored list of candidate cells for one column, refreshed on combo change.
- **Compat chip** — at-a-glance match indicator on a candidate cell — relative to the active combo, not relative to a single track.
