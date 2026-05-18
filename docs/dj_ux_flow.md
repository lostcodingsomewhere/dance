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
| Cue before firing | Headphone solo (v2 — APC40 manual cue only for v1) |
| Fire it | Tap cell in grid OR launch from APC40 |

**Eyes-on demand: medium-high.** Scanning rec banners, reading compat chips. Faster cycle than song-mode picking.

### SET ARC — 10–60 min

What's happening: the user is shaping the trajectory across many swaps. Decisions here are at the level of "the past 5 minutes have been heavy, time to pull back".

| Need | Source |
|---|---|
| Energy curve over time | Sparkline in SetRail footer |
| Recent stem swaps | History list (per-column ideal, single-stream acceptable) |
| Have I leaned hard on one source? | "Used 6× already" badge on tracks heavily mined |
| Set shape vs intent | Optional set-arc template overlay (v2+) |

**Eyes-on demand: occasional.** Glanced at every 5–10 minutes during lulls.

## What's currently fighting the flow

The current UI was designed assuming song-mode. The live-remixing pivot exposes these gaps:

1. **Recs are a single ranked list.** UpNextRail shows "next track" candidates. Useless for "I just want different vocals" — the user has to mentally filter by column themselves.
2. **Cells are not first-class.** SceneMap shows rows. Firing one cell requires clicking inside a row, which is awkward and conflates the "anchor a whole song" action with the "swap one stem" action.
3. **No per-column context awareness.** The system can't tell which column the user is shopping. Recs aren't filtered or re-scored on intent.
4. **Stems don't loop by default.** Clips emitted by `als/writer.py` play through and stop. For live-remixing, every clip needs `Loop = true` and the user needs a per-clip override for the rare cases they want a stem to play through (vocal verse-chorus-verse).
5. **Compat chips are per-track.** `K0 B0` currently means "this whole track matches the NOW track in key + BPM". For per-stem swapping the right question is "this candidate stem vs the current ACTIVE COMBO" — different math, different chip semantics.
6. **Two surfaces compete.** Eye flicks between Ableton's session view and the companion app during transition prep. Addressed by [`proposals/frontend-as-primary-surface.md`](proposals/frontend-as-primary-surface.md).

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
