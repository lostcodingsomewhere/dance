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

Decision: "I want different/new [drums | bass | vocals | other | a full anchor song]." Then: scan that role column's recs in the plan grid, pick a candidate, load it onto a deck (⤒A / ⤒B), fire it. Live's clip-launch quantize handles the timing — the swap happens on the next phrase boundary (typically bar or beat-quantized).

This is the highest-frequency decision in the loop. In song-mode DJing the picker fires every 3–5 minutes; here, **swaps fire every 30–90 seconds**. Cognitive load per decision is lower (one stem, not a whole song) but the cadence is higher.

| Need | Source |
|---|---|
| Per-role rec stream | The role column in the plan grid (RoleColumnsGrid, `mode="live"`): recs hang below your queued plan picks, live-rescored against the playing combo + trailing-journey trend |
| Your planned next pick for the role | The plan zone on top of the same column — queued picks (emerald edge), if a set is active |
| Compat with current combo | Per-card ScoreBreakdown chip row: embedding / key / BPM / energy / timbre / transition-fit |
| Search escape hatch | ⌘K hybrid search (fuzzy + CLAP vibe); a chosen result appends to the Song column's plan queue |
| Cue before firing | Per-card ▶ preview button → cue bus (Scarlett 4i4 outs 3/4 → headphones, master unaffected) |
| Load it onto a deck | ⤒A / ⤒B on the card picks the deck at load time |
| Fire it | Tap cell in SceneGrid OR launch from APC40 |

**Eyes-on demand: medium-high.** Scanning a role column's recs, reading ScoreBreakdown chips. Faster cycle than song-mode picking.

### SET ARC — 10–60 min

What's happening: the user is shaping the trajectory across many swaps. Decisions here are at the level of "the past 5 minutes have been heavy, time to pull back".

| Need | Source |
|---|---|
| Energy curve over time | Compact sparkline inline in MasterStrip (always visible) |
| Recent stem swaps | Recent-plays history in the MasterStrip's session chip |
| What did I plan to bring in next? | The plan zone (queued picks) on top of each role column — the set's plan, mirrored live |
| Have I leaned hard on one source? | "Used N× already" badge on tracks heavily mined (v2+) |
| Set shape vs intent | Optional set-arc template overlay (v2+) |

**Eyes-on demand: occasional.** Glanced at every 5–10 minutes during lulls — but because both signals live in the MasterStrip at the top, no surface switch is needed.

## Surfaces in the live-remixing Booth

Top-to-bottom layout (current; mirrors what's in `views/Booth.tsx`):

| Surface | Where | What it answers |
|---|---|---|
| MasterStrip | Top bar (rendered by the app shell, above the Booth view) | BPM (click → genre-anchored slider with explicit Apply) · KEY · combo VU meter · energy arc · Live-bridge heartbeat · resync · vibe search · view tabs · session chip (play count + end-set) |
| TwoDeckStrip | Just below MasterStrip — "what's playing right now?" | Two panels, one per deck (A / B). Each panel stacks its four playing stem waveforms (drums / bass / vocals / other) for that deck, with live playhead from Live's per-clip `playing_position`, faint colored section bands behind the peaks, unicode section icons (▲ buildup, ▼ drop, ◌ breakdown, ◇ bridge, ▷ intro, ◁ outro, ● verse, ★ chorus) at section starts with hover-tooltips, and click-to-scrub. The per-deck Mix/anchor reference shows as a header chip on its panel. (Replaced the earlier single-deck ComboStrip / MasterVisualizer surfaces.) |
| Crossfader | Between the two deck panels and the grid | Mirrors the APC40 hardware A/B crossfader; drag on-screen to set. A on the left, B on the right — blends Deck A's stems against Deck B's. |
| BoothColumnHeaders | Above the SceneGrid | 8 fader-order chips (DRUMS A · DRUMS B · BASS A · … · OTHER B), one per APC40 strip, A/B sharing a role hue. Per-chip Solo "S" button cues that deck through Live's Solo/PFL bus → headphones. Mix/anchor is not a column here — it's a per-deck chip in the TwoDeckStrip header. |
| SceneGrid (8 cols × 4 rows, expandable to 8 rows) | The centerpiece | Canonical APC40 mirror in **fader order**: drums_a, drums_b, bass_a, bass_b, vocals_a, vocals_b, other_a, other_b. Tap a cell to fire one stem; tap a row label to fire/stop the whole scene (anchor mode). Tap a playing cell or row to stop it. Hover any loaded cell → small × in the top-right removes that one stem from the grid. |
| CueStrip | Conditional, appears when previewing | The parallel-to-master cue surface. Shows what's auditioning in headphones (Scarlett 4i4 outs 3/4), with the source track's full waveform (same section bands + cue icons as TwoDeckStrip), ⏹ stop, and a "→ Load … to master" one-click commit. |
| RoleColumnsGrid — **the plan grid** (5 role columns) | The spine, below the cue strip | The one surface the app is built around. Five role columns — DRUMS · BASS · VOCALS · OTHER · SONG — each stacks your **queued plan picks on top** (the plan zone, emerald-edged, with × to remove) and **recommendations below**. In the Booth it renders `mode="live"`: recs tail what's playing (combo embedding + trailing-journey trend) and each card carries ⤒A / ⤒B to load that pick onto a deck. Every card has a ▶ cue preview and a ScoreBreakdown chip row. ＋ on a rec queues it into the plan; with no active set the plan zone is hidden and it's just the live recs. |

The plan grid is a flat 5-column CSS grid (`grid-cols-5`) — one column per role, in the order drums · bass · vocals · other · song. It does **not** line up with the 8-column SceneGrid above it (that mirrors the APC40's 4 roles × 2 decks); the plan grid is role-keyed, deck-agnostic, and the deck is chosen per-load via ⤒A / ⤒B.

There are no sidebars. Glanceable signals (energy arc, session/play count) live in the MasterStrip so the SceneGrid + plan grid own the center. The SceneGrid defaults to 4 rows visible with an `▾ show all 8 rows` toggle — auto-expands if cells are loaded in rows 5–8. (There is no PlayedStrip footer; play count + end-set moved into the MasterStrip's session chip.)

## What information must be co-located

Some things must be visible in one glance, no tab-switching:

- **Per-role rec stream + that role's plan queue, in one column.** Each role column stacks your queued plan picks over its live recs, so picking a vocal means seeing both your planned move and fresh candidates in the same place.
- **Current playing combo + all rec streams.** Recs are scored against the combo, so the combo has to be readable while shopping.
- **Master BPM/key + per-card ScoreBreakdown chips.** Compatibility is a relationship; both sides visible.

Things that can be separated by a tab/keystroke:

- Pre-set planning — the **Set** view renders the *same* plan grid (`mode="plan"`): same five role columns, but recs are scored against the rest of the plan + the plan's journey, and there's no deck-load (you're queuing, not firing). Library browse is folded into the ⌘K palette.
- Pipeline / Ops / Settings (administrative). Pipeline doubles as the library inventory surface.
- Session history beyond the current set.

## Attention pattern

| Phase | Primary | Secondary | Background |
|---|---|---|---|
| NOW (combo grooving) | Ears + APC40 hands | Glance at SceneGrid for playing cells | — |
| SHOPPING (deciding swap) | The focused role column in the plan grid (recs + your queued picks) | Current combo summary | Set arc / energy sparkline |
| SWAP MOMENT (loading + firing) | ⤒A/⤒B on the rec card, then SceneGrid tap or APC40 | Beat indicator | — |
| LULL (combo locked in) | Set arc / the plan zone (queued picks per role) | Library / vibe search via ⌘K for next-but-one | — |

A well-tuned single-surface UI should support this pattern without forcing the user to remember where to look.

## Hardware constraints

Non-negotiable for this user (see [`../HARDWARE.md`](../HARDWARE.md)):

- **APC40 mk2 — 8×5 clip grid.** Hardware-fixed. Each row = scene; the 8 columns are stem-role × deck tracks (drums/bass/vocals/other × A/B, in fader order). The companion app's grid mirrors this orientation, and the 8 channel faders map 1:1 (faders 1–4 = Deck A, faders 5–8 = Deck B). The hardware A/B crossfader blends Deck A against Deck B.
- **Ableton Live Standard 12.4.** Audio engine, mixer, clock, clip-launch quantize, master out. Not negotiable.
- **MacBook Pro M2 Pro 16 GB.** The screen the companion app runs on, primary surface during dev.
- **No Bluetooth audio path.** Wired-only.

## Glossary

- **Combo** — the current set of playing stem cells. The thing the listener hears.
- **Swap** — firing a new cell in a column where one cell is already playing, replacing it.
- **Layer** — firing a cell in a column that's currently silent.
- **Anchor** — firing a whole row to play the original song combo. Recovery / fallback move.
- **Stem role** — drums / bass / vocals / other (+ **song**, the full-track anchor; the recommender calls it `mix`). These five roles are the columns of the **plan grid**. In the 8-column SceneGrid each of the 4 stem roles also occupies two columns — one per deck (A / B) — with mix/anchor as a per-deck header chip.
- **Cell** — the intersection of a row (track) and column (stem role) in the SceneGrid. One stem of one track.
- **Plan grid** — the RoleColumnsGrid: five role columns, each stacking queued plan picks over recs. Renders `mode="live"` in the Booth (recs tail what's playing, ⤒A/⤒B deck-load) and `mode="plan"` in the Set view (recs scored against the rest of the plan, no deck-load).
- **Plan / plan queue** — a Set's intent, stored as per-role queues `{role: [track_id, …]}` on `sets.plan`. The plan zone (top of each role column) shows your queued picks; ＋ on a rec or ⌘K (→ Song column) appends to it. **A set IS its plan.**
- **Rec stream** — the per-role list of candidate stems below the plan zone, scored through the unified brain (combo/journey vibe + halftime BPM + key/timbre/transition-fit) and refreshed on combo change (live) or plan change (plan mode).
- **ScoreBreakdown chip** — at-a-glance per-feature match row on a rec card (embedding / key / BPM / energy / timbre / transition-fit) — relative to the active combo (live) or the rest of the plan (plan mode), not to a single track.
