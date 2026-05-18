# Proposal — Frontend as the primary DJ surface

**Status:** implemented (rev 3 — layout reshaped to drop song-mode artifacts)
**Date:** 2026-05-17 (initial); 2026-05-18 (rev 3)
**Builds on:** [`../vision.md`](../vision.md), [`../dj_ux_flow.md`](../dj_ux_flow.md)

## TL;DR

The companion app is the only screen the user looks at during a set. The screen is full-width: a top **MasterStrip** (BPM, KEY, energy arc, OSC heartbeat, view nav), a horizontal **ComboStrip** under it (one card per stem role, showing what's playing per role with its source-track metadata), then **five per-column rec banners** above the **8×5 SceneGrid** (the APC40 mirror), and a thin **PlayedStrip** footer for set history. Ableton is the invisible audio engine; the APC40 is the hands.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ MasterStrip   BPM 124  KEY 5A   Arc▁▂▄▆█  0 decks  •OSC   ⌘K   Booth Crate    │
├──────────────────────────────────────────────────────────────────────────────┤
│ CURRENT COMBO                                                  live remix     │
│ ┌─ drums ──┐┌─ bass ───┐┌─ vocals ─┐┌─ other ──┐┌─ mix ─────┐                  │
│ │ Track A  ││ Track B  ││ silent   ││ Track C  ││ silent    │   (per-role)    │
│ │ 124 · 8A ││ 124 · 8B ││          ││ 122 · 7A ││           │                  │
│ └──────────┘└──────────┘└──────────┘└──────────┘└───────────┘                  │
│                                                                                │
│ NEXT PER COLUMN · LIVE RE-SCORED AGAINST THE COMBO                            │
│ ┌── drums ──┐┌── bass ──┐┌── vocals ─┐┌── other ─┐┌── mix ──┐                  │
│ │ rec rec   ││ rec rec  ││ rec rec   ││ rec rec  ││ rec rec │  ← banners      │
│ │ rec rec   ││ rec rec  ││ rec rec   ││ rec rec  ││ rec rec │                  │
│ └───────────┘└──────────┘└───────────┘└──────────┘└─────────┘                  │
│                                                                                │
│ SCENE GRID · TAP TO FIRE (MIRRORS APC40)                                       │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │  row 1  ▶  (drums)                                                       │ │
│ │  row 2          ▶            ▶                                           │ │
│ │  ...                                                                     │ │
│ │  row 8                                                                   │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────────┤
│ PlayedStrip   Set · 12 plays · 02:27 AM   [#1 Hyph][#2 Mort][#3 …]   end set  │
└──────────────────────────────────────────────────────────────────────────────┘
```

No sidebars. The COMBO STRIP makes "what's playing" honest (per-role, not per-track), and everything glanceable (energy arc, history) goes into thin strips at the edges so the grid + banners own the center.

## Problem

See [`../dj_ux_flow.md`](../dj_ux_flow.md) §"What's currently fighting the flow". Headline: the current UI was designed for song-mode and breaks down for the live-remixing style declared in [`../vision.md`](../vision.md). Recs are a single ranked list; cells are not first-class; stems don't loop; compat math is per-track; and Ableton's UI competes with the companion app for attention.

## Proposed change

Five concrete shifts, in roughly this order of dependency.

### Shift 1 — Loop-by-default on emitted clips

`src/dance/als/writer.py` change: every audio clip written into the template gets `Loop = true` in the XML. The clip plays its source region in loop until swapped or stopped.

Per-clip override exposed in the UI: small ⤾ toggle on each cell. Off = play through and stop. On (default) = keep grooving.

Why first: nothing else in the redesign matters if stems don't loop. The whole live-remixing model depends on this.

### Shift 2 — Transport from the FE

New endpoints, all routing through `src/dance/osc/client.py`:

- `POST /ableton/transport/fire-scene/{idx}` — fires a whole row (anchor mode).
- `POST /ableton/transport/fire-clip/{track}/{slot}` — fires one cell.
- `POST /ableton/transport/stop-track/{idx}` — stops one column (clears one stem from the combo).
- `POST /ableton/transport/stop-all` — panic / clear combo.

The FE can launch clips and scenes without the user touching Ableton or the APC40.

### Shift 3 — Scene grid as visual centerpiece

The 8×5 grid becomes the **canonical visual representation of what the APC40 is touching**. Same orientation as the APC40 (bottom row = scene 1, matching Live's default APC40 mapping). Each cell shows:

- Track title (truncated)
- Per-cell compat chip relative to the active combo
- Visual state: empty / loaded / playing (with a beat-pulse on the playing cells)
- Loop-override indicator if the cell has been flipped off-default

Interactions:
- **Tap a cell** → fire via OSC (`/live/clip/fire`).
- **Tap a row label** → fire the whole row (anchor mode).
- **Right-click / long-press a cell** → small per-cell control panel (loop toggle, peek metadata, "more like this").
- **Hover a cell** → preview metadata in a small tooltip without firing.

### Shift 4 — Drop song-mode UI; full-width banners + grid

**Removed (no longer rendered):**
- `SetRail` (left sidebar) — its content is now distributed: energy arc → MasterStrip, played history → PlayedStrip, scene list → absorbed into SceneGrid.
- `NowCard` — replaced by the ComboStrip, which shows the active stems per role rather than a single track.
- `SceneMap` — superseded by the canonical SceneGrid (8×5).
- `UpNextRail` — replaced by per-column rec banners.

**Added:**
- `ComboStrip` (`components/ComboStrip.tsx`) — horizontal 5-card row directly under MasterStrip. Each card surfaces the source-track metadata (title, key, BPM, artist) of whatever is *currently playing* in that role. When all five cells point at the same scene, it surfaces an "⚓ anchored to scene N · [title]" hint so anchor-mode is legible.
- `EnergySparkline` (`components/EnergySparkline.tsx`) — compact arc inline in MasterStrip, fed by `useCurrentSession`. Lives next to BPM/KEY/heartbeat as a glance-anywhere set-arc cue.
- `PlayedStrip` (`components/PlayedStrip.tsx`) — thin footer with set name, play count, horizontally scrollable history of the last ~30 plays, and the end-set button.

**Per-column rec banners** (existing): each of the 5 columns gets a banner above the grid showing **3–5 candidate cells, continuously re-scored against the currently-playing combo.**

- Drums banner answers "given current bass + vocals + other, what drums fit?"
- Bass banner answers "given current drums + vocals + other, what bass fits?"
- Vocals, Other, Mix — same pattern.
- Mix column's banner is the song-anchor escape hatch ("what whole song would I drop into right now?")

Per-column vibe filter chips ("denser", "darker", "more vocal", "instrumental") live inline with the banner header.

⌘K vibe search auto-scopes to the column with focus (last-clicked or last-hovered).

Tap a banner candidate → it loads into the next empty row of that column → fire it from the grid or APC40 when ready.

**Backend addition:** `GET /recs/by-column?combo=<active_stem_ids>&column=<drums|bass|vocals|other|mix>` returning a ranked list of stem candidates. Scoring is per-stem embedding similarity + key/BPM compat against the active combo's aggregate features. Recompute on combo-change events pushed via the existing WebSocket, not on a polling timer.

### Shift 5 — Hide Ableton

Workflow shift, not a code shift (but enabled by 1–4):

1. User opens Live, starts AbletonOSC, **minimizes Live**.
2. Companion app fullscreen.
3. APC40 on the desk.
4. Live UI only visited for setup / troubleshooting.

Small support code:
- Heartbeat indicator in MasterStrip — pings AbletonOSC every 2 s; goes red on stale (>3 s no response). User has confidence the engine is alive without looking at Live.
- "Ableton is alive" badge becomes the single trust signal for the engine layer.

## Alternatives considered

### Alt A — Status quo (song-mode UI)

Pro: zero code change. Con: doesn't support the live-remixing vision. Hard veto.

### Alt B — Focused single rec rail (one column at a time)

Single right-hand rec rail with a "focus" dropdown that filters by column. Deeper per-column lists but only one column visible at a time.

Pro: less screen real estate. Con: forces the user to commit to "I'm shopping for X" before seeing what's available — friction the choose-your-adventure vision rejects. The mental model is multi-column simultaneous; the UI should match.

### Alt C — Proposed (Option A: horizontal banners above each column)

Pro: matches the mental model. Eye scans horizontally across columns, sees all paths forward. Con: real estate cost — banners + grid + SetRail won't all fit comfortably on a 14" MBP without compromises.

**Recommendation: Alt C** with SetRail demoted to a slide-up overlay (footer present but collapsed; slides up on click or during lulls).

### Alt D — Replace Ableton entirely

Build our own audio engine. Throws away Live's mixer, clock, quantize, APC40 native binding. Not seriously considered.

## Trade-offs

| Concern | Mitigation |
|---|---|
| **OSC latency for transport** (~20–50 ms RTT). | Live's clip-launch is bar-quantized — the launch happens on the next downbeat regardless of when OSC arrives within the bar. Latency is hidden by quantization. |
| **Per-column rec scoring is 5× more work** than single-list (5 streams, each re-scored on every combo change). | Cheap. Embedding similarity is a dot product over ~1000 candidate stems. Sub-10 ms even single-threaded. Recompute on combo-change events (WebSocket-pushed), not on a timer. |
| **Screen real estate on 14" MBP** (1512×945 logical). | SetRail collapsible. Banners short (one row of cards per column). Master strip slim. Comfortable but tight — needs an ergonomic check. |
| **APC40 stays source of truth for stem faders.** Driving from FE would race. | FE shows fader readouts only (read-only). No conflict. v2 work if useful. |
| **Ableton crash / OSC unresponsive becomes invisible** if user isn't looking at Live. | Heartbeat indicator (Shift 5). Red dot on stale; user knows to alt-tab. |
| **Loop=true on every clip** changes `.als` shape — needs re-verification in Live (CLAUDE.md rule: don't change writer without verifying in Live). | Phase 1 includes a real-Live verification pass on a fresh `.als`. Open Live, fire a scene, confirm clips loop, confirm overrides work. |
| **Per-stem compat math is more granular** than per-track. Need to trust per-stem analysis. | Already have it. CLAP per-stem embeddings + per-stem `audio_analysis` rows exist (see `src/dance/core/database.py`). The scoring just needs to operate on those instead of per-track aggregates. |

## Open questions

1. **What does the compat chip on a banner candidate show?**
   - i: per-column-specific (a drum candidate's chip shows compat with current bass + vocals + other)
   - ii: simple/uniform (key + BPM vs master, same chip everywhere)
   - **Recommendation:** start with ii (simple) for v1; revisit if it feels too coarse. The combo-aware scoring still drives ranking; the chip is just a quick read.

2. **Per-clip loop override — where does the UI live?** Inline on every cell (icon clutter on 8×5) or hidden behind right-click / long-press?
   - **Recommendation:** right-click / long-press reveals a per-cell control panel. Don't clutter the default view.

3. **Does the Mix column get the same banner treatment as the stem columns?**
   - **Recommendation:** yes, uniform. The Mix column's banner answers "what whole song would I anchor to right now?" — useful escape hatch and stays consistent with the per-column model.

## Migration plan

Phased, each phase shippable on its own.

### Phase 1 — Loop-by-default + transport endpoints (~1 day)

- `src/dance/als/writer.py`: emit `Loop = true` on all audio clips.
- Real-Live verification: generate fresh `.als`, load in Live, fire scenes, confirm loops.
- `POST /ableton/transport/fire-scene/{idx}`, `/fire-clip/{t}/{s}`, `/stop-track/{t}`, `/stop-all`.
- Wire to OSC client; expose buttons in existing SceneMap as a placeholder.

**Validation:** fire a scene from the FE; verify the loop continues until manually stopped. Latency feels < 100 ms perceived.

### Phase 2 — Scene grid as canvas (~2–3 days)

- New `components/SceneGrid.tsx` — 8×5, fed by `useDeckMap` + `useAbletonState`.
- Tap-cell-to-fire, tap-row-to-anchor, right-click for per-cell controls.
- Per-cell visual states (empty / loaded / playing with beat pulse).
- Replace SceneMap in Booth.tsx layout.

**Validation:** play a track end-to-end from the FE without touching Live or the APC40.

### Phase 3 — Per-column rec banners + per-stem compat scoring (~3–4 days)

- Backend: `GET /recs/by-column` returning per-column ranked candidates given the active combo.
- Scoring: per-stem embedding similarity + key/BPM compat against active combo aggregate.
- WebSocket push of combo-change events (already pushed via `AbletonState.playing_clips`; FE derives combo).
- FE: `components/ColumnRecBanner.tsx` × 5, placed above the grid.
- Per-column vibe chips inline with each banner header.
- ⌘K vibe search auto-scopes to focused column.

**Validation:** start with one stem playing → other columns' recs should make sense. Swap a stem → other columns re-score visibly. Mock combinations should match the user's intuition for what fits.

### Phase 4 — Layout reorg + heartbeat + polish (~1 day)

- Restructure `views/Booth.tsx`: banners + grid as centerpiece, NOW/UpNextRail absorbed or removed, SetRail demoted to slide-up footer.
- AbletonOSC heartbeat → red dot in MasterStrip on stale.
- Master strip shows BPM + Camelot key of the dominant playing stem (or the "anchor" if a row is firing).

**Validation:** ergonomic check on 14" MBP at 1512×945. Nothing important is below the fold during a set.

### Phase 5 — Real-set verification

Per CLAUDE.md workflow rule 2: play a **30-minute set end-to-end with Live minimized**. Report what broke, what was awkward, what the next iteration needs.

## Out of scope (explicitly)

- Replacing Ableton's audio engine.
- Stem faders driven from the FE (read-only readouts only).
- Set-arc personalization / "this is landing" feedback loop.
- Cue-in-headphones from FE.
- Multi-DJ / multi-session collab.
- iPad-first responsive layout.
- Pipeline or DB schema changes beyond the per-column rec endpoint.

## Future unlocks

If the redesign lands, these become natural follow-ons:

- **Combo memory** — "I had this exact combo 12 minutes ago, want to revisit?"
- **Surprise me** — fire a random plausible swap; system picks for you.
- **Energy-target mode** — set an intended arc; recs bias toward hitting target energy at target time.
- **Per-stem effects** — per-column FX returns (filter, delay, reverb) driven from the FE.
- **Crowd feedback capture** — thumbs up/down on the live combo feeds the rec scorer.
- **Stem fader writeback** — drive Ableton mixer volumes from FE faders for trackpad mixing scenarios.

None of these are blocking; all are downstream of this proposal landing.

---

**Decision needed from user:**

1. Go / no-go on the redesign (Alt C — Option A layout)?
2. Answers to the 3 open questions?
3. Approval to start Phase 1 (loop-by-default + transport endpoints)?
