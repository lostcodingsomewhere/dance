# Live-contract additions: crossfade-assign + `soloed_kinds` + `deck_map_revision`

Status: **approved, implemented** (2026-06-13).

Three additions to the OSC / WebSocket contract between the backend and Live,
plus the companion app. All three close gaps where the live-loaded deck-rig
diverged from the static `.als` export or forced the FE to poll.

## 1. New OSC verb — `/live/track/set/crossfade_assign`

### Problem

The crossfader did nothing to app-loaded stems. The static `.als` writer
(`dance/als/writer.py`) sets each track's `CrossFadeState/Manual` so A-side
decks follow crossfader group A, B-side decks follow B, and the mix references
stay "None" (always audible). But when the app live-loads stems over OSC it
only creates + names + colors + mutes the deck tracks — it never set their
crossfade assignment, because **stock AbletonOSC doesn't expose
`mixer_device.crossfade_assign`**. So an exported Set blended under the
crossfader and a live-loaded deck didn't.

### Change

Patch our local AbletonOSC fork (`abletonosc/track.py`) to add a settable +
gettable handler for `mixer_device.crossfade_assign`:

- `/live/track/set/crossfade_assign (track_index, value)`
- `/live/track/get/crossfade_assign (track_index)` → reply `(track_index, value)`

`crossfade_assign` is a plain int attribute on `mixer_device` (not a
`DeviceParameter` with a `.value` like `volume`/`panning`), so it gets its own
get/set callbacks rather than the generic mixer-property path. Full patch +
quit/reopen instructions: [`docs/abletonosc_setup.md`](../abletonosc_setup.md).

### LOM enum (verified against Live 12.4 MixerDevice)

| value | meaning | writer `CrossFadeState/Manual` |
|-------|---------|--------------------------------|
| `0`   | A (follows crossfader A side) | `_CROSSFADE_A = "0"` |
| `1`   | None (always audible — Live's default) | `_CROSSFADE_NONE = "1"` |
| `2`   | B (follows crossfader B side) | `_CROSSFADE_B = "2"` |

These are identical to the writer's values, so live-load routing == export
routing.

### Backend wiring

- `AbletonOSCClient.set_track_crossfade_assign(track, value)` /
  `get_track_crossfade_assign(track)` — with `CROSSFADE_A/NONE/B` int
  constants on the client.
- `AbletonBridge._create_deck_columns` now calls `set_track_crossfade_assign`
  for each of the 10 deck tracks, deriving the group from the deck-kind via
  `AbletonBridge._crossfade_assign_for` (A→0, B→2, else→1). That helper is the
  single semantic source of truth alongside the writer's
  `_crossfade_value_for`; a regression test asserts they agree for every
  `_DECK_KINDS` entry.
- Best-effort: on a backend running against **stock** AbletonOSC the address is
  unhandled, the call silently no-ops, and the deck stays at Live's default
  (None) — same as today. No crash, no required upgrade ordering.

## 2. New `AbletonState` field — `soloed_kinds: string[]`

### Problem

The FE rendered PFL/Solo lights by remembering the last `/pfl/{side}` it sent —
it never reflected Live's *actual* solo state (e.g. the user toggling an `S`
button on the APC40 / in Live directly, or a per-track solo from
`/transport/solo-track/{idx}`).

### Change

The bridge subscribes to per-track Solo on each deck column
(`/live/track/start_listen/solo`, added in `_subscribe_deck_columns`). Solo
needs **no fork patch** — `solo` is already in upstream AbletonOSC's track
`properties_rw`. On every `/live/track/get/solo (track, 0|1)` push the bridge
records the raw per-track state and recomputes `soloed_kinds`: the list of
deck-kinds (`"drums_a"`, `"mix_b"`, …) whose Live track currently has Solo on,
emitted in canonical `_DECK_KINDS` order. Track indices that aren't deck
columns (the Cue track, stray pushes before recovery) are ignored.

### Shape

```jsonc
"soloed_kinds": ["drums_a", "mix_b"]   // [] when nothing soloed
```

Canonical deck-kinds (10): `drums_a, drums_b, bass_a, bass_b, vocals_a,
vocals_b, other_a, other_b, mix_a, mix_b`. With Solo/Cue mode = Cue (set on
bridge init) a soloed track routes to the headphone PFL bus (outs 3/4), so
`soloed_kinds` == "decks currently in the cue."

## 3. New `AbletonState` field — `deck_map_revision: number`

### Problem

The companion polled `GET /ableton/decks` on a ~2 s timer to notice when the
deck-cell map changed (a load, a clear, a resync). Laggy and wasteful.

### Change

A monotonic integer on the bridge, bumped via `_bump_deck_revision()` on
**every** deck-cell map mutation — the same code paths that already persist
`deck_state.json`:

- `push_track_to_live` (clip load / anchor-fill mix cell)
- `delete_cell` (clear)
- `adopt_cells` (resync)
- `reset_deck_columns`, `clean_live_decks` (clear-all)
- `recover_deck_columns` (column layout adopted at boot — `columns` is part of
  the `/decks` payload)

The FE watches `deck_map_revision` on the WebSocket and refetches `/decks` the
instant it changes — no timer.

### Shape

```jsonc
"deck_map_revision": 7   // starts at 0, only ever increases
```

## Serialized contract (full `AbletonState` / `AbletonStateOut`)

Both new fields (plus `crossfader`, which was already broadcast by the bridge
but missing from the response model) are in `AbletonState.to_dict()` and the
`AbletonStateOut` Pydantic model, so they flow over both `GET /ableton/state`
and the `/ws` push:

```jsonc
{
  "tempo": 128.0,
  "is_playing": true,
  "beat": 4.5,
  "playing_clips": { "10": 0 },
  "track_volumes": { "10": 0.85 },
  "track_meters": { "10": 0.3 },
  "playing_positions": { "10": 12.0 },
  "crossfader": -0.25,
  "soloed_kinds": ["drums_a", "mix_b"],
  "deck_map_revision": 7
}
```

## Migration / compatibility notes

- **Additive, non-breaking.** Both fields default (`[]`, `0`) so any FE that
  ignores them is unaffected; old WS clients keep working.
- **AbletonOSC fork required only for the crossfader.** Solo + revision work on
  stock AbletonOSC. The crossfade-assign call no-ops gracefully on stock — the
  user just won't get crossfader blending on live-loaded decks until they apply
  the one-line fork patch and **fully restart Live** (toggling the Control
  Surface is not enough; Live caches Python modules in `sys.modules`).
- **No DB / `.als` template change.** This is OSC + WS + in-memory bridge state
  only. The `.als` writer is untouched; it already emits the matching
  `CrossFadeState/Manual`.
- **`crossfader` was already in `to_dict()`** but absent from `AbletonStateOut`,
  so the `/ableton/state` REST response silently dropped it. Adding it to the
  model fixes that; the `/ws` push already carried it.

## Live-verification checklist (run in Ableton — backend dev cannot)

Prereqs: apply the `crossfade_assign` fork patch (above), **fully quit and
reopen Live**, select AbletonOSC as Control Surface, start the backend, open
the companion app, and `POST /ableton/load-track` (or use the UI) to create the
10 deck columns + load a track's stems.

### (a) Crossfader blends A vs B for app-loaded stems

1. Load one track's stems onto **Deck A** and a different track's stems onto
   **Deck B** (same scene or adjacent scenes).
2. Fire both decks so audio is playing from A and B simultaneously.
3. In Live, confirm each `Deck * A` track's mixer shows crossfade assign **A**
   and each `Deck * B` track shows **B** (the small `A`/`B` toggle under the
   crossfader area; `Deck Mix A/B` should show neither — "None").
4. Move the crossfader (APC40 hardware fader or on-screen) fully left → only
   Deck A is audible; fully right → only Deck B; center → both. Before this
   change the crossfader did nothing to these tracks.

### (b) PFL/Solo lights reflect Live's real solo state

1. With Solo/Cue mode = Cue (the backend sets this on init — verify the master
   strip's `Solo`/`Cue` toggle reads **Cue**), click the `S` button on
   `Deck Drums A` **directly in Live** (not via the app).
2. The companion's PFL indicator for that deck/kind should light up within a
   beat — confirm `GET /ableton/state` (or the `/ws` stream) shows
   `"soloed_kinds": ["drums_a"]`.
3. Solo a second deck (e.g. `Deck Mix B`) → `soloed_kinds` becomes
   `["drums_a", "mix_b"]` (canonical order, regardless of click order).
4. Un-solo both in Live → `soloed_kinds` returns to `[]`.
5. Also confirm the app's own `POST /ableton/pfl/{side}` round-trips: hitting
   PFL A in the UI lights the real `S` buttons on the A-side decks in Live AND
   shows up in `soloed_kinds`.

### (c) Deck-map updates are instant

1. Open the companion's SceneGrid and the `/ws` stream side by side.
2. Load a track (`/ableton/load-track`). Confirm `deck_map_revision` increments
   on the next WS push and the SceneGrid repaints **without** waiting for the
   ~2 s poll.
3. Clear a cell (the cell `X` button → `DELETE /ableton/decks/cell/...`):
   `deck_map_revision` bumps again, cell disappears immediately.
4. Run `/ableton/decks/resync`: `deck_map_revision` bumps once, grid reflects
   the adopted clips.
5. Confirm the counter only ever increases and never resets mid-session
   (a backend restart resets it to 0 — acceptable, the FE refetches on
   reconnect anyway).
