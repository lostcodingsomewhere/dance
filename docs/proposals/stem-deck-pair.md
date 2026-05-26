# Stem Deck-Pair (A/B Columns)

Status: **proposal — not implemented**
Spawned from: "is it possible to fire off multiple stems within the same column?" + the realization that classic DJ rigs run *two* decks per role for a reason.

## The pain we're solving

Today each stem role (drums / bass / vocals / melody) is **one** Live track. Live's session view enforces one-clip-at-a-time per track, so:

1. **Can't layer stems.** Kick from track A + hat from track B = impossible without stopping one of them.
2. **Can't transition smoothly.** When you swap a drum stem from song A → song B, the swap is instant. There's no overlap, no crossfade, no "ride them both for 4 bars and then drop A."
3. **Setting up the next song mid-set is destructive.** Previewing in the Cue bus is fine, but committing to a deck row replaces what's there. You can't *pre-stage* the next song with audio routed to master but at zero volume, the way a real DJ rig lets you.

The deck-pair model — A and B sides of every role — solves all three with one mechanic the user already knows from CDJs.

## Proposed shape

### Bridge: 9 deck tracks instead of 5

```
_DECK_KINDS = (
  "mix",          # reference, muted, unchanged
  "drums_a",  "drums_b",
  "bass_a",   "bass_b",
  "vocals_a", "vocals_b",
  "other_a",  "other_b",
)
```

Each `*_a` and `*_b` track is identical in routing — both go to master. The "side" is purely an organizing principle; volume + sends are how you actually crossfade. The APC40's crossfader is the optional shortcut: map A-side tracks to crossfader bank A, B-side to bank B, and you get a hardware crossfade for free without touching the FE.

### .als template

Currently has 2 audio tracks in the template that get cloned 5× by `writer.py`. The clone loop just needs to iterate the new `_DECK_KINDS` (9 tracks). The XML template itself shouldn't need re-authoring — color palette extends to 8 stems (4 colors × 2 shades, A = bright, B = muted variant of the same color) and the writer assigns them.

> Caveat per `CLAUDE.md`: every `.als` change needs Live-in-the-loop verification. After the writer change, regenerate one set, open in Live, confirm 9 audio tracks load + audio plays + nothing breaks. Bumping from 5 to 9 tracks shouldn't faze Live (it routinely has dozens), but the rule is the rule.

### SceneGrid layout — TWO options, pick one

Both keep the *conceptual* 4 stem roles + SONG visible. They differ on density.

**Option L: "Wide" — 9 visual columns + SONG = 10 columns**

```
| Drums A | Drums B | Bass A | Bass B | Vocals A | Vocals B | Melody A | Melody B | SONG |
|  cell   |  cell   |  cell  |  cell  |   cell   |   cell   |   cell   |   cell   | cell |
```

Pros: most legible, matches the deck-pair mental model directly, each cell is independently clickable.
Cons: 10 columns on a 14" MBP at 1512px logical = ~140px per column before the rec banner. Tight. The track-title text would need to truncate harder. Drag-and-drop reorder gets weirder.

**Option C: "Compact" — 5 visual columns, A/B stacked in each cell**

```
| Drums |  Bass  | Vocals | Melody | SONG |
|  A    |   A    |   A    |   A    |      |
|  B    |   B    |   B    |   B    |      |
```

Each column shows two stacked half-height cells. The colored column headers (DRUMS / BASS / ...) stay 5-wide and unchanged. Each cell tap fires the A or B clip in that role; the row label still anchor-fires the whole row.

Pros: existing column language unchanged, identical horizontal real estate, only the row height doubles. Drag-and-drop column reorder stays 5-wide.
Cons: row gets ~2× taller, so fewer rows fit on screen before "show all 8 rows" kicks in.

**My recommendation: Option C.** The column language is *the* mental model the DJ has internalized — keep it. Doubling row height costs us maybe 2 rows of visible grid which is recoverable by tightening padding / using compact cell render.

### MasterVisualizer + ComboStrip

Today the master visualizer shows 5 stacked-stem waveforms (the 4 active stems + the mix reference). With A/B, the playing combo could draw from any subset of A-side + B-side. Two reasonable behaviors:

- **Single waveform per role, blended.** If only A is playing, show A. If only B, show B. If both, show them overlaid with A in front and B as a softer ghost behind. Tells the eye "you're blending songs."
- **Two waveforms per role.** Stack them — same as the cell split, just at visualizer scale. Honest but doubles vertical space.

I'd start with blended overlay and only upgrade if it gets confusing.

### Recommendations

`useColumnRecs(column)` today scopes by `"drums" | "bass" | "vocals" | "other" | "mix"`. Two options:

- **Scope per side**: `drums_a` and `drums_b` get *separate* rec feeds. The rec banner above each split half-cell shows recs for that side. Cleaner mental model: "what should I bring in on side B?" But UI cost — twice as many rec banners.
- **Scope per role, target picked at load time**: `drums` recs surface as before. The load button has a side picker (or auto-picks the empty side; falls back to A if both full).

I'd start with **role-scoped + auto-pick** (low UI cost). The DJ can shift-click to force the other side if they want.

### Cue / preview

Already independent (the Cue track is its own track, outs 3/4 → headphones). No changes needed — preview a stem in headphones regardless of which side ends up holding it.

### Persistence migration

`_deck_cells` is `dict[(scene_idx, kind), track_id]`. Existing kinds are `"drums"` / `"bass"` / ... and will become `"drums_a"` / `"drums_b"`. The persisted JSON would need a one-shot migration: rewrite `"drums"` → `"drums_a"` on load (then save back). Single try/except in `_restore_state`.

API: `DeckCellOut.kind` becomes any of 9 strings instead of 5. Frontend `StemRole` type widens. Most code that switches on role can branch on `role.startsWith("drums")` etc.

### .als writer offline path

Same swap. `STEM_ORDER` expands to 9; the offline-load defaults to dropping each stem into its `*_a` slot. The `*_b` slots stay empty by default (the offline path is a *snapshot* of one track's stems — no second song to load yet).

## What this enables in practice

| DJ move | How it works with deck pairs |
|---|---|
| Hard-cut to next song | Load song B into B-side, fire B-side scene, panic-stop A-side |
| Long blend | Load song B, slowly bring up B's vocals/melody, drop A's drums, swap drum stem, etc. |
| Drop the bass | Mute A's bass, load B's bass to B-side, fire on the next bar |
| Layer fills | A-side plays the song; B-side holds a one-shot drum fill that fires for 2 bars and stops |
| 4-deck mashup | Drums-A from song 1, Bass-A from song 2, Vocals-B from song 3, Melody-B from song 4 |
| Hardware crossfader | APC40 crossfader maps to "fade between A and B," same as a real DJ rig |

## What I want from you before I write code

1. **Layout: L (wide, 9 columns) or C (compact, A/B stacked)?** I'd pick C.
2. **Rec scope: role-only with auto-pick side, or per-side recs?** I'd pick role-only first.
3. **Visualizer: blended overlay or stacked?** I'd start with blended.
4. **Crossfader hookup: do you actually want APC40 crossfader → A/B routing wired, or leave the hardware fader unmapped and just use the per-track volume faders?** This affects whether we set `MixerDevice.CrossFadeAssignment` in the .als template per-track.

If those choices feel right I'll write a follow-up commit that lays out the bridge change first (small, isolated), then the writer change with a real-data Live verification step, then the SceneGrid layout. Three commits, gated on you opening Live so we can prove the .als still loads at every step.

## Open risks

- **9 audio tracks in Live, all routed to master** — CPU load. Each track is just a clip player so it shouldn't matter, but worth a 30-second smoke test in a real set.
- **Persistence migration is one-way** — a user on an old commit who tries to load the new persistence file would see crashes. Migration is forward-only; flag in the commit.
- **SetRail / SetEditor's "load this song" gesture** assumes 4 stems → 4 columns. Will need to learn "load to which side" (auto-pick empty side, fall back to A).
- **Existing tests** — `test_osc.py` mocks 5 deck columns; expanding to 9 will break those assertions. Manageable, just flagging.
