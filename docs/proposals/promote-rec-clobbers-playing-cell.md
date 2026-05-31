# Promote-a-rec must enqueue, not clobber the playing cell

**Status:** proposed (2026-05-30)
**Touches:** `AbletonBridge.push_track_to_live` slot/side selection (OSC load contract) — hence this note per CLAUDE.md workflow rule 1.

## The bug

Promoting a recommendation from a stem column's rec banner overwrites the
currently-loaded (and often *playing*) stem cell instead of landing on a
free slot. A DJ with `drums_a` playing clicks "Load drums → Deck A" on a
candidate and the playing drums cut out mid-set, replaced by the new clip.

## Why it happens

The frontend is correct. `ColumnRecBanner.tsx` always sends an explicit
`side` ("a" or "b" from the two load buttons) and omits `scene_index`, so
the backend chooses the scene.

`push_track_to_live` resolves the scene *before* it honors the forced side
(`bridge.py` ~1334–1361). For a single-stem load it scans for the lowest
scene where **either** side A or B of the stem kind is free:

```python
while (i, f"{first_src}_a") in self._deck_cells and (i, f"{first_src}_b") in self._deck_cells:
    i += 1
scene_index = i
```

Then it applies the forced side. So with `drums_a` occupied/playing at
scene 0 and `drums_b` free, the scan stops at scene 0 (B is free), but the
forced side A writes `create_audio_clip(drums_a, 0, …)` straight over the
playing cell. The write loop has no free-cell guard.

(Auto-pick mode — no forced side — is *mostly* safe because `_pick_side`
prefers the less-full side, but it can still land on an occupied cell when
the same-kind cell is taken on the side that happens to be less full
overall. The frontend never uses auto-pick today, but the fix closes this
hole too.)

Whole-song loads are already safe: `next_free_row()` requires *all* eight
stem cells in a row to be empty, so it never overwrites.

## The fix

Resolve scene **and** side together for single-stem loads so the chosen
side's cells (for every requested kind) are empty:

- **Forced side** ("a"/"b"): scan for the lowest scene where *that side* is
  free for all requested kinds. Never overwrites the opposite or the same
  side's playing cell.
- **Auto-pick** (side omitted): scan for the lowest scene where *either*
  side is free; when both are free, defer to `_pick_side` (less-full,
  tie → A) as today.

New helper `_free_stem_slot(kinds, *, forced_side) -> (scene_index, side)`
encapsulates this. The existing either-side scan + post-hoc `_pick_side`
block is replaced by a call to it for the single-stem path. Whole-song and
explicit-`scene_index` paths keep `next_free_row()` / caller's scene and
the `_pick_side` fallback unchanged.

### Semantics changed

- A forced-side single-stem load now advances to the next scene whose
  forced side is free, rather than overwriting an occupied cell at an
  earlier scene. This is the intended "enqueue" behavior.
- An explicit `scene_index` still overwrites (caller pinned an exact cell —
  their call). Unchanged.

### Not changed

- Whole-song load slot selection (`next_free_row`).
- The A/B side semantics, deck-column layout, OSC message shapes.
- The frontend.

## Tests

`tests/test_osc.py`: add a case that loads a stem to side A at scene 0,
then promotes a second stem to side A again, and asserts the second lands
on scene 1 (forced side free there) — the scene-0 cell's track_id is
preserved. Also assert that a forced-A load when A is busy but B is free
does **not** touch the A cell.
