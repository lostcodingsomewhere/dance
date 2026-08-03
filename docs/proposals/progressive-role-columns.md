# Progressive role columns (column dormancy)

**Status:** proposal — not implemented
**Related:** [`dj_ux_flow.md`](../dj_ux_flow.md) "The audition loop", `RoleColumnsGrid`

## The problem

The plan grid always shows five role columns: DRUMS · BASS · VOCALS · OTHER ·
SONG. That is the right surface for stem performance and the wrong surface for
week one.

The stated on-ramp is *"basically A/B sides, with stems all separate… maybe we
start with just normal mixing A B with same respective tracks stems on each"* —
i.e. whole tracks on two decks, stems available but not yet the point. In that
regime four of the five columns are noise: they compete for the eye, they
split the recommender's attention across roles, and they make the first
decision ("what do I play next?") look like five decisions.

`session-1.md` already handles this by *excluding* the plan grid from the first
session entirely. That is a good answer for 45 minutes and a bad answer for
week two — the grid has to become usable gradually, not arrive all at once.

## What NOT to do

**Don't build a second, simpler surface.** The whole argument for the keyboard
loop is that one motion vocabulary transfers from planning to the booth. A
"beginner view" with different mechanics would teach habits that have to be
unlearned, which is worse than showing four columns too early.

So whatever this is, it must be the *same* grid with fewer columns visible —
not a different screen.

## Options

### A. Manual column visibility (toggle chips)

A row of five toggles above the grid; hidden columns collapse to a thin
labelled rail. Persisted per user, like `stemColumnOrder` already is.

- **For:** trivially predictable, no inference, reversible in one click, and
  the user decides when they're ready for bass. Reuses the existing persisted-
  layout pattern.
- **Against:** it's a setting, and settings get configured once and forgotten.
  Doesn't *teach* anything.

### B. Dormancy (auto-hide unused roles)

A column dims/collapses when it has no plan picks and hasn't been interacted
with; it wakes when it gets a pick, a load, or an arrow-key visit.

- **For:** zero configuration, and the grid grows as the DJ's practice grows.
- **Against:** a surface that changes shape on its own is hostile mid-set —
  the booth is exactly where "the column moved" is unacceptable. Also
  circular: a column stays dormant because it's hidden, so it never gets used.
  Would need dormancy in **Set mode only**, never in Booth, which then breaks
  the "identical surfaces" property the loop depends on.

### C. Song-first default, with the stem columns one keystroke away

Ship the grid defaulting to the SONG column only. `←/→` at the edge (or a
single `\` / tab-style key) reveals the stem columns. Nothing is hidden
permanently and nothing moves on its own.

- **For:** matches the stated on-ramp exactly (A/B whole tracks first),
  costs one keystroke to escape, and the keyboard loop already makes column
  traversal cheap. No new mental model.
- **Against:** a first-run default that some users will never discover past;
  needs a visible affordance ("4 more roles →").

## Recommendation

**C, with A as the persistence layer** — default to SONG-only, let the arrow
keys reveal the rest, and remember whatever the user last had open.

Rejecting B outright: the booth cannot have a self-reshaping grid, and a
behaviour that only exists in Set mode forfeits the transfer property that
justified the shared surface in the first place.

## Open question this proposal cannot answer

Whether five columns is *actually* overwhelming in practice, or whether it only
looks that way in a screenshot. Nobody has planned a real set on this grid yet.

The keyboard loop just changed the economics — with `←/→` costing nothing,
extra columns may read as available rather than demanding. That is a real
possibility and it argues for **doing nothing until the grid has been used in
anger**, then deciding from experience rather than from a guess.

Recommended sequence: plan one real set with the loop as shipped. If the stem
columns felt like clutter, implement C. If they felt like options, this
proposal should be closed unimplemented.
