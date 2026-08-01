# Where recs live: planning vs performing

**Status:** DECIDED (partially shipped) — the redesign is deliberately deferred
**Date:** 2026-08-01
**Question:** should building a set and performing it be separate flows, or one
surface in different modes?

---

## Answer

**One surface. And the thing that should differ between the two is not a mode
flag — it's whether there is anything to score against.**

The `mode="plan"` / `mode="live"` prop on `RoleColumnsGrid` stays. It is the
best structural decision in the app: one component, two props, and the muscle
memory you build while planning transfers directly to the Booth. Splitting into
two apps would throw that away and buy nothing.

But "planning" and "performing" were never really two activities. They are two
points on one axis: **how much context the recommender has.** When nothing is
playing you are shopping against your plan; when a combo is running you are
shopping against the combo. Same surface, same card, same position on screen —
different scoring source, decided by the data, not by which tab you clicked.

## What was actually wrong (measured, not theorised)

The user's instinct — *"during live we don't need recs, we need the pre-defined
set"* — was half right, and the wrong half is the interesting one.

`docs/dj_ux_flow.md` already contains the resolution in its three-clock model:
NOW (0–8 bars, ears and hands, no reading) · NEXT SWAP (5–60 s, shopping) ·
SET ARC (10–60 min, shape). **Both the swap clock and the arc clock run during a
live set.** The plan answers the arc; the recs answer the swap. Neither goes
away. Removing live recs would gut locked decision #4 in `vision.md` and the
"surprise myself" success criterion.

The real defect was never recs-versus-plan. It was that **the recs were lying**:

```
Booth, nothing playing — /recommend/by-column, empty combo:
  drums   Bad Memories (0.00) | Ghosts Again (0.00) | Massive (0.00)
  bass    Bad Memories (0.00) | Ghosts Again (0.00) | Massive (0.00)
  vocals  … identical …
  other   … identical …
  mix     … identical …
```

With no combo, no trailing plays and no master tempo there is not one computable
feature. `scoring.combine` renormalizes over the empty set and returns 0.0 for
every candidate, so the "ranking" is database order — rendered five times. That
is the state the Booth **opens in**, so roughly twenty cards of noise were the
first thing anyone ever saw.

Two related defects found the same way:

- The SONG column's live recs were **always empty** — `useColumnRecs` was handed
  the plan role `"song"` where the recommender's feed is `"mix"`. That is the one
  column a whole-track A/B set is built from.
- A deck's ▶ fired the **wrong track** from the second track of a set onward.
  See the deck-arm commit; unrelated to recs but the same class of "never
  actually played it" bug.

## Shipped

- `useColumnRecs` exposes `hasContext` and does not query without it.
- Cold Booth **with a set** → plan-scored recs, labelled "from your plan".
  Verified live: five columns went from identical/0.00 to 55 / 50 / 78 / 63 / 75.
- Cold Booth **with no set** → says so, instead of faking a ranking.
- SONG column maps through `visRole`.

That is the whole of the "where do recs live" answer that was worth building
today, and it required no schema change.

## Deliberately NOT built

A judge panel scored four independent designs (two-surfaces, one-surface-modes,
progressive-disclosure, plan-as-spine) against day-one simplicity, expert power,
build cost and vision fit. **Every design scored 3–4 out of 10 on day-one
simplicity**, and the sharpest critique landed on the most ambitious one:

> The Booth surface is a plan-playback device for a user who has never made a
> plan. There is exactly ONE set, its entire plan is `{"other": [1]}`, and
> `session_plays` has 6 rows ever.

That is the correct objection, and it applies to the whole exercise. Designing
the planning↔performing flow before a single set has been performed is designing
against imagination. The ideas worth revisiting **after** the first real set:

- **Density by `is_playing`.** Nothing playing → collapse the live rail, give the
  grid the room. Something playing → collapse the plan band to next-up-only and
  shrink recs to ~3. The cold-open fix above is the cheap half of this.
- **Role dormancy.** Roles with no picks and no plays collapse to a 44 px rail,
  so day one is four narrow rails and one wide SONG column — a two-deck song DJ
  app without being a different app.
- **Keyboard audition loop.** `↑/↓` move, `space` cue to headphones, `enter`
  commit. Planning means auditioning fifty candidates and it is currently 100%
  mouse. Cheapest large win once planning actually happens.
- **The hole detector.** When a deck is firing and one role's fader sits near
  zero for several seconds, you have pulled that stem out on the APC40 — wake
  that column and surface candidates. This is how a song DJ discovers stem
  swapping: by doing what they already do and being met.

## Schema

**No change.** The plan stays `{role: [track_id, …]}`. "Where am I in the plan"
is derivable from `session_plays` ∩ plan when it is needed; it does not need a
stored cursor.

## Known, not fixed

Adding a rec to the plan from the Booth (＋) invalidates `["sets", setId]` but
not `["recommend", "by-column", …]`, so the queued track stays in the live rec
list. Live recs exclude *playing* tracks, not *queued* ones — arguably correct
(you may still want it), arguably noise. Left alone deliberately: it is a design
question the first real set will answer.
