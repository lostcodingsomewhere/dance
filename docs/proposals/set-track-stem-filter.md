## Proposal — Per-slot stem filter on `set_tracks`

**Status:** approved 2026-05-23 (user picked "Option C" in conversation, implementing same day)
**Builds on:** [`./set-rail-and-search-consolidation.md`](./set-rail-and-search-consolidation.md)

## TL;DR

Add an optional `stem_kinds` list to each Set track entry so the DJ can
pre-plan stem-level intent ("this slot appears only as drums") without
exploding the data model into a fully grid-based set.

Default = NULL = load all 4 stems (today's behavior). When set, only
those stems load when the slot is force-loaded from the rail or
editor.

## Problem

The Set Rail captures planning intent at the **song level**. The
SceneGrid + per-column rec banners + ⤧ song-splitter capture
**performance-time** stem decisions. There's a missing third layer:
*intentional* stem-level planning ("for song 3, I want just the
vocals over the previous combo").

Today this lives only in the DJ's head — they remember to drop the
bass live, or use the splitter ad-hoc. The plan can't record it.

## Considered alternatives

**A — current (song-only plan, stems live).** Matches how ~95% of DJs
think. Misses intentional stem-level planning. Status quo.

**B — fully grid-based set (each set entry is a 4-slot scene).** Max
fidelity but huge complexity. Conflates the live SceneGrid (now) with
the plan (next). Combinatorial explosion in the rail. Rejected.

**C — song-ordered plan + optional per-slot stem filter (this).**
Minimal data model change: one nullable JSON-list column. Mental
model stays "ordered songs," each can be annotated. The ⤧ splitter
remains the *exploratory* version of the same idea; this is the
*planned* version.

## Proposed change

### Schema

Add `stem_kinds` column to `set_tracks`:

```python
stem_kinds = Column(Text, nullable=True)
```

Stored as a JSON string of stem kind strings, e.g. `'["drums","vocals"]'`.
NULL means "all stems" (today's default). Empty list is invalid (use
NULL instead). Valid kinds are the `StemKind` enum values
(`drums`/`bass`/`vocals`/`other`).

JSON-as-Text follows the existing project convention (see
`TrackEdge.meta`). Alembic revision adds the column nullable, no
backfill needed.

### API

`SetTrackOut` gains:

```python
stem_kinds: list[str] | None = None
```

`SetTrackAddRequest` and `SetTrackUpdateRequest` accept the same.
Router validates each value is in `StemKind`; rejects non-list,
non-string, or empty list (use null to clear).

### FE

`SetTrack` type gains `stem_kinds: string[] | null`.

Rail row + editor row render a small chip next to the energy bar:

- `ALL` (gray) when null
- `DR·BA` or similar 2-letter stems when filtered (violet)

Click chip → popover with 4 toggles (one per stem kind). Save updates
via the existing `PATCH /sets/{id}/tracks/{tid}` endpoint.

The shift-tap force-load action passes the filter to
`pushTrackToLive({ kinds })` (which already supports the parameter —
the per-column rec banner uses it for single-stem loads). Default
behavior unchanged for slots with NULL.

## Trade-offs

| Concern | Resolution |
|---|---|
| **Adds a planning verb the user has to think about.** | Default NULL = today's behavior. Most slots stay "ALL"; the chip is unobtrusive when not engaged. |
| **Tail-rec scoring doesn't know about it yet.** | v1 ships scoring unchanged. A v2 follow-up can bias scoring toward strong-drums candidates when the next planned slot is drums-only. |
| **JSON-in-Text isn't queryable.** | Fine — we never need to filter `set_tracks` by stem_kinds at the DB layer. It's a per-row decoration. |
| **Could collide with the ⤧ splitter UX.** | They're complementary: ⤧ is "split this rec now (exploratory)"; stem_kinds is "this set slot is intentionally limited (planned)." Each has a clear home. |

## Migration

Phased, each step shippable.

1. **Schema + CRUD + tests** — backend only, FE keeps ignoring the field.
2. **Rail chip + picker** — read + edit, no force-load behavior change.
3. **Force-load respects filter** — shift-tap in rail and "load" in editor pass `kinds` through.
4. *(future)* **Tail-rec biasing** — score candidates for their strength in the next-planned slot's stems.

## Out of scope

- Grid-as-set (alternative B above).
- Per-stem note (e.g. "use only the drop section of vocals").
- Stem-level tail recs (defer — current scoring is whole-track).
- "Combine with previous slot" layering — interesting but a separate concept.

---

**Implementation log:** committed in <commit-sha> on 2026-05-23.
