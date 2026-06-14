> **SUPERSEDED (2026-06-14)** by the plan-grid model — a Set is now its `plan` (per-role queues inside the `RoleColumnsGrid`); the slide-out Set Rail (⌘\\) and its tail-recs are retired. See [`./rec-brain-port.md`](./rec-brain-port.md) and the Sets section of [`../api.md`](../api.md).

## Proposal — Set Rail + Cmd-K consolidation (pre-set flow rethink)

**Status:** approved 2026-05-23 — implementing Alt D, all 5 open questions locked to recommendations
**Date:** 2026-05-23
**Builds on:** [`./frontend-as-primary-surface.md`](./frontend-as-primary-surface.md), [`../dj_ux_flow.md`](../dj_ux_flow.md)

## TL;DR

The pre-set flow has accreted **four overlapping concepts** for "tracks in my world" — Library, Stack, pinnedSongRecs, DjSession — across two surfaces (Crate, Booth). Two of those concepts (Stack, pinnedSongRecs) are ephemeral localStorage state that should not be. Crate is mostly a worse re-render of what Pipeline already shows.

Collapse this to **two concepts + two surfaces**:

| Concept | Where | Persists |
|---|---|---|
| Library (everything ingested) | Pipeline view + Cmd-K | DB ✓ (already) |
| Set (curated plan, ordered) | Set Rail (drawer in Booth, expanded in editor) | DB ✓ (new tables) |

Three things change:

1. **Set Rail** — a slide-out sidebar in Booth (⌘\\), expandable to a full editor. Named sets, library of many, one active. Drag-to-reorder. Tail-recs at the bottom that score against the set's energy/key/embedding arc. Tap a rail item → soft-pin to Mix-column recs (reversible); shift-tap → force-load into a column.
2. **Cmd-K** — hybrid ranking: fuzzy artist/title first, CLAP vibe results below. Filter chips (BPM, key, energy, tag, in-set). Result actions: add-to-Set / load / cue-preview. Replaces Crate's filter UI.
3. **Crate view retires** — the Pipeline view already lists everything ingested; Cmd-K already finds anything. Two surfaces did one job.

```
┌────────────────────────────────────────────────────────────────────┬─────────────────┐
│ MasterStrip   124 BPM · 5A · ▰▰▰░ · ▁▂▄▆█ · ⌘K        Booth Pipeline│ SET · WAREHOUSE │ ← rail
├────────────────────────────────────────────────────────────────────┤  SAT (28)       │   (⌘\ toggles)
│ SCENE GRID · 8×5                                                   │  ─────────────  │
│  ▶ ▶ ▶ ▶ ▶                                                         │  1 ▶ Hyph 124   │
│     ▶  ▶                                                            │  2   Mort 125   │
│                                                                     │  3   Floti 124  │
│ MASTER · stacked stems                                              │  4   Tess  126  │
│  drums ~~~~|~~~~~                                                   │  …              │
│  bass  ~~|~~~~~~                                                    │  28  Rhei 122   │
│                                                                     │  ─ tail recs ─  │
│ COMBO   drums:A   bass:B   vocals:—   melody:C   mix:—              │  + Hessle 125   │
│                                                                     │  + Boyz   126   │
│ NEXT PER COLUMN · live re-scored                                    │  + Pearso 124   │
│  drums rec   bass rec   vocals rec   melody rec   mix rec           │                 │
├────────────────────────────────────────────────────────────────────┤  ⊕ from ⌘K      │
│ PlayedStrip   Set · 12 plays · 02:27 AM   [#1 Hyph][#2 …]  end set  │  ▾ load set...  │
└────────────────────────────────────────────────────────────────────┴─────────────────┘
```

## Problem

See [`../dj_ux_flow.md`](../dj_ux_flow.md). Three concrete pain points in the current pre-set flow:

1. **Stack is ephemeral.** You spend an evening curating "tracks I'm bringing to the warehouse gig," then close the tab and it's gone. There is no persisted, named "Set" object. `companion-app/src/store.ts:16` keeps it as a localStorage `number[]`.
2. **pinnedSongRecs duplicates Stack semantically** but lives in Booth, also localStorage, also unscoped to any named plan. Two stores for "tracks I want to reach for soon."
3. **Crate exists but doesn't earn its tab.** Pipeline already lists every ingested track with richer state info. Cmd-K already searches the library. Crate's only unique affordance is the filter chips — moveable.
4. **Cmd-K is vibe-only.** Great for "punchy techy with vocals" — useless for "play that one Four Tet track I know I have." A DJ needs both modes.

The redesign in [`./frontend-as-primary-surface.md`](./frontend-as-primary-surface.md) explicitly demoted SetRail and reorganized Booth around the SceneGrid + per-column recs. That was correct for the live moment. It left the **pre-set flow** unresolved — this proposal closes that gap without re-introducing the song-mode sidebar.

## Proposed change

Four shifts, mostly orthogonal — each ships standalone.

### Shift 1 — `sets` data model + CRUD

New tables in `src/dance/core/database.py`:

```python
class Set(Base):
    __tablename__ = "sets"
    id: int (pk)
    name: str                            # "Warehouse Sat", "Wedding 90min"
    notes: str | None
    created_at: datetime
    updated_at: datetime
    is_active: bool                      # exactly one true at a time (partial unique idx)

class SetTrack(Base):
    __tablename__ = "set_tracks"
    id: int (pk)
    set_id: int (fk → sets.id, on_delete=cascade)
    track_id: int (fk → tracks.id)
    position: int                        # 0-indexed, unique per set_id
    added_at: datetime
    note: str | None                     # "after the breakdown", "early energy"
```

Endpoints in new `src/dance/api/routers/sets.py`:

- `GET /sets` — list all (id, name, track count, updated_at)
- `POST /sets` — create (name)
- `GET /sets/{id}` — full set with ordered tracks
- `PATCH /sets/{id}` — rename, update notes
- `DELETE /sets/{id}`
- `POST /sets/{id}/tracks` — append track_id (or insert at position)
- `DELETE /sets/{id}/tracks/{track_id}`
- `PATCH /sets/{id}/tracks/{track_id}` — reorder (new position)
- `POST /sets/{id}/activate` — mark this set as the active one
- `GET /sets/{id}/tail-recs?k=10` — recommend next-appends given the set's trajectory (see Shift 2)

Migration: Alembic revision adds tables + a partial unique index on `is_active WHERE is_active`. On first FE load, if localStorage `stack` is non-empty, prompt: "We saved your Stack as a Set — name it?" and POST it as the active set. localStorage entries removed after migration.

### Shift 2 — `GET /sets/{id}/tail-recs`

This is the **Spotify-queue-tail intuition done right for a DJ**. Instead of "similar to last item," score by *next step in the arc*:

Inputs: the set's ordered tracks → derived signals
- Aggregate per-stem CLAP embedding (trailing 3-track window, weighted toward end)
- Energy trajectory (slope from last 3–5 tracks → projected next value)
- Key trajectory (Camelot walk: ±1 or relative major/minor)
- BPM band (recent median ± tolerance)

Ranking: candidates from `track_edges` (HARMONIC_COMPAT ∪ TEMPO_COMPAT ∪ EMBEDDING_NEIGHBOR), filtered to exclude tracks already in the set or already in the current `DjSession` plays, scored by combined arc-fit. Returns ranked `TailRec[]` shaped like existing `ColumnRec` so the rail can render with the same `TrackCard` component.

Re-computes on set mutation (lazy, on request). Sub-50 ms expected — same scale as per-column recs.

### Shift 3 — `<SetRail />` component

`companion-app/src/components/SetRail.tsx` — slide-out drawer.

**Modes:**
- **Closed** (default in Booth) — narrow 32 px tab on right edge with set name + count. ⌘\\ or click to open.
- **Drawer** (open in Booth) — 320 px right sidebar. Ordered tracks, drag-to-reorder, tail-recs section at bottom, "⊕ from ⌘K" footer button. **Auto-collapses 3 s after a clip fires** (eyes back to grid).
- **Expanded** (Set editor view) — full pane, two columns: set on left (with reorder + arc viz + notes per track), library/tail-recs on right. Replaces the Crate route at `/set` or `/set/:id`.

**Tap-target hierarchy** (from open question 1, locked):
- Tap a rail track → **soft-pin to Mix-column recs** (reversible; equivalent to today's pinnedSongRecs ↗ action). The track surfaces in Booth's Mix-column rec banner; nothing touches Ableton.
- Shift-tap or ⇧⌘L → force-load into next empty stem column (existing Load behavior).
- Right-click → context menu (preview / load to specific column / remove / move).

**Set switcher** in header chip: "WAREHOUSE SAT ▾" — dropdown lists all sets, "+ new", "duplicate current", "rename", "delete". Only one set is active (`is_active = true`); switching activates the chosen one server-side.

State: `useActiveSet()` hook fetches `GET /sets/{id_active}` on mount + WebSocket invalidation on `set.updated`. `pinnedSongRecs` localStorage state is retired; the same affordance now reads from the active Set.

### Shift 4 — Cmd-K rewrite + Crate retirement

`companion-app/src/components/CommandBar.tsx` becomes a hybrid palette:

**Ranking:** when the query is non-empty,
1. Run a SQL `ILIKE`/trigram fuzzy match on `tracks.title` and `tracks.artist` — top 5 hits, surfaced first under a "Tracks" header.
2. Run CLAP text-rec (existing `POST /api/v1/recommend/text`) — top 8 hits, surfaced below under a "Vibe" header.
3. Hide either section if it returns zero.

When the query is empty: show the active set's tracks + last 10 ingested + filter chips. Cmd-K becomes the library browser too.

**Filter chips** (inline below the input): BPM range slider, key (Camelot picker), energy (low/mid/high), tag multi-select, "in active set" toggle. Chips persist for the palette session.

**Result actions** on each row:
- `⏎` → primary action (configurable in settings: load-to-column-with-focus *or* add-to-active-set)
- `⌘⏎` → secondary
- `⇧⏎` → cue preview (Scarlett 4i4 outs 3/4)

**Backend additions:**
- `GET /tracks/search?q=<text>&limit=5` — fuzzy name/artist (Postgres trigram or SQLite `LIKE` with prefix index; fine either way at our scale).
- The existing `/recommend/text` endpoint is unchanged.

**Crate retirement:**
- Delete `companion-app/src/views/Crate.tsx`, `companion-app/src/components/Stack.tsx`.
- Remove Crate tab from MasterStrip nav. Tabs become **Booth | Pipeline**.
- Pipeline becomes the inventory surface (it already is, structurally). Add a small filter input + chip row to Pipeline's terminal column so "browse my library" still has a non-modal home for users who want to scroll.
- localStorage `stack` migrated to Sets (Shift 1) then removed.

## Alternatives considered

### Alt A — Keep Crate, just persist Stack

Smallest change: add `sets` tables, back Stack with a DB row, leave Booth + Crate + Cmd-K as is. **Con:** doesn't address the *pre-set ↔ live-set* mental seam. The user has to bounce between Crate and Booth during a set to consult the plan. The whole point of the rail is that the plan is always one keystroke away from the grid.

### Alt B — Set Rail in Booth, no editor pane

Rail is the only Set surface; you edit by drag-reorder in the drawer. No expanded editor route. **Con:** drag-reorder in a 320 px drawer with 30 tracks is painful. Editor view stays.

### Alt C — Sets stay client-side (IndexedDB), no backend tables

Pro: zero migration risk; tail-recs can call existing endpoints with a client-passed track list. **Con:** no cross-device sync, can't seed recs from server-side history, no schema validation, lost on browser nuke. We want this in the DB.

### Alt D (proposed) — Sets in DB + Rail + Cmd-K hybrid + Crate retires

Recommended. Coherent two-concept model. Plays cleanly with the live-set UI from the prior proposal — rail respects the grid by auto-collapsing on fires.

## Trade-offs

| Concern | Mitigation |
|---|---|
| **Rail in Booth steals attention from SceneGrid.** | Auto-collapse 3 s after any clip fire. Default-closed in Booth. Narrow tab affordance instead of permanent strip. |
| **New tables = migration risk.** | Additive only — no changes to `tracks` / `track_edges` / `sessions`. Alembic up-only; rollback is `DROP TABLE`. localStorage Stack migrates on first load via one-shot prompt; idempotent. |
| **Tail-rec quality is the whole pitch.** Bad recs make the section feel noisy. | v1 ships with a simple weighted combo (embedding + key + BPM compat) and an explicit "why" tooltip. If quality is weak, gate the section behind a toggle until tuned. Real-set verification (Phase 5) is the test. |
| **Cmd-K hybrid ranking can confuse users** (which list is which?). | Hard section headers ("Tracks", "Vibe") + visually distinct row treatment. Both lists are short (5 + 8). No interleaving. |
| **One-active-set constraint via partial unique index** has portability concerns (SQLite supports, Postgres supports, same pattern already used for `audio_analysis` per CLAUDE.md). | Reuse the same raw-DDL pattern from `init_db()`. Autogenerate will skip the index, same as today. Document in the migration. |
| **Two routes during transition (old Crate still rendered while Sets ship).** | Phase the rollout: Shifts 1–2 backend-only, Shift 3 lands rail behind a feature flag, Shift 4 removes Crate only after Sets is verified end-to-end. |
| **`is_active` flip is a global write** — if two browser tabs are open, last write wins. | WebSocket broadcast on activate; both tabs reflect the active set. Edge case but cheap to handle. |

## Locked decisions (approved 2026-05-23)

1. **Cmd-K primary `⏎` action** — host-aware setting: default **load to focused column** in Booth, **add to active set** in the Set editor view. Palette knows its context.
2. **Set arc viz** — tiny EnergySparkline at the top of the rail; full arc viz only in the expanded editor. Rail stays scannable.
3. **End-of-set behavior** — rail explicitly shows "set complete"; tail-recs section becomes the candidate list below. Explicit beats clever.
4. **Per-track notes** — editor only for v1. Rail shows a 📝 indicator if a note exists; hover/tap reveals it. Keeps the rail clean.
5. **Pipeline filter scope** — minimal v1: single text filter + tag chip row on the terminal column. Cmd-K is the real library search; richer Pipeline filters wait for explicit demand.

## Migration plan

Each phase shippable independently.

### Phase 1 — Sets schema + CRUD (~1 day)

- Alembic revision: `sets`, `set_tracks`, partial unique index on `is_active`.
- `src/dance/api/routers/sets.py` — full CRUD endpoints listed above (no tail-recs yet).
- `src/dance/core/database.py` — model classes.
- Tests: standard CRUD + position-reorder invariants.

**Validation:** `curl` round-trip; create a set, add three tracks, reorder, list, activate, delete.

### Phase 2 — Tail recs endpoint (~1–2 days)

- `GET /sets/{id}/tail-recs?k=10` — scoring logic in `src/dance/recs/tail.py`.
- Reuse arc-aggregation helpers from per-column rec scoring.
- Synthetic-audio test fixtures: build a 5-track set, assert tail recs prefer same-band BPM and adjacent Camelot.

**Validation:** real-data run — build a set from current library, eyeball the tail-recs against expectation (per CLAUDE.md rule 2).

### Phase 3 — Set Rail in Booth (~2–3 days)

- `components/SetRail.tsx` — drawer modes (closed/open), drag-reorder (use `@dnd-kit` if not already in deps; otherwise simple HTML5 DnD), tail-recs section.
- `useActiveSet()` hook + WebSocket invalidation.
- ⌘\\ keybind, auto-collapse on clip fire (3 s).
- Retire `pinnedSongRecs` localStorage state; Mix-column rec banner reads from active set's pinned items instead.
- Behind a feature flag (`VITE_FEATURE_SET_RAIL`) so it can land without disabling Crate.

**Validation:** open Booth → rail tab visible → open rail → add 3 tracks via ⊕ → reorder → tail-recs populate → tap a track → it shows in Mix-column recs → fire a clip → rail auto-collapses.

### Phase 4 — Cmd-K hybrid + Set editor view (~2 days)

- `GET /tracks/search?q=` fuzzy endpoint.
- `CommandBar.tsx` rewrite: hybrid sections, filter chips, contextual ⏎ action.
- `views/SetEditor.tsx` — expanded two-pane editor at `/set` route. Arc viz, per-track notes, drag-reorder, library browse on right.

**Validation:** Cmd-K finds "four tet" by name; same Cmd-K finds "deep rolling bass" by vibe. Set editor reorders persist via API. Notes save.

### Phase 5 — Crate retirement + localStorage migration (~½ day)

- One-shot FE check: if `localStorage.stack` non-empty, prompt to import as a named Set; POST to `/sets`, activate, clear localStorage.
- Delete `views/Crate.tsx`, `components/Stack.tsx`, store actions `addToStack`/`removeFromStack`/`moveInStack`/`clearStack`, `pinnedSongRecs` state.
- Remove Crate from MasterStrip nav. Nav becomes **Booth | Pipeline** (+ Set editor reachable via rail "expand" button or `/set`).
- Add minimal filter input + tag chips to PipelineBoard's terminal column.
- Flip feature flag default to on; remove flag a release later.

**Validation:** fresh browser session — Crate route 404s gracefully (redirect to Booth). Old user with localStorage Stack gets the migration prompt once; their tracks land in a named Set.

### Phase 6 — Real-set verification

Per CLAUDE.md workflow rule 2: plan a set for an actual gig — name it, populate it from Cmd-K, reorder, accept tail recs, save. Then play 30 min using the rail to surface candidates. Report what felt right, what got in the way of the grid, whether tail recs were trustworthy.

## Out of scope (explicitly)

- Multi-DJ collaboration on a shared Set.
- Public/shareable Set links.
- Auto-generated sets ("generate a 90-min progressive house set" — Future unlock).
- Per-track effects/cue preferences stored on `set_tracks`.
- Cross-set analytics ("which sets do I actually play?").
- Touch-first / iPad-responsive rail.
- Reorganizing Pipeline beyond the minimal terminal-column filter row.

## Future unlocks

If this lands:

- **Auto-generated sets** — "generate a 90-min set, start at 122 BPM, climb to 128, end at 125" using the tail-rec scorer in a loop.
- **Set templates** — save a set with track *roles* instead of track IDs ("opener", "peak", "comedown"), instantiate against current library.
- **Per-track DJ notes** elevated — voice memo, key reminders ("cue in at bar 33"), surfaced when the track loads.
- **Set replay** — replay a past `DjSession` as a Set to study what happened.
- **Crowd-feedback writeback** — thumbs up/down on what's playing feeds the tail-rec scorer for future sets at this venue.
- **Set arc auto-suggest** — "your last 5 plays trended down 12 BPM, want to course-correct?" surfaced inline.
- **Song → stem-rec split (inverse of `pinnedSongRecs`).** Today: stem card → ↗ pins as a whole-song candidate in the Mix-column rec banner (stem → song). Symmetric inverse: on a Mix-column song rec, a ⤧ "split" action pins each of its 4 stems as candidates in the matching stem-column rec banners (song → stems). Lets the user cherry-pick which stems to remix from a recommended whole song without committing the whole song to Mix. Cheap: stems already exist in `stem_files`; rec banners already render `ColumnRec`-shaped items. Net-new is a `splitToStemRecs(songRec)` store action + a button on the Mix-column rec card.
