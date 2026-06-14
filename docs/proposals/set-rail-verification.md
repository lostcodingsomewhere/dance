> **SUPERSEDED (2026-06-14)** by the plan-grid model — the Set Rail (⌘\\) and its tail-recs no longer exist on screen; planning now happens in the `RoleColumnsGrid`. See [`./rec-brain-port.md`](./rec-brain-port.md) and the Sets section of [`../api.md`](../api.md).

## Set Rail consolidation — real-set verification handoff

**Parent proposal:** [`./set-rail-and-search-consolidation.md`](./set-rail-and-search-consolidation.md)
**Status:** Phases 1–5 implemented (DB + scoring + rail + Cmd-K + Crate retirement).
**This doc:** the manual real-set verification I can't perform autonomously, per CLAUDE.md workflow rule 2.

## Why this needs you, not me

The proposal's Phase 6 is a 30-minute real set with Ableton running, the
Scarlett 4i4 wired, and the APC40 plugged in. Automated tests cover the
Python and TS contracts (292 + 16 passing) but they can't tell us whether
the *flow* is right when you're standing at the gear and reading the
screen mid-mix. The questions below are the things I'd want a sanity
check on before declaring the consolidation done.

## What changed (one-page summary)

| Area | Before | After |
|---|---|---|
| Saved set concept | none (Stack in localStorage) | `sets` + `set_tracks` tables, named, switchable |
| Tail-recs | none | `GET /sets/{id}/tail-recs` scoring by trailing embedding / key walk / BPM median / energy slope |
| Pre-set surface | Crate view | retired — Pipeline shows inventory, Cmd-K finds anything |
| Live "queue" reach | `pinnedSongRecs` (localStorage, Mix-only) | `<SetRail>` drawer in Booth (⌘\\), or full-pane `/set` editor |
| Search | Cmd-K vibe-only | Cmd-K hybrid: fuzzy title/artist first, vibe results below; BPM/key chips |
| Nav | Booth / Crate / Pipeline | Booth / Set / Pipeline |

Migration prompt: on first load with a non-empty legacy `localStorage.stack`,
a one-shot modal offers to import it as a named Set.

## Run book

```bash
# Apply the migration (only needed once per DB)
alembic upgrade head

# Start backend
uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000

# Start FE
(cd companion-app && npm run dev)
```

Open Ableton Live 12.4 → start AbletonOSC → minimize Live. Open
http://localhost:5173, plug in the APC40 + Scarlett, headphones in
outs 3/4, speakers in outs 1/2.

## What to verify (checklist)

Skim the list, do the steps in order, jot what felt wrong. Each item is
~30–90 s.

### Set lifecycle

- [ ] Cold load → no active set → SetRail edge pill shows "no active set"
      with the open arrow. Click it → drawer with "create empty set" CTA.
- [ ] Create a set named "Verification". Add 6–8 tracks via the rail's
      ⊕ from ⌘K button. ⌘K opens, type a partial artist name, results
      land in the "Tracks" section above the "Vibe" section.
- [ ] In Cmd-K, "+ Set" on a row adds it to the current Set without
      closing the palette. "Load" loads to Live and auto-closes.
- [ ] Switch to the Set view (top nav). Verify reorder via ▲/▼ persists
      after a hard reload. Add a 📝 note to one track; verify it shows
      as the amber indicator in the rail back in Booth.
- [ ] Create a second set, switch via the dropdown in the Set view
      header. Verify only one set is `is_active=true` at any time
      (curl `/api/v1/sets`).
- [ ] Delete a set — confirm dialog → it's gone from the switcher list.

### Tail recs quality

- [ ] With ≥3 tracks in the set, the rail's "Tail recs" section shows
      candidates with scores in the 70–100 range. Hover for the "why"
      tooltip (key, BPM, energy reasons).
- [ ] Add a track with a very different BPM (e.g. set is 124, add a 95
      BPM track at the end). Tail recs should shift toward 95 — verify
      by checking the BPM of the top suggestion.
- [ ] Build an upward energy ramp (3 → 4 → 5 → 6) and see if the top
      tail rec lands near energy 7. If not, the projection might need
      tuning or the candidate pool is too thin.
- [ ] Toggle `?exclude_session_plays=true` (via the API) after playing
      a couple tracks via Ableton — verify those tracks vanish from the
      tail-recs list.

### Live performance, rail behavior

- [ ] Open the rail in Booth, fire a clip. The rail should auto-close
      ~3 s later — eyes back on the SceneGrid.
- [ ] Tap (no shift) a set track in the rail → that track shows up at
      the top of the Mix-column rec banner as a pinned card.
- [ ] Shift-tap a set track → it loads into Live (next empty stem
      column). The combo updates, recs re-score.

### Cmd-K hybrid

- [ ] Type a partial artist → "Tracks" section returns results in
      <300 ms; vibe section stays hidden under 8 chars.
- [ ] Type a vibe phrase ≥8 chars ("punchy techy with vocals") → both
      sections populate; "Vibe" is below "Tracks" with ✦ score badges.
- [ ] Set the BPM ≥ chip to 130 → both sections re-query and only show
      tracks at ≥130 BPM. "× clear" wipes the chips.

### Crate retirement

- [ ] Nav has exactly **Booth | Set | Pipeline** — no Crate tab.
- [ ] Pipeline's "Done" terminal column has a filter input; typing
      narrows the list in-place.

### Migration

- [ ] In a fresh browser profile, manually set `localStorage` to
      `{"stack": [<some real track ids>]}` then reload. The migration
      modal should appear once. "Import as Set" should create the
      named set, activate it, and stop appearing on subsequent reloads.
      "Skip" sets a dismiss sentinel and also stops the prompt.

### Old assumptions still holding

- [ ] ColumnRecBanner still works on Mix column when a track is
      soft-pinned (it now reads from in-memory `pinnedSongRecs` only,
      not localStorage — so a reload clears the pins, which is the
      new intended behavior).
- [ ] Per-column recs (drums / bass / vocals / melody / mix) still
      re-score when the combo changes.
- [ ] WebSocket heartbeat still green (Live online) when Ableton is up.

## Things I'm watching for

- **Tail-rec relevance**: I tuned the weight split as `_W_EMBED=0.40 /
  _W_KEY=0.25 / _W_BPM=0.20 / _W_ENERGY=0.15` (see
  `src/dance/recommender/tail.py`). If candidates feel "off vibe but
  matching keys/BPMs," embedding weight may want bumping. If they feel
  "vibey but BPM-mismatched," BPM weight needs more.
- **Auto-collapse window**: 3 s is a guess. If you reach back to the
  rail mid-mix and it's already closed, push to 5–6 s. If it lingers
  through a clip launch and feels distracting, drop to 1.5 s.
- **Cmd-K vibe threshold**: I gate vibe queries at ≥8 chars to avoid
  spamming CLAP on every keystroke. If short phrases feel useful, lower
  it; if it fires too eagerly, raise it.
- **Closed-tab affordance**: the violet edge pill is intentionally
  minimal. If you forget the rail exists, surface it (label, larger
  hit area, badge for active set count).

## When you're done

Open an issue / message me with:
1. Which checklist items passed clean.
2. Which felt off (with one-line "what would be better").
3. Tail-rec gut-check on 3 example sets: which candidate would you
   *not* have picked, and which obvious one was missing?

Phase 7 (if needed) tunes from your notes. The bones are in place —
this is the "does it feel right" pass.

---

**Files of interest if you want to poke under the hood:**

| Concern | File |
|---|---|
| DB models | [src/dance/core/database.py](../../src/dance/core/database.py) (Set, SetTrack classes) |
| Migration | [src/dance/alembic/versions/c6e3ba31c1f0_add_sets_and_set_tracks_tables.py](../../src/dance/alembic/versions/c6e3ba31c1f0_add_sets_and_set_tracks_tables.py) |
| CRUD API | [src/dance/api/routers/sets.py](../../src/dance/api/routers/sets.py) |
| Tail-rec scoring | [src/dance/recommender/tail.py](../../src/dance/recommender/tail.py) |
| Fuzzy search | [src/dance/api/routers/tracks.py:50](../../src/dance/api/routers/tracks.py#L50) (`/tracks/search`) |
| Rail UI | [companion-app/src/components/SetRail.tsx](../../companion-app/src/components/SetRail.tsx) |
| Cmd-K | [companion-app/src/components/CommandBar.tsx](../../companion-app/src/components/CommandBar.tsx) |
| Editor | [companion-app/src/views/SetEditor.tsx](../../companion-app/src/views/SetEditor.tsx) |
| Migration prompt | [companion-app/src/components/StackMigrationPrompt.tsx](../../companion-app/src/components/StackMigrationPrompt.tsx) |
