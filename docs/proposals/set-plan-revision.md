# Plan edits must not silently drop another writer's picks

**Status:** proposal, not implemented (2026-10-13)
**Touches:** the plan API (`/sets/{id}/plan*`) request/response shapes, so
CLAUDE.md rule 1 applies. No SQLAlchemy model change in the recommendation.
**Builds on:** `4b2b509` "sets: stop losing plan picks to concurrent writes"

---

## What 4b2b509 already fixed (not re-proposed here)

- Adds go through `POST /plan/append`, which merges server-side. The UI no
  longer adds with read-modify-PUT.
- A cold cache is no longer treated as an empty plan: `currentQueues()` returns
  `null` and callers refuse to run (`useSetPlan.ts:42-50`).
- `PUT` validates track ids before it persists anything.
- Every plan write holds a process-wide `_PLAN_WRITE_LOCK`, and append re-reads
  the row inside it (`sets.py:47-60`, `:620-632`).

The lock serialises the writers. It cannot help a writer that has nothing to
read, and `PUT` is that writer.

## The remaining hole

`PUT /sets/{id}/plan` (`sets.py:567-606`) replaces the whole plan. It never
reads the row, so it takes the lock only to overwrite whatever is there. Its
body comes from the tab's react-query cache (`useSetPlan.ts:65-72`). That
cache refreshes in only three cases: on mount when it is more than 5 s stale,
on reconnect, or after this tab's own mutation (`useSetPlan.ts:32`, `:97`).
Refetch-on-focus is off (`main.tsx:11`) and `useSetPlan` has no
`refetchInterval` (`useSetPlan.ts:13-18`). `useActiveSet` does poll every 5 s, with
the comment "other tabs / ⌘K add to the plan" (`useSets.ts:30-31`), but it
refreshes `["sets","active"]`, not `["sets",id,"plan"]`, so that poll never
reaches the grid.

| Writer | Path | Safe? |
|---|---|---|
| Grid ＋ / `enter` (`RoleColumn.tsx:316,350`, `RoleColumnsGrid.tsx:85`) | `POST /plan/append` | yes |
| ⌘K (`CommandBar.tsx:504,577`) | `POST /plan/append` | yes |
| Grid × (`RoleColumn.tsx:531`) | `PUT /plan` built from the cache | **no** |
| `moveInRole` (`useSetPlan.ts:74-83`) | `PUT /plan` | no, but nothing calls it |
| a future MCP server, a second tab, `curl` | append today; needs remove/move later | |

The result: a tab that has been open for twenty minutes can wipe everything
another writer appended in that time with a single × click. The other writer
got a 200 with its pick included. The clicking tab then renders the server's
response, so its user never sees the pick at all. Nobody gets an error.

### Measured, not theorised

I ran the real `create_app` under uvicorn, which uses a threadpool exactly like
production, against a throwaway DB in a scratch `HOME`. I did not touch
`~/.dance`. The script is a scratch file and is not committed. It also ran
throwaway prototypes of options C and A′ next to the real routes:

```
[1] tab A has cached song=[1,2,3]; another client appends 4; tab A clicks × on index 0
    POST /plan/append 4 -> 200, song=[1,2,3,4]
    PUT  /plan          -> 200, song=[2,3]
    final song=[2,3]    -> track 4 LOST, no error to anyone
    GET /sets/active updated_at changed on the append: True

[2] append(9) and remove(1) released together by a threading.Barrier, 100 trials/row, two runs
    today: remove = whole-plan PUT                      append lost  47 | 69
    option C prototype: POST /plan/remove op            append lost   0 |  0
    option A' prototype: PUT + expected_revision (hash)  append lost   0 |  0   (409 on 15 | 12)

[3] PUT /plan round-trip on localhost: median 2.1-2.9 ms
```

Row [3] shows that within one tab the in-flight window is a few milliseconds,
so racing your own ⌘K by hand is unlikely. The real exposure is
**cross-writer staleness, and that is unbounded**. A second tab is a mild case
of it. An MCP agent appending while the Set view stays open is the case it was
built for.

## Options

### A. Revision counter + `expected_revision` on PUT → 409

`sets.plan_revision INTEGER NOT NULL DEFAULT 0`. This needs an Alembic
revision on head `e4a91c7b2d05`, using `batch_alter_table` like
`b7d4e2f1a9c3`. Every plan write bumps the counter inside the lock.
`SetPlanOut.revision` carries it. `SetPlanUpdateRequest.expected_revision`
must match, otherwise the response is
`409 {"detail":"plan_changed","plan":<SetPlanOut>}`. On a 409 the client seeds
its cache from the body, re-applies its intent, and retries once.

- **For:** general. It works for any edit, including bulk replace and undo.
  It follows the `deck_map_revision` precedent (`bridge.py:60-64`).
- **Against:** a migration, and nothing applies it automatically.
  `init_db` only runs `create_all` (`database.py:832-836`), and
  `alembic upgrade head` only runs from `bin/setup.sh:60`. If you forget the
  upgrade, every `Set` query fails with "no such column", which breaks both the
  Booth and the Set view.
- **Against:** every client (the UI, the MCP server) has to write rebase-and-retry
  code. For every edit the UI makes, "re-apply the intent" means "remove track
  X from role R", and that is option C done N times on the client instead of
  once under the lock the server already holds.

**A′ (no column):** make the revision a content hash,
`sha256(encode_plan(parse_plan(s.plan)))[:16]`, computed in `_plan_out`. This
needs no migration. The ABA case (the plan changes and then changes back) is
harmless for a whole-value precondition. If the current content equals the
client's base, the client's replacement is exactly correct. Prototype [2]
confirmed it: no silent losses, and a guarded PUT from a pre-append snapshot
returns 409. **If a guarded PUT is ever needed, use A′, not A.**

### B. ETag / If-Match

This is A′ moved into headers: `ETag` on GET, append and PUT, `If-Match` on
PUT, and `412` on a mismatch ([RFC 9110 §13.1.1](https://www.rfc-editor.org/rfc/rfc9110#section-13.1.1),
[§15.5.13](https://www.rfc-editor.org/rfc/rfc9110#section-15.5.13)), plus
optionally `428` when the header is missing
([RFC 6585 §3](https://www.rfc-editor.org/rfc/rfc6585#section-3)).

- **For:** it is the HTTP-standard way to do this.
- **Against:** react-query caches the body, not the headers, so `getPlan`
  would have to return `{plan, etag}`. Every plan handler would also need a
  `Response` parameter to set the header.
- **Against:** `request()` spreads `...init` *after* `headers`
  (`api.ts:47-52`), so a caller that passes `If-Match` silently loses
  `Content-Type: application/json`. I checked FastAPI 0.103.2: it then answers
  `422`. That is a latent bug to fix in any case.
- **Against:** MCP tool calls carry JSON arguments, not headers.

There is no gain over A′ here, and it costs more.

### C. Server-side ops only (add / remove / move); retire the whole-plan PUT

`POST /plan/remove {role, track_id}` and `POST /plan/move {role, track_id,
to_index}` work like append: they take the lock, re-read the row, apply the
op, and write it back.

- **For:** no update can be lost, by construction, and no client needs merge
  code. Prototype [2] measured 0 lost.
- **For:** it finishes what 4b2b509 started. Its own comment says "No
  server-side remove endpoint exists, so this stays read-modify-PUT"
  (`useSetPlan.ts:67`).
- **For:** the ops are the same size as MCP tool calls. No schema or migration
  change.
- **Against:** an op addressed by index breaks when the client's list is stale.
  So ops address by `track_id`, and a stale `to_index` is clamped into the
  current list. Append already de-duplicates (`sets.py:626-627`), so a
  `track_id` is unambiguous within a role.
- **Against:** with PUT retired there is no bulk replace. Undo, "import this
  plan" or an MCP "write me a whole plan" tool would each need a new endpoint,
  and that endpoint would be a guarded PUT using A′.

### D. Last-write-wins + a `set_changed` WS broadcast + refetch

- **For:** it fixes *display* staleness, which A, B and C do not. An MCP pick
  would appear in an open grid without a remount.
- **Against:** it only shrinks the window; it does not close it. Row [2] above
  lost 47–69% with a cache that was a few milliseconds old.
- **WS shape:** a discrete frame does not work on `/ws`, for two reasons:
  - `useAbletonState.ts:24-28` assigns *every* frame to `AbletonState`, so a
    `{"type":"set_changed"}` frame would blank tempo and clips until the next
    push.
  - The in-flight `claude/ws-shared-socket` branch (`f4852e0`) coalesces `/ws`
    to newest-wins at ≤30 Hz, so a discrete frame can be dropped.
- **WS shape that does survive:** a counter inside a snapshot, the way
  `deck_map_revision` does it (`bridge.py:540-549`, `useDeckMap.ts:14-36`).
  For example, `AbletonState.plan_revision`, bumped after a commit. This
  changes `AbletonStateOut` (`schemas.py:415`) and makes a DB concern ride on
  the OSC bridge state. The alternative is a new `/ws/sets` channel that uses
  the unthrottled `broadcast_threadsafe`, like `/ws/pipeline`.
- **D-lite, with no backend change:** the 5 s `useActiveSet` poll already
  returns `updated_at`, and every plan write bumps it (`sets.py:603`, `:629`;
  confirmed in [1]). The plan grid always shows the active set (`Booth.tsx:51`,
  `SetEditor.tsx:20,33`). So invalidating the plan query whenever that value
  changes caps the staleness at 5 s.

## Recommendation

**C + D-lite.** C closes the loss. D-lite closes the visible staleness. Neither
needs a migration or a WS contract change.

**Backend (`sets.py`, `schemas.py`).** Add one helper,
`_mutate_plan(session, set_id, fn)`. It takes the lock, refreshes the row,
parses the plan, applies `fn`, encodes the result, commits and returns
`_plan_out`. Move append onto it, and add:

| Endpoint | Body | Absent track |
|---|---|---|
| `POST /sets/{id}/plan/remove` | `SetPlanRemoveRequest{role, track_id}`, removes every occurrence | 200, no-op |
| `POST /sets/{id}/plan/move` | `SetPlanMoveRequest{role, track_id, to_index}`, index clamped | 200, no-op |
| `PUT /sets/{id}/plan` | **deleted**, along with `SetPlanUpdateRequest` (`schemas.py:395-399`) | |

A no-op returns 200 rather than 404 so that a stale tab clicking × on an
already-removed pick ends up in the right state instead of raising an error.
`SetPlanOut` is unchanged.

**Frontend.**
- `api.ts` gains `removeFromPlan` and `moveInPlan`, and drops `putPlan`.
- In `useSetPlan.ts`, `removeFromRole(role, trackId)` and
  `moveInRole(role, trackId, to)` call the ops. `currentQueues` and the `put`
  mutation are deleted.
- `RoleColumn.tsx:531` passes `item.track_id`.
- For D-lite, `useSetPlan` adds roughly 8 lines: an effect on
  `useActiveSet().data?.updated_at` that runs
  `invalidateQueries(["sets", id])`. `["sets","active"]` is already polled
  every 5 s, so the only new traffic is one plan GET each time the value
  actually changes.

**Tests.**
- New file `tests/test_set_plan_ops.py`:
  - remove and move by `track_id`;
  - no-op on an absent track;
  - 400 on a bad role, 404 on a missing set;
  - a `Barrier` race of append ‖ remove that never loses the append. As in
    4b2b509, this test must be shown to **fail** when remove is routed through
    a whole-plan write.
- `test_api.py` tests that move onto append/remove: `:3418`, `:3451`, `:3459`
  (which seeds through PUT at `:3467`), `:3602`, `:3652`.
- vitest:
  - `removeFromRole` sends the `track_id`;
  - a change in the active set's `updated_at` invalidates the plan query;
  - update the `usePlanMutations` mocks that return `put`
    (`gridKeys.test.tsx:33`, `RoleColumnsGrid.test.tsx:21`).

**Migration cost.** No data migration and no Alembic revision. The companion
app is the only API client (the MCP server is not built yet), so deleting PUT
breaks nothing outside this repo. The backend and frontend ship in one PR.
Roughly +60 lines of backend code and −40 lines of frontend code, before tests.

The multi-worker caveat in the lock comment (`sets.py:55-59`) still applies
unchanged.

## Decisions for the user

1. **C + D-lite (recommended), or A′?** A′ keeps client-side editing and adds
   a content-hash `expected_revision` to PUT, with client rebase and retry.
2. **Retire PUT now (recommended), or keep it guarded with A′?** Keeping it
   only makes sense if an MCP "replace the whole plan" tool or undo is
   definitely coming.
3. **Build `/plan/move` now (recommended; it lets PUT go and gives the MCP a
   reorder), or delete the dead `moveInRole` and defer?**
4. **Remove/move on an absent track: 200 no-op (recommended), or 404?**
5. **Freshness: D-lite keyed on `updated_at` (recommended), a plain
   `refetchInterval: 5_000` on `useSetPlan`, or a WS push?** A WS push means
   a counter in a snapshot, which changes `AbletonStateOut` and needs its own
   go-ahead.

## Verification plan

1. Run
   `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q tests/test_set_plan_ops.py tests/test_api.py -k plan`.
   Include the fail-first check on the race test.
2. Re-run the scratch race against the real endpoints, with the prototype routes
   swapped out. Expect `append lost 0` over 100 trials, against 47–69 today.
3. Browser check against a scratch backend: point `DANCE_API_TARGET` at it
   (`vite.config.ts:7`). With the Set view open, `curl` an append and confirm
   it shows up in the grid within 5 s. Then click × on a different pick and
   confirm the curled pick survives. Repeat once on the real active set.
4. Run `npx tsc -p tsconfig.json && npx vitest run` in `companion-app/`.
