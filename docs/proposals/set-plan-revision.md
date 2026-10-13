# Plan writes: a stale whole-plan PUT must not erase another writer's picks

**Status:** proposal, not implemented (2026-10-13)
**Touches:** the plan API's request/response shapes (`/sets/{id}/plan*`), so
CLAUDE.md rule 1 applies. The recommendation needs no SQLAlchemy model change,
no Alembic revision and no WebSocket change.
**Builds on:** `4b2b509` "sets: stop losing plan picks to concurrent writes".
**Related:** the dance-mcp proposal (branch `claude/proposal-dance-mcp`,
unmerged) lists this as a prerequisite for agent writes.

---

## What 4b2b509 already fixed (not re-proposed)

- Adds go through `POST /plan/append`, a server-side merge
  (`useSetPlan.ts:60-63`). The UI no longer adds by read-modify-PUT.
- A cold cache is not treated as an empty plan: `currentQueues()` returns
  `null` and the callers refuse to run (`useSetPlan.ts:42-50`).
- `PUT` rejects unknown track ids (`sets.py:589-600`).
- Every plan write holds the process-wide `_PLAN_WRITE_LOCK`, and append
  re-reads the row inside it (`sets.py:47-60`, `:620-632`).

The lock serialises writers. It cannot help a writer whose body was built from
a stale snapshot, and `PUT` is exactly that writer: it never reads the row
(`sets.py:567-606`). It takes the lock only to overwrite.

## The remaining hole

| Writer | Path | Loses others' picks? |
|---|---|---|
| Grid ＋ / `enter` (`RoleColumn.tsx:323,361,379`, `RoleColumnsGrid.tsx:85`) | `POST /plan/append` | no |
| ⌘K (`CommandBar.tsx:504,577`) | `POST /plan/append` | no |
| Grid × (`RoleColumn.tsx:531` passes the row **index**) | `PUT /plan` built from the tab's cache (`useSetPlan.ts:65-72`) | **yes** |
| `moveInRole` (`useSetPlan.ts:74-83`) | `PUT /plan` | yes, but nothing calls it |
| dance-mcp sidecar, a second tab, `curl` | append only today | no |

The × path is the only `PUT` caller. Its body comes from the react-query cache,
and nothing refreshes that cache when *another* writer changes the row.
`refetchOnWindowFocus` is off app-wide (`main.tsx:11`; the TanStack default is
on, per the [docs](https://tanstack.com/query/v5/docs/framework/react/guides/window-focus-refetching)).
`useSetPlan` has no `refetchInterval` (`useSetPlan.ts:13-18`). `useActiveSet`
does poll every 5 s, with the comment "other tabs / ⌘K add to the plan"
(`useSets.ts:30-31`), but it refreshes `["sets","active"]`, not
`["sets",id,"plan"]`, so the poll never reaches the grid. The cache refreshes
only on mount after 5 s (`staleTime`, `useSetPlan.ts:17`), on network
reconnect, or after this tab's own writes (`useSetPlan.ts:31-32`, `:96-97`).

So a Set view left open while an agent or a second tab appends will erase
every one of those picks the next time its user clicks ×. The other writer got
a 200. The clicking tab writes the PUT response into its cache, so its user
never sees the lost pick. Nobody gets an error.

The same mechanism also bites within one tab: two × clicks inside one
round-trip both read the same snapshot, and the second PUT puts back the
first-removed pick.

### Measured

Scratch probe, not committed. It served the real `create_app()` from this
worktree under uvicorn on `127.0.0.1:8771`, against a throwaway `HOME`/data
dir/DB with a socket-free bridge stub. It drove the endpoints over TCP with
httpx. It also mounted two throwaway prototype routes next to the real ones:
option C's remove op and option A′'s hash-guarded PUT, both under the real
`_PLAN_WRITE_LOCK`.

```
$ export PYTHONPATH=$PWD/src   # this worktree
$ /Users/arya/git/dance/.venv/bin/python $SCRATCHPAD/planrev/probe.py 8771
[1] agent POST /plan/append 4 -> 200 song = [1, 2, 3, 4]
[1] tab   PUT  /plan (x on index 0, from cache) -> 200 song = [2, 3]
[1] final song = [2, 3] | track 4 lost: True
[1] /sets/active updated_at changed on the agent append: True
[2] intended drums = [3]; got [1, 3] (track 1 resurrected)
[3] today_put        append lost  75/100   409s   0/100
[3] C_remove_op      append lost   0/100   409s   0/100
[3] Aprime_hash_put  append lost   0/100   409s  16/100
[4] GET /plan, 40 picks, 200 calls: median 1.80 ms, p95 2.21 ms
[5] PUT /plan with Content-Type text/plain -> 422 {"detail":[{"type":"model_attributes_type",...
```

Row [3] releases `append(9)` and "remove track 1" together through a
`threading.Barrier`, 100 fresh sets per row. A second run gave 76 / 0 / 0
losses (16 conflicts). Row [3] is the worst case. The common case is row [1]:
a cache that is minutes old, where today's path loses the pick every time.

## Options

### A. Revision counter + `expected_revision` on PUT → 409

`sets.plan_revision INTEGER NOT NULL DEFAULT 0`, bumped by every plan write
inside the lock. `SetPlanOut.revision` carries it. `PUT` requires
`expected_revision` and answers `409 {"detail": {"error": "plan_changed",
"plan": <SetPlanOut>}}` on a mismatch. The client seeds its cache from the
409 body, re-applies its intent, and retries once.

- **For:** general. It protects any whole-plan edit, including future bulk
  replace, undo and import.
- **Against:** a migration that nothing applies automatically. It needs an
  Alembic revision on head `e4a91c7b2d05` using `batch_alter_table` like
  `b7d4e2f1a9c3_add_sets_plan.py:34`. `init_db` only runs `create_all`
  (`database.py:837-841`), which never adds a column to an existing table.
  Checked in a scratch DB: once the column is mapped, an un-migrated DB fails
  every `Set` query with `sqlite3.OperationalError: no such column:
  sets.plan_revision`. Both the Booth and the Set view would break until the
  user runs `alembic upgrade head`. (`dance doctor` would flag it:
  `doctor.py:231`.)
- **Against:** every write path must remember to bump the counter. A missed
  bump silently re-opens the hole.
- **Against:** for the only edit the UI makes, "re-apply the intent" means
  "remove track X from role R". That is option C, done on the client instead of
  once under the lock the server already holds.

**A′, the same thing without a column:** make the revision a content hash,
`sha256((encode_plan(parse_plan(s.plan)) or "").encode()).hexdigest()[:16]`,
computed in `_plan_out` (`sets.py:521`). `encode_plan` emits roles in fixed
`PLAN_ROLES` order and returns `None` for an empty plan (`set_queues.py:53`),
so the hash is deterministic. This needs no migration and has no bump to
forget. The ABA case (the plan changes and then changes back) is harmless for a
whole-value precondition: if the content equals what the client edited, its
replacement is exactly what it meant. Row [3] above is this variant: 0 losses,
with 16% of the racing PUTs refused.

### B. ETag / If-Match → 412

The same token (A or A′) travels as a strong `ETag` on plan responses, and
`PUT` sends `If-Match`. A mismatch returns 412 Precondition Failed
([RFC 9110 §13.1.1](https://www.rfc-editor.org/rfc/rfc9110.html#name-if-match),
[§15.5.13](https://www.rfc-editor.org/rfc/rfc9110.html#name-412-precondition-failed)).
A missing header can return 428
([RFC 6585 §3](https://www.rfc-editor.org/rfc/rfc6585.html#section-3)).

- **For:** HTTP-native, and the semantics are the same as A.
- **Against:** it fits this client poorly.
  - `request()` throws response headers away (`api.ts:42-66`), so reading the
    `ETag` needs a second helper.
  - `request()` also drops `Content-Type` whenever a caller passes
    `init.headers`, because `...init` comes after the merged `headers` object
    (`api.ts:47-52`). Checked in node:
    `{headers:{"Content-Type":…,...init.headers},...init}.headers` →
    `{"If-Match":"\"abc\""}`. With fetch's default `text/plain` body type,
    FastAPI 0.103.2 answers 422 (row [5]). That latent bug is worth fixing
    regardless.
  - Every plan endpoint would need a `Response` parameter to set the header.
  - The token would be invisible in the OpenAPI schema and to an MCP tool's
    JSON result.
- Nothing B does is unavailable to A′ as a body field.

### C. Server-side ops only (add / remove / move), no client snapshot

Add `POST /sets/{id}/plan/remove` `{role, track_id}`, the twin of `append`
(`sets.py:609-632`): take the lock, re-read the row, filter, commit. Add
`/plan/move` `{role, track_id, to_index}` when a reorder gesture actually
ships. The UI stops sending whole plans.

- **For:** it closes the hole for every writer that exists, with no schema
  change: row [3] lost 0/100 and refused nothing. The double-× case resolves
  correctly ("remove 1", "remove 2" → `[3]`), and so does the agent case
  (remove commutes with append).
- **For:** ops key on `track_id`, not on an index into a possibly stale list.
  Within a role, ids are unique in practice: `append` dedupes (`sets.py:626`)
  and the UI never duplicates.
- **For:** the dance-mcp tools are already "append, never PUT".
- **Against:** it protects only the ops it defines. `PUT` still exists for
  scripts and future bulk edits, and on its own it stays a blind overwrite.
- **Against:** the UI still *displays* a stale plan until it refetches. Nothing
  is lost any more, but a pick appended elsewhere doesn't show up.

### D. Last-write-wins + WS `plan_changed` broadcast + refetch

After each plan commit, broadcast
`{"type": "plan_changed", "set_id": 7, "revision": "…", "op": "append"}`. The
client invalidates `["sets", set_id, "plan"]` when `revision` differs from its
cache.

- **The channel must be new.** The `/ws` reducer replaces the whole
  `AbletonState` with each frame (`useAbletonState.ts:11-15`), so a plan event
  there would blank tempo and playing clips. That leaves two choices:
  - A `/ws/sets` channel. It would be one more `WSManager` (the
    `app.py:71-72,109-110` pattern, `broadcast_threadsafe` from the sync
    endpoint thread, `ws.py:66-72`) and a `createSharedSocket` consumer. Vite
    already proxies the `/ws` prefix (`vite.config.ts:17`), and the origin
    guard covers every websocket scope (`origin_guard.py:119`).
  - Piggybacking a counter on `AbletonState`, the way `deck_map_revision`
    drives `useDeckMap` (`bridge.py:60-64`, `useDeckMap.ts:14-36`). That puts
    DB plan state into the OSC bridge's state object.
- **For:** near-instant freshness in every open tab.
- **Against:** it is still last-write-wins. It shrinks the staleness window
  while the socket is up, but row [3] shows that even a few milliseconds of
  overlap loses most races. A missed frame leaves the window unbounded again:
  the socket reconnects after 2 s with no replay (`sharedSocket.ts:44`,
  `:166-170`), so the client must also refetch on every reconnect.
- **Against:** about 100 lines across backend, frontend and tests, to take
  freshness from ≤5 s (a one-line `refetchInterval`, below) to sub-second.
  Plan prep doesn't need that.

### Summary

| | Stops loss: stale tab | Stops loss: race | Same-tab double × | Schema / Alembic | Client work |
|---|---|---|---|---|---|
| A (counter) | yes, via 409 | yes | 409 then retry | **column + migration** | rebase-and-retry |
| A′ (hash) | yes, via 409 | yes | 409 then retry | none | rebase-and-retry |
| B (If-Match) | as A/A′ | as A/A′ | as A/A′ | as A/A′ | header plumbing + `api.ts` fix |
| C (ops) | **yes, no error** | **yes** | **correct** | none | smaller than today |
| D (WS) | narrows it | no | no | none | new socket consumer |

## Recommendation

**C for the UI, A′ on the PUT that remains, and a 5 s plan poll for
freshness. No D for now.**

1. **`POST /sets/{id}/plan/remove`** `{role, track_id}` → `SetPlanOut`.
   - It removes every occurrence of `track_id` from that role, under
     `_PLAN_WRITE_LOCK`, re-reading inside it (copy `sets.py:620-632`).
   - Absent track → 200, unchanged (idempotent, mirroring append's no-op).
   - Bad role → 400; unknown set → 404.
2. **`SetPlanOut.revision: str`** (the A′ hash), returned by GET, PUT, append
   and remove. This is free for agents too: an MCP `get_plan` can quote it.
3. **`SetPlanUpdateRequest.expected_revision: str`, required.**
   - `put_plan` moves its `_require_set` + `session.refresh` inside the lock and
     compares before writing. Checking outside the lock would re-open the race.
   - A mismatch returns `409 {"detail": {"error": "plan_changed", "plan":
     <current SetPlanOut>}}`.
   - After step 4 the UI never calls PUT. Requiring the field costs only the two
     existing PUT tests (`test_api.py:3602`, `:3652`) and turns any future
     script's blind overwrite into a 422 instead of silent data loss.
4. **Client** (`companion-app/`):
   - `api.ts`: add `removeFromPlan(setId, role, trackId)`. Delete `putPlan`
     (no caller left), or give it an `expectedRevision` argument if you keep it.
   - `useSetPlan.ts`: `removeFromRole(role, trackId)` becomes a mutation on
     `removeFromPlan` with the same `onSuccess` as `useAppendToPlan`
     (`:95-98`). Delete `put`, `currentQueues` and the uncalled `moveInRole`.
     Add `refetchInterval: 5_000` to `useSetPlan`, matching `useActiveSet`.
     Row [4] puts one such GET at ~1.8 ms round-trip on localhost with 40
     picks.
     react-query runs one interval per query key, not one per role column.
   - `RoleColumn.tsx:531`: `removeFromRole(role, item.track_id)`.
   - `types.ts`: add `SetPlan.revision` and
     `SetPlanUpdateRequest.expected_revision`.
5. **`/plan/move`** waits for a reorder gesture. It is "a new endpoint following
   an existing pattern" under rule 1, so it won't need another proposal.

WS: no change. Schema: no change. Alembic: no revision.

## Decisions for the user

1. **C + A′ (recommended)**, A with a real counter column, or D on its own.
2. **PUT's future.** Options: `expected_revision` required (recommended);
   optional (old scripts keep working, but stay a footgun); or delete `PUT
   /plan` outright, since nothing in the UI or the MCP proposal needs it after
   this change.
3. **Remove semantics:** idempotent 200 when the track is absent (recommended),
   or 404. With 404, a double-click shows an error toast for the second click.
4. **Freshness:** `refetchInterval: 5_000` on the plan (recommended, one line).
   Alternatives: invalidate the plan when the polled `active.updated_at`
   changes (zero extra requests, but rename/activate also bump it:
   `sets.py:231,256`, `database.py:699`), or D's WebSocket.
5. **Should `PUT` also reject duplicate ids within a role** (400)? It accepts
   them today (`sets.py:586-587`). Rejecting them makes "remove by id"
   exactly match what the user sees. Recommended, though no current client
   sends duplicates.

## Migration cost

- **Backend:** `sets.py` ≈ +40 lines (remove endpoint, hash helper, guarded
  PUT, `revision` in `_plan_out`). `schemas.py` ≈ +12 (`SetPlanRemoveRequest`,
  two fields).
- **Frontend:** `api.ts` ≈ +10/−10, `useSetPlan.ts` ≈ +15/−35,
  `RoleColumn.tsx` 1 line, `types.ts` 2 fields. Two test mocks return a
  `put` member (`RoleColumnsGrid.test.tsx:21`, `gridKeys.test.tsx:33`); drop
  it.
- **Data:** none. Existing `sets.plan` values hash as they are, and no
  `alembic upgrade` is needed.
- **Docs:** `docs/api.md:107-120` (add the endpoint row, note
  `revision`/`expected_revision`, drop "the frontend edits locally and PUTs").
  Update the docstrings at `useSetPlan.ts:1-4` and `schemas.py:395-399`.
- **Tests** (new file `tests/test_set_plan_revision.py`, about 150 lines):
  - remove removes, and repeats as a 200 no-op;
  - remove and append released by a `threading.Barrier`, 20 rounds, append
    never lost;
  - row [1] replayed through PUT with the stale revision → 409, the body's plan
    contains the agent's pick, and the DB is unchanged;
  - PUT with the current revision → 200 and a new revision;
  - the revision is stable across GETs, changes on every write, is equal for
    equal content (ABA), and is constant for the empty plan;
  - PUT without `expected_revision` → 422.

  Plus one vitest (`companion-app/tests/planMutations.test.tsx`): × calls
  `removeFromPlan` with the track id, never `putPlan`.

## Verification plan

1. Run the new and the old plan tests, then the whole suite:
   `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q tests/test_set_plan_revision.py`,
   then
   `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q tests/test_api.py -k plan`,
   then the full suite.
   In `companion-app/`, run `npx tsc -p tsconfig.json && npx vitest run`.
2. Re-run the scratch probe against the real routes (`/plan/remove`, and the
   guarded `/plan`). Expected:
   - [1] the stale PUT gets 409 and track 4 survives;
   - [2] two removes give `[3]`;
   - [3] 0/100 lost on the remove-op row.
3. Real UI, throwaway backend. Serve with the probe's launcher (stub bridge,
   scratch `HOME`, port 8765), not `uvicorn dance.api:create_app`, which would
   read `~/.dance/.env` and bind the OSC listen port (`bridge.py:106`) that the
   real backend holds. Then run `DANCE_API_TARGET=http://127.0.0.1:8765
   PORT=5174 npx vite` and open the Set view in two windows.
   1. `curl -X POST …/plan/append` a pick. Within 5 s it shows in both windows.
   2. Click × on a different pick in window 1. The curl pick survives.
   3. Double-click × across two picks. Both go.
   4. `curl -X PUT` with an old revision → 409.

## Not in scope

- Per-pick provenance ("Claude added this") needs a plan-shape change. That is
  its own proposal (dance-mcp decision 6).
- Multi-worker uvicorn. `_PLAN_WRITE_LOCK` is per process (`sets.py:57-59`).
  A′'s check happens under that lock, so it inherits the same limit.

## Prior art

[gakonst/djbooth](https://github.com/gakonst/djbooth) (MIT, `apps/djbooth`,
read at `452c46f`; shape only, no code taken) edits playlists with ops keyed
by a stable membership id: `add-track`, `remove-track`, `move-track`
(`src/shared/types.ts:260-267`). Its one whole-list op carries an
`expectedMembershipIds` precondition (`:268-273`, checked at
`src/server/library.ts:1325-1332`) and maps a mismatch to HTTP 409
(`src/server/server.ts:1254-1255`). That is C plus A′.
