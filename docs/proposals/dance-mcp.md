# dance-mcp: Claude helps prep a set, never plays one

**Status:** proposal — not implemented
**Date:** 2026-10-13
**Touches:** a new top-level `dance-mcp/` package with its own venv, one contract
test in `tests/`, `.gitignore`, docs. No route, request/response schema, model,
Stage, OSC message or `.als` change, so the tools themselves don't trigger
CLAUDE.md rule 1.
**Depends on:** the dispatcher lock (merged in `0902330`, done); plan edits
that can't drop picks ([`set-plan-revision.md`](set-plan-revision.md), on
branch `claude/proposal-set-plan-revision`, not merged); a short tagger fix
(see Prerequisites).

---

## The problem

An assistant is useful at prep time. Requests like *"find rolling-bass tracks
near 124 I haven't queued"*, *"put three in the bass column of tonight's set"*
or *"ingest this playlist and tell me when it's analysed"* all fit. Today
Claude Code can do them only by hand-writing `curl` against the API. Claude
Desktop can't do them at all.

[`vision.md:48`](../vision.md) draws the line: *"Not a 'DJ AI' that decides for
you."* So the boundary is the design. The agent works **at prep time only**.
It reads freely. It adds to sets and the library behind permission prompts.
It never touches the transport, the mixer or a delete. Every pick it makes
lands in the plan, where the user sees it and can remove it.

## What was measured

I built `create_app()` from this worktree and served it with uvicorn on
`127.0.0.1:8765`. It ran against a throwaway `HOME`, data dir and DB, with a
socket-free stub bridge reporting one deck clip playing. I drove it with
httpx, the way a sidecar would. The scratch script isn't committed. It never
calls the ingest routes (they spawn yt-dlp). For the 409 it holds the
dispatcher lock itself, so `/process` can't start a run.

```
[1] total (method, path) pairs: 90  (ableton 34, sets 15, pipeline 12, tracks 9, sessions 5, recommend 4, …)
    DELETE: tracks/{id}, sets/{id}, sets/{id}/tracks/{tid}, ableton/decks/cell/{t}/{s}
    proposed routes: 27; missing from app: []
[2] GET /health (httpx, no Origin): 200
    same, Origin: https://evil.example: 403   same, Host: dance.example: 403
    httpx default request headers: ['accept', 'accept-encoding', 'connection', 'user-agent']
[3] /tracks/search keys: [analysis, artist, duration_seconds, file_path, id, state, tags, title]
      file_path: /private/tmp/…/probe-home/library/Layton Giordani - Manifesto.mp3
    /health/deps details: ('yt_dlp', '/Users/arya/git/dance/.venv/bin/yt-dlp'), ('ffmpeg', '/opt/homebrew/bin/ffmpeg'), …
[4] agent append -> bass = [1]; again (idempotent) -> [1]
    UI PUT from its cached plan -> bass = []
    GET plan latency over TCP: median 1.0 ms, max 1.2 ms
    list_sets track_count after a plan append: 0
[5] /ableton/state playing_clips: {'5': 0}   /ableton/decks columns: {'drums_a': 5, 'drums_b': 6}
[6] POST /pipeline/process while another run holds the lock: 409 "Another pipeline run is already in
      progress (pid 73955, `probe holding the lock`, …) … (lock file: /private/tmp/…/data/dispatcher.lock)."
[7] created 60 jobs; GET first -> 404 ; GET last -> 200 ; GET /jobs?limit=100 -> 50
```

What the probe shows:

1. **The API already covers prep.** All 27 routes the tools need exist today.
2. **A sidecar passes the Origin guard as it is.** httpx sends no `Origin` and
   a loopback `Host`, which the guard admits by design (`origin_guard.py:26`
   names "a future MCP server"). So the sidecar must never set `Origin`. It
   must also refuse to start if `DANCE_API_URL` isn't loopback.
3. **Results and errors carry absolute paths:** `TrackOut.file_path`
   (`schemas.py:120`), `RecommendationOut.file_path` (`:140`),
   `StemFileOut.path` (`:72`), `PreviewResult.audio_path` (`:214`),
   `AlsExportResult.out_path` (`:583`), the `/health/deps` details
   (`app.py:186`), and the dispatcher-lock 409 (`dispatcher_lock.py:100-110`).
   The library paths contain spaces (`Layton Giordani - Manifesto.mp3`).
4. **A second writer erases picks.** The Set view rebuilds the whole plan from
   its react-query cache and PUTs it on every remove (`useSetPlan.ts:65-72`).
   That cache doesn't poll (`useSetPlan.ts:13-18`), and refetch on focus is
   off (`main.tsx:11`). So the user's next × erases an agent's
   `plan/append`. [`set-plan-revision.md`](set-plan-revision.md) measures the
   race in full.
5. **Job history keeps 50** (`jobs.py:106`). `/ingest/playlist` creates one
   job per *new* track, so a playlist that adds more than 50 evicts its own
   early job ids.
6. **`GET /sets` `track_count` counts legacy `set_tracks` rows**
   (`sets.py:182`), not plan picks. It reads 0 for every plan-only set.
7. **`POST /tracks/{id}/tag` loads its own CLAP on every call.**
   `tracks.py:213` builds `ClapZeroShotTagger(settings)` with no encoder, so
   `_ensure_encoder` loads `EmbeddingStage` from scratch (`tagger.py:188-196`).
   That's `laion/clap-htsat-unfused`, 1.1 GB in `~/.cache/huggingface`, loaded
   outside `GPU_SEMAPHORE`.

### Packaging: `mcp` can't join the dance venv as pinned

spotdl 4.4.3 (installed) and 4.5.2 (latest) both pin `fastapi>=0.103,<0.104`
and `uvicorn>=0.23.2,<0.24`, and `pyproject.toml:54-55` mirrors that pin. I
read the PyPI metadata for all 73 non-yanked `mcp` releases (scratch script):

| | requires | dance `.venv` has |
|---|---|---|
| fastapi 0.103.1, 0.103.2 | `anyio>=3.7.1,<4.0.0` | fastapi 0.103.2, **anyio 3.7.1** |
| `mcp` 0.9.1 → 2.3.0, every release | `anyio>=4.5` (0.9.1–1.1.0: `>=4.6`; 2.x: `>=4.9`) | — |
| `mcp` 1.2.0–1.2.1 / 1.13.0 → 1.30.0 and all 2.x | `uvicorn>=0.30` / `uvicorn>=0.31.1` | uvicorn 0.23.2 |
| `mcp` 2.0.0 → 2.3.0 | `pydantic>=2.12`, `starlette>=0.27` (py<3.14), `httpx2>=2.5` | pydantic 2.12.5, starlette 0.27.0 |

I asked pip's resolver for the combination without installing anything:

```
.venv/bin/python -m pip install --dry-run --ignore-installed --target <scratch> --report <json> \
  "spotdl==4.4.3" "fastapi>=0.103,<0.104" "uvicorn>=0.23,<0.24" "httpx>=0.27,<0.28" \
  "pydantic-settings>=2.7" "mcp"
→ mcp 1.12.4, fastapi 0.103.0, anyio 4.15.1, starlette 0.27.0, uvicorn 0.23.2, sse-starlette 3.0.3
```

So "pin an older mcp" has one solution, and it **downgrades the backend** to
fastapi 0.103.0 on anyio 4. 0.103.0 lacks the anyio pin only because it
shipped on 2023-08-26, four days before anyio 4.0.0 (2023-08-30). The next
release, 0.103.1 (2023-09-02), added *"Pin AnyIO to < 4.0.0 to handle an
incompatibility"* ([release notes](https://github.com/fastapi/fastapi/releases/tag/0.103.1)).
That combination is untested here. It also freezes the SDK at 1.12.4
(2025-08-07). That's 29 releases behind the maintained 1.x line (1.30.0,
2026-09-07) and a major version behind 2.3.0 (2026-10-02).

## Options

### A. A stdio sidecar that calls the running backend over HTTP

`dance-mcp` is a separate process that Claude spawns over stdio. It imports
nothing from `dance`. Each tool makes one or two calls to existing routes on
`DANCE_API_URL` (default `http://127.0.0.1:8000`).

- **For:** it shares everything that lives in the backend process. That
  covers the job registry (`jobs.py:274`), the plan write lock
  (`sets.py:60`), the lazy CLAP text encoder (`app.py:112`) and the OSC
  bridge. Jobs the agent starts reach the UI through `/ws/pipeline`
  (`ws.py:138`, `usePipelineJobs.ts`). A set switch shows within 5 s
  (`useSets.ts:31`). With no dance imports, the venv split costs nothing.
- **Against:** it needs the backend running, so the connection error must say
  how to start it. It also inherits the gaps in findings 3–7.

### B. An in-process server that imports `dance` and opens the DB

- It can't be installed (see the table above).
- If it could, it would bypass the process-local plan lock and job registry.
  It would load a second 1.1 GB CLAP on a 16 GB Mac that is already running
  Live and Demucs, and the UI would never see its jobs. **Rejected.**

### C. A Streamable-HTTP `/mcp` endpoint mounted in the FastAPI app

- Same install wall as B.
- It would put write tools on the port the browser talks to. **Rejected.**

### D. No server: a project skill that teaches Claude Code to `curl` the API

- **For:** no packaging at all.
- **Against:** no Claude Desktop, and no path stripping. Nothing
  structurally excludes the 34 `/ableton/*` routes: `curl -X POST
  …/ableton/transport/stop-all` is one approval away. Approval rules are Bash
  prefixes, not per-tool. Worth having *alongside* A as the workflow layer
  (see Wiring), not instead of it.

## Recommendation: A, in its own venv, with a hand-written allowlist

### Tool surface

The tool names are the allowlist. Nothing is generated from OpenAPI, because
the exclusions are the safety property. (djbooth derives its MCP tools from a
CLI command map filtered by include globs, `packages/djbooth/src/cli.ts:30`.
Here, a glob like `ableton_*` would sweep in the transport.)

**Read** (`readOnlyHint: true`; `openWorldHint: false` unless marked):

| tool | route(s) | returns (projected, no paths) |
|---|---|---|
| `status` | `GET /health`, `GET /api/v1/pipeline/status`, `GET /api/v1/health/deps`, `GET /api/v1/pipeline/recent?state=error&limit=10` | state counts, `weighted_progress`, dep **statuses** (not `detail`), recent errors |
| `search_tracks` | `GET /api/v1/tracks/search` (`tracks.py:57`) | id, title, artist, bpm, key, energy, state, tags |
| `spotify_search` (open world) | `GET /api/v1/spotify/search` (`spotify.py:24`) | spotify_id, title, artist, duration. `ingest_track` needs the id |
| `vibe_search` | `POST /api/v1/recommend/text` (`recommend.py:128`) | as `search_tracks`, plus score. The first call loads CLAP, ~12–30 s on MPS (`recommend.py:110`), so this call's HTTP timeout is 60 s |
| `similar_tracks` | `GET /api/v1/recommend/by-seed/{track_id}` (`recommend.py:76`) | as above, plus reasons |
| `get_track` | `GET /api/v1/tracks/{track_id}` + `…/regions` (`tracks.py:152`, `:163`) | analysis and sections. The route has no "track-level only" filter, so the tool drops rows that have a `stem_file_id` |
| `list_sets` | `GET /api/v1/sets` | id, name, is_active, updated_at. Not `track_count` (finding 6) |
| `get_plan` | `GET /api/v1/sets/{set_id}/plan`; with no `set_id`, `GET /api/v1/sets/active` first | queues by role |
| `plan_recs` | `GET /api/v1/sets/{set_id}/plan-recs?role=` (`sets.py:635`) | recs with score_breakdown |
| `live_state` | `GET /api/v1/ableton/state`, `GET /api/v1/ableton/decks` | tempo, is_playing, what's loaded where. No meters or playheads |
| `wait_job` | polls `GET /api/v1/pipeline/jobs/{job_id}` every 1 s, `timeout_s ≤ 25` | job snapshot and `done`. For a `pipeline_run` it adds `/pipeline/status` counts, because every stage item just says "running…" (`pipeline.py:922`) |

**Write** (permission prompt on every call; `readOnlyHint: false`).
`destructiveHint` is set explicitly on every tool because the spec defaults
it to `true` ([schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-11-25/schema.ts)):

| tool | route | destructive / idempotent / open world | notes |
|---|---|---|---|
| `create_set` | `POST /api/v1/sets` | F / F / F | created inactive (`sets.py:207`) |
| `activate_set` | `POST /api/v1/sets/{set_id}/activate` | **T** / T / F | it deactivates the other set, so it isn't additive |
| `plan_append` | `POST /api/v1/sets/{set_id}/plan/append`. **Never** `PUT /plan` | F / T / F | no-op if already queued (`sets.py:626`; probe [4]) |
| `ingest_track` | `POST /api/v1/pipeline/ingest/track` | F / T / **T** | idempotent on `spotify_id` (`pipeline.py:465`) |
| `ingest_playlist` | `POST /api/v1/pipeline/ingest/playlist` | F / T / **T** | **no `user_token` argument**, because tool arguments land in transcripts. The route falls back to `DANCE_SPOTIFY_USER_TOKEN` |
| `run_pipeline` | `POST /api/v1/pipeline/process` → job id | F / F / F | runs *every* pending track, not only the agent's. Demucs is ~6.9 min/track (16.7 h for 145 tracks, [`library-duplicates.md`](library-duplicates.md)) |
| `export_als` | `POST /api/v1/tracks/{track_id}/als` | **T** / T / F | **no `out_path`**. It overwrites `<als_output_dir>/<title> - <artist>.als` (`generator.py:66-71`), including edits saved there from Live |
| `tag_track` | `POST /api/v1/tracks/{track_id}/tag`, CLAP only, never `deep` | **T** / T / F | replaces `inferred` tags. Ships after prerequisite 3 |

`run_pipeline`, `ingest_playlist` and `export_als` also set
`_meta["anthropic/requiresUserInteraction"]: true`. With that flag, Claude Code
prompts on every call, even in `acceptEdits`, `auto` and `bypassPermissions`.
It offers no "don't ask again", allow rules don't skip it, and `dontAsk`
denies the call ([docs](https://code.claude.com/docs/en/mcp#require-approval-for-a-specific-tool)).
The flag doesn't bind an Agent SDK host: its `canUseTool` callback receives the
call and may approve it. The other writes keep normal prompts, so building a
plan isn't twenty dialogs.

**Opt-in, off by default** (`DANCE_MCP_ALLOW_PREVIEW=1`): `preview` and
`preview_stop` call `POST /api/v1/ableton/preview` and `…/preview/stop`
(`ableton.py:831`, `:871`). Audio goes to the Cue track, which is routed to
the headphones on outs 3/4, so the master is untouched.

**Not wrapped: 65 of the 90 pairs** (63 with preview on):

- the preview pair, unless it's switched on;
- the other 30 `/ableton/*` routes: transport, fire/stop/seek/solo, tempo,
  volume, crossfader, FX (including `GET /fx/state`), PFL, record,
  `load-track`, warp-check, and decks reset/clean/resync/`DELETE cell`;
- the other three `DELETE`s;
- `PUT /plan`, both `PATCH`es, the legacy `/sets/{id}` and `/sets/{id}/tracks*`
  routes, and `tail-recs`;
- `/sessions/*` (5 routes, performance history);
- the Booth's live `POST /recommend` and `/recommend/by-column`;
- `/files/reveal`, `/pipeline/watch` (both methods), `/pipeline/scan`,
  `GET /pipeline/jobs`, and the CSV ingest pair;
- `GET /tracks`, `/tracks/{id}/stems` (it returns paths), and both waveforms;
- the 4 OpenAPI/docs routes and the 2 WebSockets.

### The prep-only guard

Before running, every write tool (and `preview`) reads `/ableton/state` and
`/ableton/decks`. If `playing_clips` has an entry for any track index in the
deck map's `columns`, the tool refuses with this message: *"A deck is
playing, and dance-mcp only preps. Stop playback or finish the set, then ask
again."* The bridge removes an entry when its clip stops (`bridge.py:458-460`).
The Cue track isn't a deck column, so an active preview doesn't trip the
guard. If the backend hasn't seen a clip start (for example, Live is closed),
`playing_clips` is empty and writes are allowed. If Live quits mid-play, a
stale entry may remain. The refusal then points at `live_state` so the user
can see why. With `columns` null (no decks built yet), the guard falls back to
`is_playing`.

Why have a guard:

- the plan feeds the Booth's recs;
- `activate_set` swaps what the Booth shows within 5 s;
- nobody has measured Live's audio while a Demucs pass runs on this 16 GB M2
  Pro.

### Errors name the next step; nothing returns a path

| backend says | tool returns (`isError: true`) |
|---|---|
| connection refused | "dance backend isn't running at `$DANCE_API_URL`. Start it: `uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000`" |
| `403` from the Origin guard | "The backend refused the request (Origin/Host guard). `DANCE_API_URL` must be a loopback address." |
| `404 no active set` | "No active set. Call `list_sets` then `activate_set`, or `create_set`." |
| `400 role must be one of [...]` | passed through (it already lists the roles) |
| `422` missing `role` on `plan-recs` | "Pass `role`: drums, bass, vocals, other or song." |
| `400 … need 'complete'` / `… has no stems` (`generator.py:113`, `:135`) | + "Call `run_pipeline`, then `wait_job`." |
| `409` from `/process`, either variant (`pipeline.py:1133`, `:1142`) | "A pipeline run is already going (holder: pid and command, no lock path). Use `status`; don't start another." |
| `503` Spotify not configured / `401` user token | "Ask the user to add the credential to `~/.dance/.env` and restart the backend (settings load once, `deps.py:29`). Never paste it into chat." |
| `503 CLAP model unavailable` | + "`search_tracks` and `similar_tracks` work without it." |
| `404 job not found` for an id it handed out | "Job evicted from the 50-job history (`jobs.py:106`). Use `status` or `get_track`." |

**Results are projections.** Each tool names the fields it returns, so the
path fields in finding 3 never leave the sidecar. Free text (error details,
the 500-character log tail on a failed run, `pipeline.py:1052`) also goes
through a scrubber. It replaces anything that starts with `~/`, `/Users/`,
`/private/`, `/Volumes/`, `/opt/`, `/tmp/` or `/home/` with `<path>`, up to the
next quote, closing parenthesis or end of line. It doesn't stop at whitespace,
because the library's paths contain spaces.

The projections serve three purposes:

- transcripts stay free of the home layout;
- results stay far below Claude Code's 10k-token warning and 25k default cap
  ([docs](https://code.claude.com/docs/en/mcp));
- the agent gets no file to `Read` or move. Moving stems breaks every `.als`
  (CLAUDE.md, "Don't").

The server `instructions` say the scope in three lines: prep only; append,
never rewrite; ask before ingesting or processing. Claude Code truncates
instructions and each tool description at 2,048 characters, so the important
rule goes first. Logs go to stderr, because stdout carries the protocol.

### Long-running work

There's no long-poll route, so `wait_job` polls. A `GET` costs 1 ms on
localhost (probe [4]), so the polling is free. The 25 s cap keeps each call
short and gives the user a chance to interrupt between calls. Claude Code moves
a main-conversation MCP call to the background after 2 minutes, and the stdio
idle timeout is 30 minutes ([docs](https://code.claude.com/docs/en/mcp)).
A `run_pipeline` can last hours, so the instructions tell the agent to report
progress and hand back to the user rather than loop on `wait_job`.

### Packaging (precise)

**Recommendation: a separate venv, not an older mcp pinned into `.venv`.**
Pinning means swapping the whole backend to fastapi 0.103.0 + anyio 4, plus a
frozen 2025 SDK, for a sidecar that never imports the backend.

```
dance-mcp/                       # top level, next to companion-app/ (it has its own toolchain too)
  pyproject.toml                 # name="dance-mcp"; deps: mcp>=2.3,<3 (pulls httpx2>=2.10); [project.scripts] dance-mcp
  src/dance_mcp/server.py        # MCPServer(instructions=…) + 19 tools + _call/_scrub/_guard   (~350 lines)
  src/dance_mcp/routes.json      # the 27 (method, path) pairs, the source for the contract test
  tests/test_server.py
```

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv dance-mcp/.venv   # the interpreter .venv already uses
dance-mcp/.venv/bin/pip install -e './dance-mcp[dev]'
```

mcp 2.3.0's `MCPServer.tool()` takes `annotations=` and `meta=`, and its
constructor takes `instructions=` (read from the `v2.3.0` tag). `uv` and
`pipx` aren't installed here, and a stdlib venv needs nothing new.
`.gitignore:27` (`.venv/`) already covers `dance-mcp/.venv/`. The root pytest
`testpaths = ["tests"]` never collects `dance-mcp/tests`.

### Wiring for Claude Code and Desktop

**Register at local scope rather than in a checked-in `.mcp.json`.** Claude
Code loads project-scoped servers without the approval prompt in `claude -p`,
Agent SDK and cloud sessions
([docs](https://code.claude.com/docs/en/mcp#project-scope)). Each worktree has
its own copy of `.mcp.json`. This repo has 19 linked worktrees right now, and
the workflow agents in them run as Agent SDK sessions with permissions
bypassed (this proposal was drafted by one). A checked-in `.mcp.json` would
give every one of them `plan_append` and `activate_set` against the real
backend on :8000, which is exactly what those agents' rules forbid. Local
scope is stored in `~/.claude.json` under the project's path, so worktrees
don't get it. Run this from `/Users/arya/git/dance`:

```bash
claude mcp add --env DANCE_API_URL=http://127.0.0.1:8000 --transport stdio --scope local \
  dance -- /Users/arya/git/dance/dance-mcp/.venv/bin/dance-mcp
```

(`--env` can't sit right before the name, or the CLI parses the name as
another pair.) Claude Desktop gets the same command and `env` in
`~/Library/Application Support/Claude/claude_desktop_config.json`. Whether
Desktop honours `requiresUserInteraction` is unknown. Its own per-tool
approval still applies.

**Do narrow `.gitignore`**, so the workflow layer can be versioned:

- a skill, `.claude/skills/prep-set/SKILL.md`: search → `plan_recs` →
  `plan_append`; ingest → `wait_job`; never loop on a pipeline run;
- a `.claude/settings.json` that allow-lists the 11 read tools
  (`mcp__dance__search_tracks`, …).

Allow rules for a server that isn't registered do nothing, so worktree agents
are unaffected. I tested this with `git check-ignore -v --no-index` in this
worktree and then reverted the edit:

```
.claude/*                 # replaces .gitignore:62 ".claude/"
!.claude/skills/
!.claude/settings.json
```
```
.claude/skills/prep-set/SKILL.md → not ignored   (today: ignored by .gitignore:62)
.claude/settings.json           → not ignored   (today: ignored)
.claude/settings.local.json, .claude/launch.json, .claude/worktrees/… → still ignored
```

Plain `.claude/` followed by `!.claude/skills/` doesn't work: git can't
re-include a path under an excluded directory, and the check confirmed it.
`.mcp.json` isn't ignored today, so the checked-in variant needs no
`.gitignore` change.

### Tests

- **In-memory, in the sidecar's venv:** `async with Client(server)` from `mcp`
  ([testing docs](https://py.sdk.modelcontextprotocol.io/get-started/testing/)).
  djbooth does the same with `InMemoryTransport`
  (`packages/djbooth/test/mcp.test.ts:63-76`). The backend is an httpx2
  transport over canned responses. The responses deliberately include the
  probe's path with spaces in it. The tests assert:
  - the sorted tool-name list equals the 19 names exactly (18 until
    prerequisite 3), plus the 2 preview tools only under
    `DANCE_MCP_ALLOW_PREVIEW=1`;
  - every tool's annotations match the tables, and `requiresUserInteraction`
    is on exactly the 3 named tools;
  - no `inputSchema` has a `user_token`, `out_path` or `deep` property;
  - no result string contains a path;
  - every error row maps as written;
  - writes refuse while a deck clip plays;
  - `wait_job` clamps to 25 s.
- **Route contract, in the dance suite:** `tests/test_mcp_route_contract.py`
  reads `dance-mcp/src/dance_mcp/routes.json` and asserts that every pair
  exists in `create_app(...).routes`. It also asserts that none is a `DELETE`
  and that no `/ableton/` route outside state, decks and preview is listed. It
  needs no cross-venv import. The probe already ran this check: 27/27.

## Prerequisites

1. **Dispatcher lock: done.** It's on main (`0902330`, `d7059aa`).
   `/process` returns 409 whenever anyone holds the lock, terminal runs
   included. The dispatcher no longer claims `pending:` placeholder rows
   mid-download (`dispatcher.py:275`). That matters here because *ingest →
   run_pipeline* is exactly the sequence an agent will issue.
2. **Plan edits can't drop picks** ([`set-plan-revision.md`](set-plan-revision.md),
   its C + D-lite). Until that lands, the Set view doesn't show an agent's
   picks, and its next × erases them (probe [4]). The write tier waits for
   it. The read tier doesn't need it.
3. **`tag_track` shares the encoder.** At `tracks.py:213`, pass
   `text_encoder=_get_text_encoder(request, settings)` from `recommend.py`,
   which also gates on `GPU_SEMAPHORE`. That's a three-line change with no
   schema impact, so rule 1 doesn't apply. Until it lands, leave `tag_track`
   out.

## Decisions for the user

1. **Venv split** (recommended), or fastapi 0.103.0 + mcp 1.12.4 inside
   `.venv`?
2. **Local-scope registration** (recommended), or the checked-in `.mcp.json`
   from the original brief? The checked-in file exposes the tools to every
   headless worktree agent. Blocking that would need `disabledMcpjsonServers`
   in each worktree, or the server refusing to run outside the main checkout.
3. **Ship in tiers?** Recommended: read tools now, write tools once
   prerequisite 2 has merged.
4. **Which writes always prompt?** Proposed: `run_pipeline`,
   `ingest_playlist`, `export_als`. Making all writes prompt adds friction to
   plan-building.
5. **The prep-only guard** (refuse writes while a deck clip plays):
   recommended, or no guard?
6. **Preview:** ship the env-gated pair, or drop it from v1?
7. **After set-plan-revision's C lands:** add `plan_remove` (so the user can
   undo an agent's pick by asking) and `plan_move`, or stay append-only?

## Migration cost

- New: `dance-mcp/` (~350 lines of server, ~250 of tests) and a ~40-line
  contract test.
- Changed: `.gitignore` (one line becomes three), a `docs/dev.md` section, and
  a CLAUDE.md "When the user says X" row ("help me prep" → the dance tools).
- Backend: nothing for the tools themselves. Prerequisite 3 is three lines.
- Reverting: `claude mcp remove dance` and delete the directory.

## Verification plan

1. **Unit:** run `dance-mcp/.venv/bin/python -m pytest dance-mcp/tests -q`,
   and `PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q tests/test_mcp_route_contract.py`.
2. **Stdio against a throwaway backend.** Commit the probe harness with the
   implementation: uvicorn on :8765, `HOME` and `DANCE_*` in a temp dir, stub
   bridge.
   - Spawn `dance-mcp` with the SDK's stdio client and run `create_set` →
     `activate_set` → `plan_append` ×3 → `get_plan`.
   - With `DANCE_API_TARGET=http://127.0.0.1:8765 npx vite --port 5174` open,
     the picks should appear within 5 s (D-lite), and × on another pick must
     leave them in place (C).
3. **Real rig, real weights (rule 2)**, against the user's backend:
   - `vibe_search("rolling bassline")`, timing the cold CLAP load;
   - `spotify_search` → `ingest_track` → `wait_job` → `run_pipeline` →
     `wait_job` until the track is `complete`;
   - `export_als`, then open the file in Live: 5 tracks load, audio plays,
     tempo is right;
   - with a deck clip playing, every write tool must refuse;
   - with preview on, check whether firing the Cue clip starts Live's
     transport (unverified), and that `preview_stop` leaves Live as it was.

   Record each result's size from the transcript.
4. **Wiring:**
   - `claude mcp list` lists `dance` in `/Users/arya/git/dance` and not in a
     worktree.
   - In Claude Code, ask *"plan a 20-minute warm-up in a new set"*. Reads run
     without prompts, every write prompts, and `run_pipeline` prompts again
     even after "don't ask again" on another tool.
