# dance-mcp — Claude helps prep a set, never plays one

**Status:** proposal — not implemented
**Date:** 2026-10-13
**Touches:** a new `tools/dance-mcp/` package with its own venv, `.gitignore`, docs.
No route, request/response schema, model, Stage, OSC or `.als` change.
**Depends on:** the cross-process dispatcher lock (branch `claude/dispatcher-lock`,
`2a64326` + `1530dc6`) and plan revision checks
([`set-plan-revision.md`](set-plan-revision.md)).

---

## The problem

Prep is where an assistant helps: *"find rolling-bass tracks near 124 that I
haven't queued"*, *"put three of them in the bass column of tonight's set"*,
*"ingest this playlist and tell me when it's analysed"*. Today Claude Code can
only do that by hand-writing `curl` against the API, and Claude Desktop can't
at all.

[`vision.md`](../vision.md) draws the line: *"Not a 'DJ AI' that decides for
you. It surfaces possibilities; the user decides."* So the boundary is the
design. The agent works at **prep time only**. It reads freely, adds to things
behind an approval prompt, and never touches the transport, the mixer or a
delete.

## What was measured

**The API already covers prep.** I built `create_app()` from this worktree, served it
over real TCP (uvicorn, `127.0.0.1:8765`) against a throwaway data dir and DB
with a socket-free bridge stub, and drove it with httpx the way a sidecar
would (scratch script, not committed):

```
total routes: 90   (ableton 34, sets 15, pipeline 12, tracks 9, sessions 5, recommend 4, …)
proposed routes: 25; missing from app: []
DELETE routes (all excluded): ableton/decks/cell/{t}/{s}, sets/{id}, sets/{id}/tracks/{tid}, tracks/{id}
```

**Three things the probe showed that shape the design:**

```
GET /tracks/search → keys [analysis, artist, duration_seconds, file_path, id, state, tags, title]
   file_path value leaked: /private/tmp/…/library/Navi.mp3
GET /health/deps   → ('yt_dlp', '/Users/arya/git/dance/.venv/bin/yt-dlp'), ('ffmpeg', '/opt/homebrew/bin/ffmpeg'), …

after agent append, drums = [2]
after UI PUT from its cached plan, drums = []   (agent pick silently gone)

created 60 jobs; GET first -> 404 ; GET last -> 200 ; GET /jobs?limit=100 count -> 50
```

1. Results carry absolute paths: `TrackOut.file_path` (`schemas.py:120`),
   `RecommendationOut.file_path` (`:140`), `StemFileOut.path` (`:72`),
   `PreviewResult.audio_path` (`:214`), `AlsExportResult.out_path` (`:583`),
   and the `/health/deps` details (`app.py:178`).
2. A second writer erases picks. The UI rebuilds the whole plan from its
   react-query cache and PUTs it on every remove/reorder
   (`useSetPlan.ts:65-83`). That cache never refreshes on its own: the query
   has no interval (`useSetPlan.ts:13-18`) and `refetchOnWindowFocus` is off
   app-wide (`main.tsx:11`). So if the Set view is already open, an agent's
   `plan/append` is erased by the user's next remove or reorder.
3. The job registry keeps only the newest 50 jobs, in memory as well as on disk
   (`jobs.py:106`, `:155`). `/ingest/playlist` creates one job per track, so
   on any playlist longer than 50 tracks the early job ids return 404 before
   anyone polls them.

**Packaging: `mcp` cannot share the dance venv as pinned.**
spotdl 4.4.3 (still the case in the latest, 4.5.2) requires `fastapi<0.104`
and `uvicorn<0.24` (`pyproject.toml:52-55`). Checked against PyPI metadata for
all 73 `mcp` releases:

| package | requires | dance `.venv` has |
|---|---|---|
| fastapi 0.103.1, 0.103.2 | `anyio>=3.7.1,<4.0.0` | fastapi 0.103.2, **anyio 3.7.1** |
| every `mcp` release, 0.9.1 → 2.3.0 | `anyio>=4.5` (2.x: `>=4.9`) | — |
| `mcp` 1.13.0 → 2.3.0 | `uvicorn>=0.31.1` | uvicorn 0.23.2 |
| `mcp` 2.x | `pydantic>=2.12`, `starlette>=0.27`, `httpx2` | pydantic 2.12.5, starlette 0.27.0 |

I asked pip's resolver for the combination without installing anything:

```
.venv/bin/python -m pip install --dry-run --ignore-installed --target <scratch> \
  "spotdl==4.4.3" "fastapi>=0.103,<0.104" "uvicorn>=0.23,<0.24" "httpx>=0.27,<0.28" "mcp"
→ Would install … anyio-4.15.1 fastapi-0.103.0 … mcp-1.12.4 … starlette-0.27.0 uvicorn-0.23.2
```

So "pin an older mcp" has exactly one solution. It downgrades the backend to
**fastapi 0.103.0**, the only 0.103.x without the anyio pin, and moves the
whole backend onto anyio 4. But 0.103.0's missing pin is an accident of timing,
not a statement of support. It shipped on 2023-08-26, four days *before* anyio
4.0.0. The next release, 0.103.1 (2023-09-02), added the pin, *"to handle an
incompatibility"*
([release notes](https://github.com/fastapi/fastapi/releases/tag/0.103.1)). That
route would also freeze the SDK at its 2025-08-07 release, a major version
behind 2.3.0 (2026-10-02).

## Options

### A. A stdio sidecar that calls the running backend over HTTP (the proposed shape)

`dance-mcp` is a separate process that Claude spawns over stdio. It imports
nothing from `dance`. Every tool is one or two calls to an existing route on
`DANCE_API_URL` (default `http://127.0.0.1:8000`).

- **For:** it shares everything that is process-local. That covers the job
  registry (`jobs.py:274`), the plan write lock (`sets.py:60`, a
  `threading.Lock`), the lazy CLAP text encoder (`app.state.embedding_stage`,
  `app.py:107`) and the OSC bridge. Agent-started jobs reach the UI over
  `/ws/pipeline` (`usePipelineJobs.ts`). Active-set switches show within 5 s
  (`useSets.ts:31`). Because the sidecar has no dance imports, the venv split
  costs nothing.
- **Against:** it needs the backend running, so every error has to say how to
  start it. It also inherits the API's gaps: items 1–3 above, and `POST
  /tracks/{id}/tag`, which loads a fresh CLAP per request (see Prerequisites).

### B. An in-process server that imports `dance` and opens the DB directly

- Can't be installed: see the table above.
- Even if it could, it would bypass the process-local plan lock and job
  registry. It would load a second CLAP (1.1 GB of weights in
  `~/.cache/huggingface`) on a 16 GB Mac that is already running Live and
  Demucs. And the UI would never hear about its jobs. **Rejected.**

### C. Mount a Streamable-HTTP MCP endpoint at `/mcp` inside the FastAPI app

- Same install wall as B.
- It would put write tools on the port the browser talks to (CORS allows
  `:5173`/`:5174`, `app.py:111`). **Rejected.**

### D. No server: a project skill that teaches Claude Code to `curl` the API

- **For:** zero packaging.
- **Against:** no Claude Desktop. Nothing structurally excludes the 34
  `/ableton/*` routes, since `curl -X POST …/ableton/crossfader` is one
  approval away. No path stripping. Approvals are Bash prefix patterns, not
  per-tool. Worth having *alongside* A as the workflow layer (see Wiring), not
  instead of it.

## Recommendation: A, packaged in its own venv, with a hand-written allowlist

### Tool surface

Every tool maps to named routes. Tool names are the allowlist: nothing is
generated from the OpenAPI document, because the exclusions are the safety
property. (djbooth derives its MCP tools from a CLI command map with include
globs, `packages/djbooth/src/cli.ts:30`. For dance, a glob like `ableton_*`
would sweep in the transport.)

**Read** — `readOnlyHint: true`, `openWorldHint: false` unless noted:

| tool | route(s) | returns (projected, no paths) |
|---|---|---|
| `status` | `GET /health`, `GET /api/v1/pipeline/status`, `GET /api/v1/health/deps`, `GET /api/v1/pipeline/recent?state=error&limit=10` | counts, `weighted_progress`, dep **statuses** (not `detail`), recent errors |
| `search_tracks` | `GET /api/v1/tracks/search` (`tracks.py:57`) | id, title, artist, bpm, key, energy, state, tags[:8] |
| `spotify_search` *(openWorld)* | `GET /api/v1/spotify/search` (`spotify.py:24`) | spotify_id, title, artist, duration — the input `ingest_track` needs |
| `vibe_search` | `POST /api/v1/recommend/text` (`recommend.py:128`) | same projection + score. First call loads CLAP, ~12–30 s on MPS (`recommend.py:110`), so this tool's HTTP timeout is 60 s |
| `similar_tracks` | `GET /api/v1/recommend/by-seed/{id}` (`recommend.py:76`) | same + reasons |
| `get_track` | `GET /api/v1/tracks/{id}`, `GET …/regions` (track-level only) | analysis + section list |
| `list_sets` | `GET /api/v1/sets`, `GET /api/v1/sets/active` | id, name, is_active, track counts |
| `get_plan` | `GET /api/v1/sets/{id}/plan` (`sets.py:560`); `set_id` defaults to the active set | queues by role |
| `plan_recs` | `GET /api/v1/sets/{id}/plan-recs` (`sets.py:635`) | recs + score_breakdown |
| `live_state` | `GET /api/v1/ableton/state`, `GET /api/v1/ableton/decks` | tempo, is_playing, what is loaded where. No meters or playheads |
| `wait_job` | polls `GET /api/v1/pipeline/jobs/{id}` (`pipeline.py:808`) every 1 s for `timeout_s ≤ 25` | job snapshot, `done`. For `pipeline_run` it also returns `/pipeline/status` counts, because stage items only say "running…" (subprocess, `pipeline.py:894`) |

There is no long-poll route, so `wait_job` polls. The 25 s cap keeps each call
short and gives the user a point to interrupt between polls. Claude Code moves
any main-conversation tool call to the background after 2 minutes
([docs](https://code.claude.com/docs/en/mcp)). A `pipeline_run` lasts hours,
so the server instructions tell the agent to report progress and hand back
rather than loop on it.

**Write** — permission prompt on every call; `readOnlyHint: false`;
`destructiveHint` is set explicitly on each tool because the spec's default is
`true` ([schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-06-18/schema.ts)):

| tool | route | destructive / idempotent / openWorld | notes |
|---|---|---|---|
| `create_set` | `POST /api/v1/sets` | F / F / F | created inactive (`sets.py:207`) |
| `activate_set` | `POST /api/v1/sets/{id}/activate` | **T** / T / F | not additive (it deactivates the other set), so `true` per the spec, although it is reversible |
| `plan_append` | `POST /api/v1/sets/{id}/plan/append` — **never** `PUT /plan` | F / T / F | no-op if already queued (`sets.py:626`) |
| `ingest_track` | `POST /api/v1/pipeline/ingest/track` | F / T / **T** | idempotent on `spotify_id` (`pipeline.py:445`) |
| `ingest_playlist` | `POST /api/v1/pipeline/ingest/playlist` | F / T / **T** | **no `user_token` parameter.** Tokens never go through tool arguments, which end up in transcripts. A 401 says to put `DANCE_SPOTIFY_USER_TOKEN` in `~/.dance/.env` |
| `run_pipeline` | `POST /api/v1/pipeline/process` → job id | F / F / F | hours of Demucs: ~6.9 min/track (16.7 h for 145 copies, [`library-duplicates.md`](library-duplicates.md)) |
| `export_als` | `POST /api/v1/tracks/{id}/als` | **T** / T / F | **no `out_path`**. It overwrites `<als_output_dir>/<title> - <artist>.als` (`generator.py:68-73`), including any edits saved there in Live |
| `tag_track` | `POST /api/v1/tracks/{id}/tag` (CLAP only, never `deep`) | **T** / T / F | replaces inferred tags (`tracks.py:197`) |

`run_pipeline`, `ingest_playlist` and `export_als` also set
`_meta["anthropic/requiresUserInteraction"]: true`. Claude Code then prompts
on every call, in every permission mode, with no "don't ask again"
([docs](https://code.claude.com/docs/en/mcp#require-approval-for-a-specific-tool)).
These are the expensive or overwriting calls. The cheap ones keep normal
prompts, so building a plan doesn't mean twenty dialogs.

**Opt-in, off by default** (`DANCE_MCP_ALLOW_PREVIEW=1`): `preview` /
`preview_stop` → `POST /api/v1/ableton/preview` / `…/preview/stop`
(`ableton.py:831`, `:871`). Audio goes to the Cue track, which routes to
headphones on outs 3/4, so the master is untouched.

**Not wrapped:** 63 of the 90 routes, or 65 with preview off. That count
includes the 4 OpenAPI routes and the 2 WebSockets. The rest:

- every `/ableton/*` route except `state`, `decks` and the preview pair (30
  routes, `DELETE /decks/cell` among them): transport, fire/stop/seek/solo,
  tempo, volume, crossfader, FX, PFL, record, `load-track`, decks
  reset/clean/resync, warp-check;
- the other three `DELETE`s;
- `PUT /sets/{id}/plan` and both `PATCH`es;
- `/sessions/*` (5 routes, performance history);
- the legacy `GET /sets/{id}`, `/sets/{id}/tracks*` and `tail-recs`;
- `POST /recommend` and `/recommend/by-column` (the Booth's live combo recs);
- `/files/reveal`;
- `/pipeline/watch` (both methods), `/pipeline/scan`, `GET /pipeline/jobs` and
  the CSV ingest pair;
- `GET /tracks` and `/tracks/{id}/stems` (which returns paths), and the
  waveforms.

### The performance guard

Every write tool (and `preview`) first reads `GET /api/v1/ableton/state`. If
any deck-column clip is playing (`playing_clips` on a column from
`/ableton/decks`), the tool refuses with this message: *"A deck is playing,
and dance-mcp only preps. Stop playback or finish the set, then ask again."*
The reasons: the plan feeds the Booth's cold recs, `activate_set` swaps the set
the Booth shows within 5 s, and nobody has measured Live's audio while a
Demucs pass holds the GPU. When Live is closed, the bridge reports no playing
clips and the tools are allowed. Using deck clips rather than `is_playing` matters because a cue
preview may start Live's transport (see Verification).

### Errors name the next step; results carry no paths

| backend says | tool returns (`isError: true`) |
|---|---|
| connection refused | "dance backend isn't running at `$DANCE_API_URL`. Start it: `uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000`" |
| `404 no active set` | "No active set. `list_sets`, then `activate_set`, or `create_set` first." |
| `400 role must be one of […]` | passed through + "valid roles: drums, bass, vocals, other, song" |
| `400 track N is in state 'separated', need 'complete'` | + "run `run_pipeline`, then `wait_job`" |
| `409` from `/process` | "A pipeline run is already going (holder named by the dispatcher lock). Use `wait_job`; don't start another." |
| `503` Spotify not configured / `401` user token | "Ask the user to add the credential to `~/.dance/.env`. Never paste it into chat." |
| `503 CLAP model unavailable` | + "`search_tracks` and `similar_tracks` work without it" |
| `404` on a job id it was given | "Job evicted from the 50-job history (`jobs.py:106`). Track progress with `status`." |

Results are **projections**: each tool picks the fields it returns. On top of
that, every string is scrubbed with
`(/Users|/private|/Volumes|/home|/opt|~)/\S+` → `<path>`, so paths can't leak
through error details either. Example: `out_path {resolved} is outside
als_output_dir {root}` (`generator.py:83`), or the pipeline job's 500-char log
tail (`pipeline.py:1013`). This keeps transcripts free of the home layout. It
saves tokens (the default cap is 25k per result, with a warning at 10k, per the
[docs](https://code.claude.com/docs/en/mcp)). And it gives the agent no file to
`Read` or move. Moving files would break every `.als` (CLAUDE.md "Don't move
stems").

Server `instructions` (Claude Code truncates them at 2,048 characters) say the
scope in three lines. Prep only. Append, never rewrite. Ask before ingesting or
processing.

### Packaging (precise)

**Use a separate venv. Don't pin an older mcp.** Pinning means the
fastapi-0.103.0 + anyio-4 swap under the whole backend, plus a frozen 2025 SDK,
all for a sidecar that doesn't need to import the backend.

```
tools/dance-mcp/
  pyproject.toml          # name="dance-mcp", deps: mcp>=2.3,<3 (brings httpx2, anyio>=4.9)
  src/dance_mcp/server.py # MCPServer + ~19 tool fns + _call()/_scrub()/_guard()   (~350 lines)
  src/dance_mcp/routes.json  # the 27 (method, path) pairs — single source for the contract test
  tests/test_server.py
```

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv tools/dance-mcp/.venv   # same interpreter as .venv
tools/dance-mcp/.venv/bin/pip install -e 'tools/dance-mcp[dev]'
```

The existing `.venv/` rule (`.gitignore:27`) already ignores
`tools/dance-mcp/.venv/`. `uv` isn't installed here, and a stdlib venv
needs nothing new. Unpinning the backend is a separate matter: spotdl is only
ever run as a subprocess (`spotify/downloader.py:46`), so moving it to pipx
would free FastAPI. That deserves its own proposal (`csv_importer.py:178`
relies on the yt-dlp that comes with it). It doesn't block this.

### Claude Code wiring

**Register at local scope, not a checked-in `.mcp.json`.**
According to the [docs](https://code.claude.com/docs/en/mcp#project-scope),
`claude -p` runs, Agent SDK sessions and cloud sessions load project-scoped
servers without the approval prompt. Every worktree has its own copy of
`.mcp.json`, so every headless workflow agent in this repo would get these
tools aimed at the real backend. There were 12 worktrees during this run.
Local scope is keyed to the project's path in `~/.claude.json`:

```bash
claude mcp add --env DANCE_API_URL=http://127.0.0.1:8000 --transport stdio --scope local \
  dance -- /Users/arya/git/dance/tools/dance-mcp/.venv/bin/dance-mcp
```

Claude Desktop gets the same command and env in `claude_desktop_config.json`.

**Do narrow `.gitignore`** so a prep skill (for example
`.claude/skills/plan-set/SKILL.md`, the workflow layer from option D) and a
`settings.json` that pre-allows the 11 read tools can be versioned. I tested
this in this worktree with `git check-ignore --no-index` (then reverted):

```
.claude/*                 # replaces line 62 ".claude/"
!.claude/skills/
!.claude/settings.json
```
```
.claude/skills/plan-set/SKILL.md   → not ignored   (today: ignored by .gitignore:62)
.claude/settings.json              → not ignored   (today: ignored)
.claude/settings.local.json        → ignored        .claude/worktrees/…  → ignored
.claude/launch.json                → ignored        tools/dance-mcp/.venv/… → ignored (.gitignore:27)
```

`.claude/` → `!.claude/skills/` wouldn't work: git can't re-include a path
whose parent directory is excluded. Hence `.claude/*`.

### Tests

- **In-memory, in the sidecar venv.** `async with Client(server)` from `mcp`
  ([testing docs](https://py.sdk.modelcontextprotocol.io/get-started/testing/)).
  djbooth uses the same pattern with `InMemoryTransport` in
  `packages/djbooth/test/mcp.test.ts:62-83`. The backend is an
  `httpx2.MockTransport` serving canned payloads, with paths deliberately
  included. The tests assert that:
  - the sorted tool-name set equals the 19-name list exactly (18 until
    prerequisite 3 lands), and is 21 with `DANCE_MCP_ALLOW_PREVIEW=1`;
  - every tool's annotations match the tables;
  - no `inputSchema` has `user_token`, `out_path` or `deep`;
  - no result string matches the path regex;
  - each error row above maps as written;
  - writes refuse while a deck clip plays;
  - `wait_job` clamps to 25 s and maps 404 to "evicted".
- **Route contract, in the dance suite.** `tests/test_mcp_route_contract.py` reads
  `tools/dance-mcp/src/dance_mcp/routes.json` and asserts that each pair exists in
  `create_app(...).routes`. That catches a backend route rename without any
  cross-venv import. The probe above already ran this check: 25/25 present.

## Prerequisites

1. **Dispatcher lock** (`claude/dispatcher-lock`). On main, `/process`'s 409
   only sees in-process jobs (`pipeline.py:13-19`). The lock branch also stops
   the dispatcher claiming `pending:` placeholder rows mid-download.
   *ingest → run_pipeline* is exactly the sequence an agent will issue, and
   without the fix it marks those tracks ERROR "Track file not found".
2. **Plan revision checks** ([`set-plan-revision.md`](set-plan-revision.md)).
   The clobber above becomes routine once a second writer exists. Before the
   tools ship, the UI also has to notice out-of-band appends. `plan/append`
   already bumps `sets.updated_at` (`sets.py:629`), and `useActiveSet` already
   polls it every 5 s, so invalidating the plan query when it changes is a
   front-end-only fix.
3. **`tag_track` shares the encoder.** `tracks.py:213` builds
   `ClapZeroShotTagger(settings)` without the app's encoder, so each call
   loads CLAP from scratch (`tagger.py:188-196`) outside `GPU_SEMAPHORE`.
   Passing `text_encoder=_get_text_encoder(request, settings)` is a trivial
   fix (CLAUDE.md rule 1). Until it lands, leave `tag_track` out.

## Decisions for the user

1. **Venv split** (recommended) vs fastapi 0.103.0 + mcp 1.12.4 in `.venv`.
2. **Local-scope registration** (recommended) vs checked-in `.mcp.json`. The
   latter would also need every write tool to carry `requiresUserInteraction`.
3. **Which writes always prompt.** Proposed: `run_pipeline`,
   `ingest_playlist`, `export_als`. The alternative is all writes, which adds
   friction while building a plan.
4. **The performance guard**: deck clip playing → refuse writes (recommended)
   vs no guard.
5. **Preview**: ship the env-gated pair, or drop it from v1.
6. **Visible provenance for agent picks** (a "Claude added this" marker in the
   plan). Plan items are bare track ids (`set_queues.parse_plan`), so this
   would need a schema change and its own proposal. Not v1.

## Migration cost

- New: `tools/dance-mcp/` (~350 lines of server, ~250 of tests) plus one
  ~40-line contract test in `tests/`.
- Changed: `.gitignore` (1 line becomes 3), a `docs/dev.md` section, and a
  CLAUDE.md "When the user says X" row ("help me prep" → the dance tools).
- Backend: nothing for the tools themselves. Prerequisites 2 (front end) and 3
  (one line) are small and separately mergeable. Reverting means one
  `claude mcp remove dance` and deleting a directory.

## Verification plan

1. `tools/dance-mcp/.venv/bin/python -m pytest tools/dance-mcp/tests -q`, and
   in the dance suite
   `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests/test_mcp_route_contract.py -q`.
2. **Stdio, throwaway data:** use the probe harness above (stub bridge, so it
   doesn't fight the real backend for the OSC port; temp `DANCE_*` dirs and DB;
   port `:8765`), committed with the implementation. Spawn `dance-mcp` with
   the SDK's stdio client. Run `create_set` →
   `activate_set` → `plan_append` ×3 → `get_plan`. With
   `DANCE_API_TARGET=http://127.0.0.1:8765 npx vite --port 5174` open, the
   picks should appear in the Set view without a reload (prerequisite 2), and
   a reorder there must not drop them.
3. **Real rig, real weights (rule 2).** Against the user's backend:
   - `vibe_search("rolling bassline")`, timing the cold CLAP load;
   - `spotify_search` → `ingest_track` → `wait_job` → `run_pipeline` →
     `wait_job` until COMPLETE;
   - `export_als`, then open it in Live and confirm 5 tracks load, audio
     plays and the tempo is right;
   - with a deck clip playing, a write tool must refuse;
   - with preview enabled, check whether firing the Cue clip starts Live's
     transport and whether `preview_stop` leaves Live as it was.

   Record each tool's result size from the transcript.
4. **Wiring.**
   - `claude mcp list` from the main checkout shows `dance`. From a worktree,
     and under `claude -p`, it must not.
   - In Claude Code, ask *"plan a 20-minute warm-up in a new set"*. Reads run
     unprompted, every write prompts, and `run_pipeline` prompts again even
     after "don't ask again" on another tool.
