# Architecture

Three loosely-coupled layers around a single SQLite DB.

```
+---------------------+    +------------------+    +-------------------+
|  Python pipeline    |    |  FastAPI backend |    |  React companion  |
|  src/dance/         |    |  src/dance/api/  |    |  companion-app/   |
|                     |    |                  |    |                   |
|  Spotify -> ingest  |    |  REST + WS       |    |  Booth / Set /    |
|  -> analyze ->      |--->|  reads SQLite,   |<-->|  Pipeline — one   |
|  separate -> ...    |    |  proxies OSC     |    |  plan grid        |
|  -> SQLite          |    |                  |    |                   |
+---------------------+    +------------------+    +-------------------+
                                    ^
                                    | UDP/OSC
                                    v
                            +-------------------+
                            |   Ableton Live    |
                            |  + AbletonOSC     |
                            +-------------------+
```

SQLite is the only source of truth. Audio files live under `library_dir` and `stems_dir`; everything else (analysis, tags, regions, embeddings, edges, sessions) is in the DB.

## Layer 1 — pipeline

`src/dance/pipeline/` is a state-machine. Each `Track` row has a `state` (see `TrackState` in `src/dance/core/database.py:68`); a "stage" is a unit of work that consumes tracks in one state and writes them into the next.

### The Stage protocol

`src/dance/pipeline/stage.py:24`:

```python
class Stage(Protocol):
    name: str
    input_state: TrackState
    output_state: TrackState
    error_state: TrackState = TrackState.ERROR

    def process(self, session: Session, track: Track, settings: Settings) -> bool: ...
```

That is the entire surface. A stage knows only its own input/output states. No stage imports another; no stage knows where it sits in the pipeline.

### The Dispatcher

`src/dance/pipeline/dispatcher.py:33`. Holds a list of registered stages. `Dispatcher.run()` loops:

```
while changed:
    for stage in stages:
        tracks = SELECT * FROM tracks WHERE state = stage.input_state
        for track in tracks: stage.process(...)
```

There is no `if stage_name == "analyze": ...` chain anywhere. Order emerges from the state graph the stages declare:

| Stage                  | input_state          | output_state         | File                                                |
|------------------------|----------------------|----------------------|-----------------------------------------------------|
| `analyze`              | `pending`            | `analyzed`           | `src/dance/pipeline/stages/analyze.py`              |
| `separate`             | `analyzed`           | `separated`          | `src/dance/pipeline/stages/separate.py`             |
| `analyze_stems`        | `separated`          | `stems_analyzed`     | `src/dance/pipeline/stages/analyze_stems.py`        |
| `detect_regions`       | `stems_analyzed`     | `regions_detected`   | `src/dance/pipeline/stages/detect_regions.py`       |
| `embed`                | `regions_detected`   | `complete`           | `src/dance/pipeline/stages/embed.py`                |

`ingest` (`src/dance/pipeline/stages/ingest.py:52`) lives outside the loop — it scans the filesystem for new files and inserts them at `pending`. `dispatcher.ingest()` (`src/dance/pipeline/dispatcher.py:108`) calls it.

```
   filesystem
       |
       v  Dispatcher.ingest()
   pending --analyze--> analyzed --separate--> separated --analyze_stems-->
   stems_analyzed --detect_regions--> regions_detected --embed--> complete
```

The outer loop is bounded by `2 * len(stages)` iterations (`dispatcher.py:145`) to prevent runaway oscillation if a buggy stage ever writes its own input_state.

### Why this shape

- Adding a stage is one file + one `dispatcher.register()` call. Existing stages don't change.
- `dance process --skip-stems` works by name lookup, not branching: skipped stages auto-advance state (`dispatcher.py:221`).
- Re-running is idempotent — if a stage already wrote everything it needs (see `separate.py:84` checking for existing stems), it short-circuits and just writes `output_state`.

### EventBus

`src/dance/pipeline/events.py:37`. Trivial synchronous pub/sub. The dispatcher emits `StageEvent(kind="started"|"completed"|"failed", stage_name, track_id, duration_ms, error)`. Subscribers register via `bus.subscribe(callback)`. Used today only by the default Rich logger (`dispatcher.py:236`); designed to feed the WebSocket progress stream eventually.

### Shared utils — `src/dance/pipeline/utils/`

DRY surface for stages. Anything used in 3+ places goes here; nothing else.

- `audio.py` — `aggregate_rms`, `normalize_bpm` (pulls half/double-time detections into 118-145), `detect_key_from_chroma`. Used by both full-mix `analyze` and per-stem `analyze_stems`.
- `db.py` — `upsert(session, Model, where=..., **values)` and `get_stems_for_track`. Don't add SQLAlchemy helpers anywhere else.
- `device.py` — `pick_device("auto")` returns `"cuda"` > `"mps"` > `"cpu"`. One function. Used by `separate.py`, `embed.py`, and `qwen_audio.py`.
- `beats.py`, `camelot.py` — beat-grid math and Camelot wheel adjacency.

## Layer 2 — FastAPI backend

`src/dance/api/`. Read-mostly REST over SQLite + a thin OSC passthrough + a WebSocket for live Ableton state.

### Composition

`src/dance/api/app.py:25` — `create_app(settings, bridge, session_factory)` is the only entry point. All three args are injectable for tests.

```python
app.state.settings        # Settings singleton
app.state.bridge          # AbletonBridge (lifespan-managed)
app.state.session_factory # SQLAlchemy sessionmaker
app.state.ws_manager      # WSManager (WebSocket connection set)
app.state.embedding_stage # Lazy-loaded CLAP for /recommend/text
```

`src/dance/api/deps.py` providers (`get_settings`, `get_bridge`, `get_session`) read from `app.state` so tests can swap by overriding the providers.

### Routers

| Prefix             | File                                  | Notes |
|--------------------|---------------------------------------|-------|
| `/api/v1/tracks`   | `routers/tracks.py`                   | List, get, regions, stems, tag, .als export |
| `/api/v1/recommend`| `routers/recommend.py`                | Graph + text recommend; `by-column` (live combo + `trailing_track_ids` journey) |
| `/api/v1/sets`     | `routers/sets.py`                     | Set CRUD + the **plan** (`/plan`, `/plan/append`, `/plan-recs`); legacy `set_tracks` + `/tail-recs` |
| `/api/v1/sessions` | `routers/sessions.py`                 | DJ session CRUD |
| `/api/v1/ableton`  | `routers/ableton.py`                  | OSC passthrough + push-to-Live |
| `/api/v1/files`    | `routers/files.py`                    | Reveal-in-Finder (allowlist-checked) |
| `/ws`              | `routers/ws.py`                       | WebSocket — pushes `AbletonState` |

All response shapes are in `src/dance/api/schemas.py`. See `docs/api.md` for the route reference; runtime docs at `/docs`.

### WebSocket

`src/dance/api/routers/ws.py`. The OSC listener runs on a background thread; FastAPI runs on asyncio. We capture the loop in `lifespan` (`app.py:51`) and the bridge subscriber calls `WSManager.broadcast_threadsafe`, which uses `asyncio.run_coroutine_threadsafe` to hop over.

## Layer 3 — React companion app

`companion-app/`. Vite + React 18 + TypeScript + Tailwind. Built around **one surface — the plan grid** (`RoleColumnsGrid`): five role columns (drums · bass · vocals · other · song), each stacking the DJ's queued plan picks on top and recommendations below. The same grid renders in two modes:

- **Booth** (`mode="live"`) — recs tail what's playing in Ableton (combo embedding + trailing-journey trend via `useColumnRecs`); each card has ⤒A/⤒B to load a pick onto a deck.
- **Set view** (`mode="plan"`) — recs are scored against the rest of the plan + the plan's journey (`usePlanRecs` → `/sets/{id}/plan-recs`); no deck-load (planning, not firing).

Both share the ▶ Cue preview and a ScoreBreakdown chip row. A *set is its plan* — built by queuing recs (＋), removing (×), or ⌘K (which appends to the Song column server-side). `MasterStrip` + `TwoDeckStrip` + `Crossfader` + `BoothColumnHeaders` + the 8-column `SceneGrid` (Ableton mirror) + `CueStrip` are the surrounding Booth surfaces.

### Deliberate non-choices

- **No router.** Three views, switched via `useAppStore((s) => s.currentView)` (`src/store.ts`). View enum (`types.ts`): `"booth" | "set" | "pipeline"`. ⌘K is the universal finder across all three.
- **No state library.** `src/store.ts` is a small `useSyncExternalStore` over a module-level mutable. Holds: `currentView`, `currentSessionId`, `loadedDecks` (scene_index → deck, linking Push-to-Live calls to `playing_clips` from the WS), `commandBarOpen`, `previewing` (the auditioning Cue candidate), and `stemColumnOrder`. Persisted to localStorage.
- **No CSS framework other than Tailwind utility classes.** Custom palette tokens live in `tailwind.config.js`.

### Layout

```
src/
  App.tsx              -- view switch (Booth / SetEditor / PipelineOps) + MasterStrip + CommandBar
  main.tsx             -- QueryClientProvider + Tailwind import
  store.ts             -- ad-hoc app store (view, session, decks, preview, column order)
  api.ts               -- typed fetch wrappers (one per endpoint)
  types.ts             -- mirrors api/schemas.py (PLAN_ROLES, PlanItem, ColumnRec, …)
  lib/roles.ts         -- role labels/styles, fader-order (TWO_DECK_COLUMN_ORDER)
  components/
    RoleColumnsGrid / RoleColumn  -- the plan grid (queued picks + recs, both modes)
    MasterStrip, TwoDeckStrip, Crossfader, BoothColumnHeaders, SceneGrid, CueStrip
    CommandBar (⌘K), ScoreBreakdown, SetMenu, …
  hooks/
    useSetPlan.ts      -- useSetPlan, usePlanMutations (add/remove), usePlanRecs
    useColumnRecs.ts   -- live per-column recs (Booth), scored vs the playing combo
    useSets.ts         -- useActiveSet, useCreateSet, useActivateSet
    usePreview.ts      -- Cue/headphones audition (start/stop/state)
    useAbletonState.ts -- subscribes to /ws, auto-reconnect with backoff
    useDeckMap, useTransport, useTracks, useRecommend, useSession, …
  views/               -- Booth, SetEditor, PipelineOps (+ PipelineBoard)
```

### Data flow

```
react-query --HTTP--> FastAPI --SQLAlchemy--> SQLite
useAbletonState --WS--> WSManager <--callback-- AbletonBridge <--UDP-- AbletonOSC
RoleColumn ⤒A/⤒B --HTTP--> /api/v1/ableton/load-track --create_audio_track--> Live   (Booth, mode="live")
RoleColumn ＋ / ⌘K --HTTP--> PUT|POST /sets/{id}/plan[/append] --> sets.plan          (plan edits)
```

The Ableton state flow is one-way push (Live -> bridge -> WS -> React). User actions go the other direction via the REST endpoints in `/api/v1/ableton/*` (deck loads) and `/sets/{id}/plan*` (plan edits).

## Schema overview

Authoritative: `src/dance/core/database.py`. Highlights:

```
tracks              -- 1 row per audio file (PK = content-hash dedup'd)
  |- stem_files     -- 4 per track (drums/bass/vocals/other)
  |- audio_analysis -- 1 full-mix row + 1 per stem (stem_file_id IS NULL = mix)
  |- regions        -- cues, loops, fades, sections, stem-solo windows
  |- track_embeddings -- CLAP vectors, full-mix + per-stem
  |- track_tags     -- M:N with `tags`; source = inferred|llm|manual
  |- beats, phrases -- beat grid + detected musical phrases

track_edges         -- pairwise recommendation graph
  kinds: harmonic_compat, tempo_compat, embedding_neighbor,
         tag_overlap, manually_paired, playlist_neighbor

sessions            -- DJ set, started_at -> ended_at (what actually played)
  |- session_plays  -- ordered by position_in_set

sets                -- curated plan (what's *planned*), at most one is_active
  |  plan            -- TEXT (JSON) per-role queues {role: [track_id,...]} —
  |                     roles = drums/bass/vocals/other/song. A set IS its plan.
  |- set_tracks      -- LEGACY ordered track list (0-indexed, unique per set_id);
                        still in the schema but no longer driven by the UI
```

A `Set`'s authoritative content is now its **`plan`** column: a JSON map of per-role queues (`{role: [track_id, …]}`, roles = the `PlanRole` enum: drums/bass/vocals/other/song, where `song` == the recommender's `mix`). Parsing/encoding/journey-context helpers are pure functions in `src/dance/core/set_queues.py`; the plan endpoints live on the sets router (`GET/PUT /sets/{id}/plan`, `POST /sets/{id}/plan/append`, `GET /sets/{id}/plan-recs?role=`). Added additively by Alembic revision `b7d4e2f1a9c3` — no other schema change. The older `set_tracks` table (ordered list + the `/sets/{id}/tail-recs` endpoint) is **legacy**: still present and migrated, but the React UI drives the plan grid off `plan`, not `set_tracks`.

Key invariants:

- `audio_analysis` uses partial unique indexes (`database.py:_create_partial_unique_indexes`) — one full-mix row per track, one row per stem.
- `sets` has a partial unique index on `is_active WHERE is_active` — only one active set at a time, enforced at the DB layer (same raw-DDL pattern as `audio_analysis`).
- `set_tracks` (legacy) has a `UNIQUE(set_id, position)` — reorder uses sentinel-renumber to avoid mid-flush collisions (`api/routers/sets.py:_renumber_for_move`).
- `track_edges` has no self-loops (CHECK constraint).
- Cascade deletes everywhere: drop a `Track` and stems/analysis/regions/embeddings/tags/edges all go with it. Drop a `Set` and its `set_tracks` go (the `plan` column travels with the row).
- `Set` and `DjSession` are decoupled — a set is reused across sessions; a session may or may not follow a set. Legacy tail-recs can optionally exclude tracks already played in the current open session (`exclude_session_plays=true`).

For SQL DDL, read `src/dance/core/database.py` — copying it here would just bit-rot.

## Recommender layer

`src/dance/recommender/`. One **shared brain** under three entry points, so the Booth, the planner, and the seed graph can never drift apart (they used to — the graph caught half/double-time BPM matches the live scorers silently missed).

The brain:

- `scoring.py` — the feature scorers and the blender. Each signal (`embed_score`, `key_score`, `bpm_score` with ±8 BPM tolerance + half/double-time folding, `energy_score`, `kick_density_score`, `presence_score`, `timbre_score`) maps to `[0, 1]` and is `None`-safe. A `Profile` is a weight vector over `FEATURES`; `combine()` does the weighted blend and **renormalizes over only the features actually present**, so a missing signal down-weights gracefully. Per-context profiles live in `PROFILES` (`column:drums` drops key and leans on `kick_density`, `column:vocals` leans on key, etc.).
- `journey.py` — the time-shaped context. `JourneyState` captures the set's *trajectory* (vibe target, vibe **trend** vector, target keys/BPM, projected energy, anti-repetition set). The vibe axis is **trend-aware**: the target is pushed forward along the direction the set has been moving (`normalize(target + β·trend)`), so recs continue the journey instead of repeating it. Two builders: `journey_from_tracks` (planner — from the ordered plan/set so far) and `journey_from_combo` (Booth — from the active stem combo + trailing live tracks).
- `structure.py` — `transition_fit`: a `[0, 1]` mix-compatibility signal from intro/outro SECTION regions ("how cleanly can B come in over A?"). Fed into `combine` as the `transition_fit` feature.

The three entry points:

- `recommender.py` — `recommend_by_column(session, column, combo_stem_ids, *, k, exclude_track_ids, trailing_track_ids)`: top-K candidate stems (or tracks for `column='mix'`) re-scored against the active combo through the shared brain. `trailing_track_ids` supplies the journey context (trend-aware vibe + soft anti-repetition). Powers the live Booth recs **and** the Set view's plan-scored recs (the sets router passes the plan's trailing sequence as `trailing_track_ids`). Returns `ColumnRecResult(track_id, stem_file_id, score, score_breakdown, reasons)`.
- `recommender.py` — `Recommender(session).recommend(seeds=[1,2], k=10, kinds=[...], weights={...}, exclude=[...])`. SQL on `track_edges`: aggregate per candidate by summing `weight * kind_weight`. Returns `RecommendationResult(track_id, score, reasons=[{kind, from_seed, weight}, ...])`.
- `recommender.py` — `recommend_by_text(query, text_encoder, k)`. CLAP joint embedding: encode query, cosine-rank all full-mix embeddings, top-K. Bypasses the graph entirely (backs the vibe half of ⌘K).
- `tail.py` — `tail_recs_for_set` (legacy): arc-fit candidates to append to a Set, scored over the trailing window via the shared brain. Backs the legacy `/sets/{id}/tail-recs` endpoint; not used by the current UI.

`graph_builder.py:47` — `GraphBuilder(session, settings).build(track_ids=None)`. Library-level operation (not a stage). Reads `audio_analysis`, `track_tags`, `track_embeddings`; writes `track_edges`. One private builder per kind (`_build_harmonic`, `_build_tempo`, `_build_embedding`, `_build_tag_overlap`); each kind is `DELETE WHERE kind=X (AND touches tracks)` then `INSERT`. Symmetric kinds materialize both directions. Feeds the seed-graph `recommend()` above.

## OSC bridge

`src/dance/osc/`.

```
                  +----------------+
  HTTP /ableton ->| AbletonBridge  |<-+
                  +----------------+  |
                   |        |        |
                   v        v        |
            AbletonOSCClient  AbletonOSCListener
              (sends UDP)      (UDP server thread)
                   |              |
                   v              ^
              port 11000 -----> port 11001
                          Live
```

- `client.py:26` — typed wrappers around AbletonOSC addresses (`/live/song/set/tempo`, `/live/clip_slot/fire`, etc.). Fire-and-forget UDP. Defaults: `127.0.0.1:11000` send, `127.0.0.1:11001` receive.
- `listener.py:25` — `ThreadingOSCUDPServer` on background thread. Per-address handler list, plus an `on_any("*", ...)` catch-all.
- `bridge.py:53` — combines the two and maintains `AbletonState` (latest tempo/beat/playing-clip per track/volume). `subscribe(cb)` for downstream consumers; `push_track_to_live(track, stems)` is the high-level "create N empty audio tracks, name and color them, status-bar nudge."

Known limitation: AbletonOSC has no command to load a sample into a clip slot, so `push_track_to_live` only prepares empty tracks. See `docs/abletonosc_setup.md` for the full explanation.

## Putting it together — request lifecycle

User taps "Push to Live" on a TrackCard in the React UI:

```
React TrackCard onClick
  -> api.pushTrackToLive(trackId)            # companion-app/src/api.ts:192
  -> POST /api/v1/ableton/load-track          # routers/ableton.py:75
  -> AbletonBridge.push_track_to_live(...)    # osc/bridge.py:215
       -> client.get_num_tracks() + wait reply on listener
       -> client.create_audio_track(-1)       # x (1 + n_stems)
       -> client.set_track_name + set_track_color
       -> client.show_message(...)
  -> returns LoadTrackResult                   # schemas.py:199
  -> React reveals stems folder in Finder via /api/v1/files/reveal
```

Meanwhile the listener thread is receiving `/live/song/get/tempo` pushes, updating `bridge.state.tempo`, and the bridge subscriber posts to `WSManager.broadcast_threadsafe` -> all open `/ws` clients see the new tempo within a frame.
