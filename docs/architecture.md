# Architecture

Three loosely-coupled layers around a single SQLite DB.

```
+---------------------+    +------------------+    +-------------------+
|  Python pipeline    |    |  FastAPI backend |    |  React companion  |
|  src/dance/         |    |  src/dance/api/  |    |  companion-app/   |
|                     |    |                  |    |                   |
|  Spotify -> ingest  |    |  REST + WS       |    |  Now / Up Next /  |
|  -> analyze ->      |--->|  reads SQLite,   |<-->|  Library / Set    |
|  separate -> ...    |    |  proxies OSC     |    |  History          |
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
| `/api/v1/recommend`| `routers/recommend.py`                | Graph + text recommend |
| `/api/v1/sessions` | `routers/sessions.py`                 | DJ session CRUD |
| `/api/v1/ableton`  | `routers/ableton.py`                  | OSC passthrough + push-to-Live |
| `/api/v1/files`    | `routers/files.py`                    | Reveal-in-Finder (allowlist-checked) |
| `/ws`              | `routers/ws.py`                       | WebSocket — pushes `AbletonState` |

All response shapes are in `src/dance/api/schemas.py`. See `docs/api.md` for the route reference; runtime docs at `/docs`.

### WebSocket

`src/dance/api/routers/ws.py`. The OSC listener runs on a background thread; FastAPI runs on asyncio. We capture the loop in `lifespan` (`app.py:51`) and the bridge subscriber calls `WSManager.broadcast_threadsafe`, which uses `asyncio.run_coroutine_threadsafe` to hop over.

## Layer 3 — React companion app

`companion-app/`. Vite + React 18 + TypeScript + Tailwind. iPad-landscape first.

### Deliberate non-choices

- **No router.** Four views, switched via `useAppStore((s) => s.currentView)` (`src/store.ts`). View enum: `"now" | "next" | "library" | "session"`.
- **No state library.** `src/store.ts` is ~75 lines of `useSyncExternalStore` over a module-level mutable. Holds: `pinnedSeeds`, `currentSessionId`, `currentView`.
- **No CSS framework other than Tailwind utility classes.** Custom palette tokens live in `tailwind.config.js`.

### Layout

```
src/
  App.tsx              -- view switch
  main.tsx             -- QueryClientProvider + Tailwind import
  store.ts             -- 4-key app store
  api.ts               -- typed fetch wrappers (one per endpoint)
  types.ts             -- mirrors api/schemas.py
  components/          -- TrackCard, EnergyBar, KeyBadge, PinButton, LoadActions, TopBar
  hooks/
    useTracks.ts       -- useTracks, useTrack, useStems (react-query)
    useRecommend.ts    -- useRecommend(seeds, k)
    useSession.ts      -- useCurrentSession, useCreateSession, useAddPlay, useEndSession
    useAbletonState.ts -- subscribes to /ws, auto-reconnect with 2s backoff
  views/               -- NowPlaying, UpNext, Library, SessionHistory
```

### Data flow

```
react-query --HTTP--> FastAPI --SQLAlchemy--> SQLite
useAbletonState --WS--> WSManager <--callback-- AbletonBridge <--UDP-- AbletonOSC
LoadActions --HTTP--> /api/v1/ableton/load-track --AbletonOSCClient.create_audio_track--> Live
```

The Ableton state flow is one-way push (Live -> bridge -> WS -> React). User actions go the other direction via the REST endpoints in `/api/v1/ableton/*`.

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

sessions            -- DJ set, started_at -> ended_at
  |- session_plays  -- ordered by position_in_set
```

Key invariants:

- `audio_analysis` uses partial unique indexes (`database.py:721`) — one full-mix row per track, one row per stem.
- `track_edges` has no self-loops (CHECK constraint, `database.py:522`).
- Cascade deletes everywhere: drop a `Track` and stems/analysis/regions/embeddings/tags/edges all go with it.

For SQL DDL, read `src/dance/core/database.py` — copying it here would just bit-rot.

## Recommender layer

`src/dance/recommender/`. Two halves:

- `graph_builder.py:47` — `GraphBuilder(session, settings).build(track_ids=None)`. Library-level operation (not a stage). Reads `audio_analysis`, `track_tags`, `track_embeddings`; writes `track_edges`. One private builder per kind (`_build_harmonic`, `_build_tempo`, `_build_embedding`, `_build_tag_overlap`); each kind is `DELETE WHERE kind=X (AND touches tracks)` then `INSERT`. Symmetric kinds materialize both directions.
- `recommender.py:43` — `Recommender(session).recommend(seeds=[1,2], k=10, kinds=[...], weights={...}, exclude=[...])`. SQL on `track_edges`: aggregate per candidate by summing `weight * kind_weight`. Returns `RecommendationResult(track_id, score, reasons=[{kind, from_seed, weight}, ...])`.
- `recommender.py:110` — `recommend_by_text(query, text_encoder, k)`. CLAP joint embedding: encode query, cosine-rank all full-mix embeddings, top-K. Bypasses the graph entirely.

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
