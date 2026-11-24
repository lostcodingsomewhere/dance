# API reference

FastAPI app factory: `src/dance/api/app.py:25` (`create_app`). All routes are mounted under `/api/v1` except `/ws` and `/health`.

Run it:

```bash
uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000
```

Live, auto-generated docs at `http://127.0.0.1:8000/docs` (Swagger) and `/redoc`. **Treat that as the source of truth for field-level details.** This page groups routes and surfaces the non-obvious behavior.

Response/request shapes live in `src/dance/api/schemas.py`.

---

## Health

| Method | Path     | Response                  |
|--------|----------|---------------------------|
| GET    | `/health`| `{"ok": true}`            |

---

## Tracks — `src/dance/api/routers/tracks.py`

| Method | Path                                  | Body / Query                                                                 | Response                          | 4xx |
|--------|---------------------------------------|------------------------------------------------------------------------------|-----------------------------------|-----|
| GET    | `/api/v1/tracks`                      | `limit, offset, bpm_min, bpm_max, key, energy, state`                        | `list[TrackOut]`                  |     |
| GET    | `/api/v1/tracks/search`               | `q, limit, bpm_min, bpm_max, key, energy`                                    | `list[TrackOut]`                  |     |
| GET    | `/api/v1/tracks/{id}`                 | -                                                                            | `TrackOut`                        | 404 |
| GET    | `/api/v1/tracks/{id}/versions`        | -                                                                            | `list[TrackVersionOut]` (the song's other versions, each with a `label`) | 404 |
| GET    | `/api/v1/tracks/{id}/regions`         | `region_type, stem_file_id`                                                  | `list[RegionOut]`                 | 404 |
| GET    | `/api/v1/tracks/{id}/pads`            | none                                                                         | `list[PadOut]` (five per stem and for the mix; `silent` is worked out on read) | 404 |
| PUT    | `/api/v1/tracks/{id}/pads/{role}/{slot}` | body `{position_ms}`; `role` is `drums`, `bass`, `vocals`, `other`, `mix` or `all`; `slot` 1-5 | `PadOut` (snapped to the nearest bar line, `source: manual`) | 404, 409 (no beat grid), 422 (outside the song) |
| DELETE | `/api/v1/tracks/{id}/pads/{role}/{slot}` | none                                                                      | `list[PadOut]` (that pad is back to the app's pick) | 404 |
| GET    | `/api/v1/tracks/{id}/stems`           | -                                                                            | `list[StemFileOut]`               | 404 |
| GET    | `/api/v1/tracks/{id}/waveform`        | `num_peaks` (query, default 200, range 50-1000)                              | `WaveformOut`                     | 404, 422, 503 |
| GET    | `/api/v1/stems/{stem_file_id}/waveform` | `num_peaks` (query, default 200, range 50-1000)                            | `WaveformOut`                     | 404, 422, 503 |
| POST   | `/api/v1/tracks/{id}/tag`             | query: `deep` (bool)                                                         | `TrackOut`                        | 404, 502, 503 |
| POST   | `/api/v1/tracks/{id}/als`             | `AlsExportRequest` (`out_path?`)                                             | `AlsExportResult`                 | 400, 403, 404 |

Notes:

- `GET /tracks` joins `audio_analysis` only when an analysis filter is provided (`tracks.py:51`).
- **Versions.** Two tracks are versions of one song when they have the same base title and share at least one credited artist, and are not the same recording. The base title drops featuring credits, bracketed or ` - ` tail text that names a version (`(Extended Mix)`, `- Club Mix`, `[HUGEL Remix]`) and "original mix"; other brackets stay part of the title. Two rows are the same recording when their lengths are within 5 s and their version text is the same, so a second copy or another spelling of one recording is not a version. Only complete tracks that are not marked as duplicates take part. Nothing is stored: the groups are worked out per request from the track table (`recommender/versions.py`; 15 ms on 244 tracks). `GET /tracks/{id}/versions` lists a track's versions with a `label` relative to it: the version's own text (`Club Mix`, `TH;EN Remix`), `original` for the plain title, or `longer cut` / `shorter cut` when the titles say the same. Every `TrackOut` carries `version_count` (how many other versions exist) and `version_family` (the same number on every version of one song, null otherwise). `/tracks/search` keeps the versions it finds next to each other.
- `key` is the Camelot code (`"8A"`, `"3B"`, ...), uppercased.
- `tag?deep=true` requires `DANCE_DEEP_TAGGER_ENABLED=true` and returns 503 otherwise (`tracks.py:126`).
- `tag` returns 502 if the tagger raises (model load failure, audio missing, etc.).
- `als` returns 403 when `out_path` resolves outside `settings.als_output_dir` (`tracks.py:175`), 400 for missing analysis / no stems / not COMPLETE.
- `waveform` endpoints decimate the audio (librosa) into `num_peaks` amplitude windows normalized to `[0, 1]`. Result is cached as a `{path}.waveform.json` sidecar so subsequent calls are ~instant. 422 for out-of-range `num_peaks`; 404 if the audio file is missing on disk; 503 on decode failure.
- `GET /tracks/search` is the Cmd-K fuzzy half — case-insensitive `LIKE` on `title` + `artist`, with prefix matches ranked above contains-matches. Optional `bpm_min/bpm_max/key/energy` chip filters. Empty `q` returns the most-recently-updated tracks (browse mode). At our scale (<10k tracks) a sequential scan stays sub-100 ms; cross 50k tracks and we'll add a trigram index.

---

## Recommend — `src/dance/api/routers/recommend.py`

| Method | Path                                  | Body / Query                                       | Response                          | 4xx |
|--------|---------------------------------------|----------------------------------------------------|-----------------------------------|-----|
| POST   | `/api/v1/recommend`                   | `RecommendRequest`                                 | `list[RecommendationOut]`         | 400 |
| GET    | `/api/v1/recommend/by-seed/{id}`      | `k` (query, default 10)                            | `list[RecommendationOut]`         |     |
| POST   | `/api/v1/recommend/text`              | `TextRecommendRequest`                             | `list[RecommendationOut]`         | 400, 503 |
| POST   | `/api/v1/recommend/by-column`         | `ColumnRecsRequest`                                | `ColumnRecsResponse`              | 400 |

`RecommendRequest`:

```json
{
  "seeds": [12, 17],
  "k": 10,
  "kinds": ["harmonic_compat", "embedding_neighbor"],
  "weights": {"harmonic_compat": 1.5, "embedding_neighbor": 0.5},
  "exclude": [3, 4]
}
```

`kinds` must be valid `EdgeKind` values (`src/dance/core/database.py:148`) — invalid -> 400. Weights default to 1.0 per kind.

`/recommend/text` accepts a free-text `query` ("punchy techy with vocals") and ranks by CLAP cosine. First call lazy-loads the CLAP model — slow (~5-10 s); subsequent calls are cached on `app.state.embedding_stage`. Returns 503 if the model fails to load.

`/recommend/by-column` is the live-remixing rec stream: takes a `column` (`drums` / `bass` / `vocals` / `other` / `mix`) plus the active combo's `combo_stem_ids` + `master_bpm` and returns top-K candidates (for `mix`, with `deck_track_ids` naming the songs on the decks, their other versions come first, up to half of `k`, each marked with `version_of`, `version_of_title` and `version_label`) filtered by stem kind, scored against the combo via per-stem embedding cosine + Camelot key compat + BPM proximity. It also accepts `trailing_track_ids` — the recency-ordered sequence played so far this set — which feeds the journey's trend-aware vibe and soft anti-repetition; when empty it scores against the combo only. Returns 400 for unknown column. Powers the per-role columns of the rec grid (`RoleColumnsGrid`) in Booth (`mode="live"`) — one query per column, re-run on combo change. (`GET /sets/{id}/plan-recs` is the planning-mode sibling — same response shape, scored against the plan instead of Live.)

---

## Sessions — `src/dance/api/routers/sessions.py`

| Method | Path                                  | Body                          | Response       | 4xx |
|--------|---------------------------------------|-------------------------------|----------------|-----|
| POST   | `/api/v1/sessions`                    | `SessionCreateRequest`        | `SessionOut`   |     |
| GET    | `/api/v1/sessions/current`            | -                             | `SessionOut`   | 404 |
| GET    | `/api/v1/sessions/{id}`               | -                             | `SessionOut`   | 404 |
| POST   | `/api/v1/sessions/{id}/plays`         | `SessionPlayCreateRequest`    | `SessionOut`   | 404 |
| POST   | `/api/v1/sessions/{id}/end`           | -                             | `SessionOut`   | 404 |

`/sessions/current` returns the most recent session with `ended_at IS NULL`. `position_in_set` is auto-incremented on `POST /plays` (`sessions.py:98`). `energy_at_play` is snapshotted from the track's current full-mix analysis.

---

## Sets — `src/dance/api/routers/sets.py`

A **Set is its plan**. The plan is per-role queues of source tracks the DJ intends to bring in — `{role: [track_id, ...]}`, roles = `drums`/`bass`/`vocals`/`other`/`song` (`song` == the recommender's `mix` column). It lives inside the rec grid (`RoleColumnsGrid`): each role column stacks queued picks on top, live recs below. Stored as JSON-as-Text on the `sets.plan` column (migration `b7d4e2f1a9c3`, additive). Distinct from `DjSession`: a Set is the *intent* (planned), a Session is the *history* (what actually played). At most one Set is `is_active = true` at a time (enforced by a partial unique index in `_create_partial_unique_indexes`).

| Method | Path                                              | Body / Query                              | Response                | 4xx |
|--------|---------------------------------------------------|-------------------------------------------|-------------------------|-----|
| GET    | `/api/v1/sets`                                    | -                                         | `list[SetSummaryOut]`   |     |
| GET    | `/api/v1/sets/active`                             | -                                         | `SetOut`                | 404 |
| POST   | `/api/v1/sets`                                    | `SetCreateRequest`                        | `SetOut`                |     |
| GET    | `/api/v1/sets/{id}`                               | -                                         | `SetOut`                | 404 |
| PATCH  | `/api/v1/sets/{id}`                               | `SetUpdateRequest`                        | `SetOut`                | 404 |
| DELETE | `/api/v1/sets/{id}`                               | -                                         | 204                     | 404 |
| POST   | `/api/v1/sets/{id}/activate`                      | -                                         | `SetOut`                | 404 |
| GET    | `/api/v1/sets/{id}/plan`                          | -                                         | `SetPlanOut`            | 404 |
| PUT    | `/api/v1/sets/{id}/plan`                          | `SetPlanUpdateRequest`                    | `SetPlanOut`            | 400, 404 |
| POST   | `/api/v1/sets/{id}/plan/append`                   | `SetPlanAppendRequest`                    | `SetPlanOut`            | 400, 404 |
| POST   | `/api/v1/sets/{id}/als` | `SetAlsExportRequest` (`out_path?`, `new_file?`: write `<set name> <date> <time>.als` and leave earlier files alone; the default is `<set name>.als`, replaced each time) | `SetAlsExportResult` (`out_path`, `page_count`, `scene_count`, `clip_count`, `skipped`): the whole plan as one Live Set, see `als-export.md` | 400 (empty plan, or nothing exportable), 403 (outside `als_output_dir`), 404 |
| GET    | `/api/v1/sets/{id}/plan-recs`                     | `role` (req), `k`                         | `ColumnRecsResponse`    | 400, 404 |
| POST   | `/api/v1/sets/{id}/tracks`                        | `SetTrackAddRequest`                      | `SetOut`                | 400, 404 |
| PATCH  | `/api/v1/sets/{id}/tracks/{track_id}`             | `SetTrackUpdateRequest`                   | `SetOut`                | 400, 404 |
| DELETE | `/api/v1/sets/{id}/tracks/{track_id}`             | -                                         | `SetOut`                | 404 |
| GET    | `/api/v1/sets/{id}/tail-recs`                     | `k, window, exclude_session_plays`        | `TailRecsResponse`      | 404 |

- `activate` is atomic: deactivates other sets in the same transaction, then sets the target — the partial unique index never sees two active rows.
- **Plan** (`SetPlanOut` = `{set_id, queues: {role: PlanItemOut[]}}`, every role key always present):
  - `GET /plan` returns the decoded queues, each item decorated with title/artist/key/BPM/energy for render.
  - `PUT /plan` replaces the whole plan; body `{queues: {role: [track_id, ...]}}`. The frontend edits locally (add / remove / reorder) and PUTs the result. Unknown roles → 400.
  - `POST /plan/append` appends one track to a role's queue with a **server-side merge** — body `{role, track_id, after_track_id?}` — so it's safe from ⌘K without the full plan loaded client-side. ⌘K appends to the `song` column. With `after_track_id` naming a track in that queue, the new one goes right after it; the app sends this when it queues a version, so the version sits next to its original and they land on opposite decks (and on the same page when the original is on deck A). No-op if the track is already queued in that role. Unknown role → 400, unknown track → 404.
  - `GET /plan-recs?role=…` returns a `ColumnRecsResponse` for one role column, scored against the **rest of the plan** (the tail of the *other* role queues, resolved to `stem_files.id`) plus the plan's journey (`plan_sequence`) so far. Tracks already queued in that role are excluded. For `role=song`, other versions of the queued songs come first (up to half of `k`), marked with `version_of`, `version_of_title` and `version_label`; a second copy of a queued song is not offered. Reuses the live per-column recommender. Plan-queue helpers live in [`src/dance/core/set_queues.py`](../src/dance/core/set_queues.py).
- `POST /sets/{id}/tracks` appends to the end when `position` is null; otherwise inserts at that position and shifts the rest. Reorder via `PATCH .../tracks/{track_id}` with a `position` uses a sentinel-renumber to avoid violating `(set_id, position)` uniqueness mid-flush.
- `DELETE .../tracks/{track_id}` compacts positions above the gap to stay contiguous `0..N-1`.
- **`set_tracks` and `tail-recs` are legacy** — the `set_tracks` table and the `/sets/{id}/tracks*` + `/sets/{id}/tail-recs` endpoints still exist in the backend but are no longer used by the UI. The plan-grid model (above) is what the Booth and Set views drive. `tail-recs` ranks every other track by arc-fit against the trailing `window` tracks (default 5): weighted-average embedding, Camelot key compat, BPM band median, energy slope projection. Weights live in [`src/dance/recommender/tail.py`](../src/dance/recommender/tail.py) (`_W_EMBED/_W_KEY/_W_BPM/_W_ENERGY`). `exclude_session_plays=true` also drops tracks played in the currently-open `DjSession`.

---

## Ableton — `src/dance/api/routers/ableton.py`

All endpoints are fire-and-forget OSC sends except `/state`, `/load-track`, `/decks*`, `/preview`, and `/transport/stop-scene` (which wait or return structured data).

**Transport + state:**

| Method | Path                                            | Body                | Response              | 4xx |
|--------|-------------------------------------------------|---------------------|-----------------------|-----|
| POST   | `/api/v1/ableton/play`                          | -                   | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/stop`                          | -                   | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/tempo`                         | `TempoRequest`      | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/fire`                          | `FireClipRequest`   | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/volume`                        | `VolumeRequest`     | `{"ok": true}`        |     |
| GET    | `/api/v1/ableton/state`                         | -                   | `AbletonStateOut`     |     |
| POST   | `/api/v1/ableton/transport/fire-scene/{idx}`    | -                   | `{"ok": true, ...}`   |     |
| POST   | `/api/v1/ableton/transport/fire-deck/{side}/{idx}` | query: `loaded_only` (bool) | `{ok, side, scene_index, fired}` | 400 |
| POST   | `/api/v1/ableton/transport/stop-deck/{side}`    | -                   | `{ok, side, stopped}` | 400 |
| POST   | `/api/v1/ableton/transport/fire-clip/{t}/{s}`   | -                   | `{"ok": true, ...}`   |     |
| POST   | `/api/v1/ableton/transport/stop-cell/{t}/{s}`   | -                   | `{"ok": true, ...}`   |     |
| POST   | `/api/v1/ableton/transport/stop-scene/{idx}`    | -                   | `{"ok": true, ...}`   |     |
| POST   | `/api/v1/ableton/transport/stop-track/{idx}`    | -                   | `{"ok": true, ...}`   |     |
| POST   | `/api/v1/ableton/transport/stop-all`            | -                   | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/transport/seek/{t}/{s}`        | query: `position` (beats) | `{"ok": true, ...}` |     |
| POST   | `/api/v1/ableton/transport/solo-track/{idx}`    | query: `soloed` (bool) | `{"ok": true, ...}`   |     |

**Decks + loading + cue/preview:**

| Method | Path                                  | Body                | Response                | 4xx |
|--------|---------------------------------------|---------------------|-------------------------|-----|
| POST   | `/api/v1/ableton/load-track`          | `LoadTrackRequest`  | `LoadTrackResult`       | 404, 503 |
| GET    | `/api/v1/ableton/decks`               | -                   | `DeckMapOut`            |     |
| POST   | `/api/v1/ableton/decks/reset`         | -                   | `{"ok": true}`          |     |
| POST   | `/api/v1/ableton/decks/clean`         | -                   | `{"ok": true, ...}`     |     |
| POST   | `/api/v1/ableton/decks/resync`        | -                   | `{ok, scanned, adopted, unmatched}` |     |
| DELETE | `/api/v1/ableton/decks/cell/{t}/{s}`  | -                   | `{ok, kind, removed_track_id}` |     |
| POST   | `/api/v1/ableton/preview`             | `PreviewRequest`    | `PreviewResult`         | 400, 404, 503 |
| POST   | `/api/v1/ableton/preview/stop`        | -                   | `PreviewResult`         |     |

`/state` returns the latest snapshot held by `AbletonBridge` — tempo/beat/is_playing + per-track playing clip + volume + output-meter level + per-clip `playing_position` in beats (subscribed on deck columns, drives the per-stem waveform playhead in ComboStrip). If Live isn't running, the fields are `null`.

`/transport/seek/{track}/{slot}?position={beats}` sets BOTH the clip's `loop_start` and `start_marker` to `position` (in beats) and re-fires. Setting both is required for looping clips so playback doesn't wrap back to 0 on loop_end — the loop region moves with the seek. Wired to ComboStrip's click-to-scrub gesture (snaps to the section the click landed in before sending).

`/transport/solo-track/{idx}?soloed=true|false` toggles Live's `track.solo` for the deck column. Per-column "S" buttons in SceneGrid headers use this to route a single stem through Live's Cue (PFL) bus → Scarlett 4i4 outs 3/4 → headphones, master untouched.

`/decks/cell/{track}/{slot}` (DELETE) stops the clip if playing, deletes it from the slot, drops the cell from the bridge's `_deck_cells` cache, and persists. The slot itself stays — it just becomes empty, ready for the next `Load to Live` of that kind. Wired to the hover-`×` button on each occupied SceneGrid cell.

`/load-track` puts a song on a page (five scene rows) of Live's deck columns as cue pads: five clips per column at the song's five cues, with the start and loop markers, warp mode, colours and tempo grid a prepared Set gives them. `kinds=None` loads the four stems and the mix; `kinds=["drums"]` loads one stem into its own column. `page` is the page the Booth shows; the load takes the first page from there on whose deck columns are empty in Live, never replacing a clip, with deck A tried before deck B when `side` is omitted. `scene_index` without `page` starts at the page holding that scene. The result carries `page`, `side` and `scene_index` (the page's first row). A song with no pads loads each stem whole on the first row with a warning, and a song with no tempo analysis loads whole stems on a free row with Live's own warp. The call returns once Live has made the clips: under a second for files Live has never seen or already has open, about 3 s the first time in a session for a song Live has auto-warped before. Loads take turns. If Live does not answer, nothing is loaded and `stems_loaded` is 0. 503 on OSC errors.

`/warp-check/{scene}` reads back the warp grid of every cue pad the app loaded on that scene's page, rewrites any that no longer hold the song's tempo, and warns only about pads on which Live would not take it. The bridge runs the same check by itself for 30 seconds after each load.

`/transport/fire-deck/{side}/{idx}` fires the five tracks of one deck (four stems and the mix) at one scene. With `loaded_only=true` it fires only the cells the deck map holds at that scene: firing an empty slot presses that track's stop button in Live, so the Booth's cue buttons send it.

`/decks` is the canonical view of "which (scene, kind) cells have which source-track loaded". Returns `cells` with stem_file_id populated. Each cell also has `cue`, `cue_label` and `cue_silent`. A cell is a cue when Live's start marker for that clip sits on the pad that belongs to its row: five rows make a page, and row *n* of a page is pad *n* of the song (`GET /tracks/{id}/pads`). A clip that starts anywhere else has `cue: null`, including a pad moved in the app after the Set was written. The FE refetches when `deck_map_revision` changes and every 10 s as a fallback.

`/decks/resync` is non-destructive. It finds the ten deck tracks by name, asks each for all its clip names in one query, then reads every clip's `start_marker` and `file_path`. Each clip name (`"{title} ({kind} {side})"`) is matched to a `Track` by title; when more than one track has that title, the clip's audio file decides. It rebuilds `_deck_cells` and returns counts plus any unmatched clips. The backend runs the same adoption by itself whenever Live opens a Set (AbletonOSC sends `/live/startup`), so the route is for a session that drifted, such as clips moved by hand in Live.

`/preview` auditions a track or stem through a dedicated **Cue** track in Live whose output is routed to Scarlett 4i4 outs 3/4 (headphones, not master). The Cue track is created lazily on first preview. `column="mix"` previews the full-track audio; stem columns preview just that stem. 404 when no audio file resolves.

`/decks/clean` deletes every `Deck *` + `Cue` track in Live and resets bridge state. Use as a nuclear option when things have drifted irrecoverably.

**Bridge persistence**: deck columns / cells / cue-track index are persisted to `{settings.data_dir}/deck_state.json` atomically on every mutation and restored on bridge `start()`. A backend restart no longer loses state.

---

## Files — `src/dance/api/routers/files.py`

| Method | Path                | Body              | Response               | 4xx |
|--------|---------------------|-------------------|------------------------|-----|
| POST   | `/api/v1/files/reveal` | `{"path": str}` | `{"ok": true, "command": str}` | 400, 403, 404, 500 |
| POST   | `/api/v1/files/open-in-live` | `{"path": str}` (an `.als` inside `als_output_dir`) | `{"ok": true, "app": str}`: opens the Set in the installed Live Standard or Suite, named by its app path so Lite is never picked | 400, 403, 404, 409 (no usable Live found) |

Allowlist: `path` must live under `library_dir`, `stems_dir`, or `als_output_dir` (`files.py:26`) — otherwise 403. Uses `open -R` (macOS), `explorer /select,` (Windows), or `xdg-open` (Linux).

---

## WebSocket

| Path  | Direction        | Payload                |
|-------|------------------|------------------------|
| `/ws` | server -> client | `AbletonStateOut` JSON |

On connect, the server sends one snapshot immediately. After that, it pushes the full state when `AbletonBridge` observes a change (tempo, beat, playing clip, volume, meters, clip positions), coalesced to at most ~30 frames/s: the newest state wins, and the state after the last change of a burst is always sent (`api/broadcast_throttle.py`). Client messages are read but ignored — the loop is there only to notice disconnects (`ws.py:65`).

Reconnect is the client's responsibility. In the companion app every `useAbletonState` caller shares one socket (`companion-app/src/lib/sharedSocket.ts`), which reconnects with a 2 s backoff and re-renders at most once per animation frame.

Sample payload:

```json
{
  "tempo": 128.0,
  "is_playing": true,
  "beat": 64.25,
  "playing_clips": {"0": 2, "1": 2, "2": -1},
  "track_volumes": {"0": 0.85, "1": 0.85},
  "track_meters": {"5": 0.42, "6": 0.0, "7": 0.18},
  "playing_positions": {"5": 128.5, "7": 64.0}
}
```

`track_meters` and `playing_positions` are subscribed only on deck-column tracks (typically tracks 5-9). `playing_positions` values are in clip-beats indexed against the clip's nominal BPM — i.e. the source track's analyzed BPM, NOT Live's project tempo. Consumers converting to file-seconds should use `cell.bpm`, not `tempo`.

---

## Origin guard and CORS

CORS alone doesn't protect a local API: it only hides responses from a foreign page, it doesn't stop a "simple" cross-origin POST (no body, `text/plain`, form) from reaching `/ableton/transport/stop-all`, and it doesn't apply to websockets or DNS rebinding at all. So `create_app` puts `OriginGuardMiddleware` (`src/dance/api/origin_guard.py`) in front of everything, HTTP and websocket alike:

| Request | Result |
|---|---|
| `Host` not loopback (`127.0.0.1`, `localhost`, `::1`, any port) | 403 — stops DNS rebinding |
| `Origin` present and not allowlisted, any method (GET included) | 403 / websocket refused with a 403 handshake |
| `Origin` same as the `Host` it arrived on (FastAPI's own `/docs`) | allowed |
| No `Origin` (curl, scripts, the CLI) | allowed |

The allowlist is the companion's Vite origins — `http://localhost:5173`, `5174`, and the `127.0.0.1` variants — and doubles as the CORS allowlist. The companion app always passes: Vite proxies `/api` and `/ws`, so the backend sees the page's own allowlisted origin. Cross-origin GETs are blocked too because several GETs do real work (waveforms, Spotify search, OSC queries).

Both lists are settings. Values are comma-separated and **replace** the defaults:

```bash
# a second companion on port 5175
DANCE_API_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5175
# serving the API to another device on purpose (also needs --host 0.0.0.0)
DANCE_API_ALLOWED_HOSTS=127.0.0.1,localhost,::1,dance-mac.local
```

`*` turns that check off. Blocked requests log `origin guard: blocked ...` at WARNING with the setting to change.
