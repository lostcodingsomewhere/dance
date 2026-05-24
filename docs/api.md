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
| GET    | `/api/v1/tracks/{id}/regions`         | `region_type, stem_file_id`                                                  | `list[RegionOut]`                 | 404 |
| GET    | `/api/v1/tracks/{id}/stems`           | -                                                                            | `list[StemFileOut]`               | 404 |
| GET    | `/api/v1/tracks/{id}/waveform`        | `num_peaks` (query, default 200, range 50-1000)                              | `WaveformOut`                     | 404, 422, 503 |
| GET    | `/api/v1/stems/{stem_file_id}/waveform` | `num_peaks` (query, default 200, range 50-1000)                            | `WaveformOut`                     | 404, 422, 503 |
| POST   | `/api/v1/tracks/{id}/tag`             | query: `deep` (bool)                                                         | `TrackOut`                        | 404, 502, 503 |
| POST   | `/api/v1/tracks/{id}/als`             | `AlsExportRequest` (`out_path?`)                                             | `AlsExportResult`                 | 400, 403, 404 |

Notes:

- `GET /tracks` joins `audio_analysis` only when an analysis filter is provided (`tracks.py:51`).
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

`/recommend/by-column` is the live-remixing rec stream: takes a `column` (`drums` / `bass` / `vocals` / `other` / `mix`) plus the active combo's `combo_stem_ids` + `master_bpm` and returns top-K candidates filtered by stem kind, scored against the combo via per-stem embedding cosine + Camelot key compat + BPM proximity. Returns 400 for unknown column. Used by the FE per-column banners — one query per column, re-run on combo change.

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

Persistent named track plans (the **Set Rail** backing store). Distinct from `DjSession`: a Set is the *intent* (planned), a Session is the *history* (what actually played). At most one Set is `is_active = true` at a time (enforced by a partial unique index in `_create_partial_unique_indexes`).

| Method | Path                                              | Body / Query                              | Response                | 4xx |
|--------|---------------------------------------------------|-------------------------------------------|-------------------------|-----|
| GET    | `/api/v1/sets`                                    | -                                         | `list[SetSummaryOut]`   |     |
| GET    | `/api/v1/sets/active`                             | -                                         | `SetOut`                | 404 |
| POST   | `/api/v1/sets`                                    | `SetCreateRequest`                        | `SetOut`                |     |
| GET    | `/api/v1/sets/{id}`                               | -                                         | `SetOut`                | 404 |
| PATCH  | `/api/v1/sets/{id}`                               | `SetUpdateRequest`                        | `SetOut`                | 404 |
| DELETE | `/api/v1/sets/{id}`                               | -                                         | 204                     | 404 |
| POST   | `/api/v1/sets/{id}/activate`                      | -                                         | `SetOut`                | 404 |
| POST   | `/api/v1/sets/{id}/tracks`                        | `SetTrackAddRequest`                      | `SetOut`                | 400, 404 |
| PATCH  | `/api/v1/sets/{id}/tracks/{track_id}`             | `SetTrackUpdateRequest`                   | `SetOut`                | 400, 404 |
| DELETE | `/api/v1/sets/{id}/tracks/{track_id}`             | -                                         | `SetOut`                | 404 |
| GET    | `/api/v1/sets/{id}/tail-recs`                     | `k, window, exclude_session_plays`        | `TailRecsResponse`      | 404 |

- `activate` is atomic: deactivates other sets in the same transaction, then sets the target — the partial unique index never sees two active rows.
- `POST /sets/{id}/tracks` appends to the end when `position` is null; otherwise inserts at that position and shifts the rest. Reorder via `PATCH .../tracks/{track_id}` with a `position` uses a sentinel-renumber to avoid violating `(set_id, position)` uniqueness mid-flush.
- `DELETE .../tracks/{track_id}` compacts positions above the gap to stay contiguous `0..N-1`.
- `tail-recs` ranks every other track by arc-fit against the trailing `window` tracks (default 5): weighted-average embedding, Camelot key compat, BPM band median, energy slope projection. Weights live in [`src/dance/recommender/tail.py`](../src/dance/recommender/tail.py) (`_W_EMBED/_W_KEY/_W_BPM/_W_ENERGY`). `exclude_session_plays=true` also drops tracks played in the currently-open `DjSession` so the rail doesn't re-suggest what's already on tonight.

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

`/load-track` populates clip slots in Live's deck-column session view. `kinds=None` loads all 4 stems into a fresh row (anchor mode); `kinds=["drums"]` loads just one stem into the next free drums slot (live-remixing mode). 503 on OSC errors.

`/decks` is the canonical view of "which (scene, kind) cells have which source-track loaded". Returns `cells` with stem_file_id populated. Polled by the FE every 2 s.

`/decks/resync` is non-destructive: walks each deck-column's first 16 scenes via `/live/clip/get/name` queries, parses each clip name (`"{title} ({kind})"`), matches to a `Track` by title, rebuilds `_deck_cells`. Returns counts + any unmatched clips so the UI can surface them. Use after a backend restart drifted state from Live's actual session.

`/preview` auditions a track or stem through a dedicated **Cue** track in Live whose output is routed to Scarlett 4i4 outs 3/4 (headphones, not master). The Cue track is created lazily on first preview. `column="mix"` previews the full-track audio; stem columns preview just that stem. 404 when no audio file resolves.

`/decks/clean` deletes every `Deck *` + `Cue` track in Live and resets bridge state. Use as a nuclear option when things have drifted irrecoverably.

**Bridge persistence**: deck columns / cells / cue-track index are persisted to `{settings.data_dir}/deck_state.json` atomically on every mutation and restored on bridge `start()`. A backend restart no longer loses state.

---

## Files — `src/dance/api/routers/files.py`

| Method | Path                | Body              | Response               | 4xx |
|--------|---------------------|-------------------|------------------------|-----|
| POST   | `/api/v1/files/reveal` | `{"path": str}` | `{"ok": true, "command": str}` | 400, 403, 404, 500 |

Allowlist: `path` must live under `library_dir`, `stems_dir`, or `als_output_dir` (`files.py:26`) — otherwise 403. Uses `open -R` (macOS), `explorer /select,` (Windows), or `xdg-open` (Linux).

---

## WebSocket

| Path  | Direction        | Payload                |
|-------|------------------|------------------------|
| `/ws` | server -> client | `AbletonStateOut` JSON |

On connect, the server sends one snapshot immediately. After that, it sends every time `AbletonBridge` observes a change (tempo, beat, playing clip, volume). Client messages are read but ignored — the loop is there only to notice disconnects (`ws.py:65`).

Reconnect is the client's responsibility. The React `useAbletonState` hook (`companion-app/src/hooks/useAbletonState.ts`) reconnects with a 2 s backoff.

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

## CORS

`create_app` (`app.py:80`) allows origins `http://localhost:5173`, `5174`, and the `127.0.0.1` variants — Vite's default and `vite --port 5174`. Anything else is blocked; edit `app.py` if you serve the UI elsewhere.
