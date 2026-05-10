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
| GET    | `/api/v1/tracks/{id}`                 | -                                                                            | `TrackOut`                        | 404 |
| GET    | `/api/v1/tracks/{id}/regions`         | `region_type, stem_file_id`                                                  | `list[RegionOut]`                 | 404 |
| GET    | `/api/v1/tracks/{id}/stems`           | -                                                                            | `list[StemFileOut]`               | 404 |
| POST   | `/api/v1/tracks/{id}/tag`             | query: `deep` (bool)                                                         | `TrackOut`                        | 404, 502, 503 |
| POST   | `/api/v1/tracks/{id}/als`             | `AlsExportRequest` (`out_path?`)                                             | `AlsExportResult`                 | 400, 403, 404 |

Notes:

- `GET /tracks` joins `audio_analysis` only when an analysis filter is provided (`tracks.py:51`).
- `key` is the Camelot code (`"8A"`, `"3B"`, ...), uppercased.
- `tag?deep=true` requires `DANCE_DEEP_TAGGER_ENABLED=true` and returns 503 otherwise (`tracks.py:126`).
- `tag` returns 502 if the tagger raises (model load failure, audio missing, etc.).
- `als` returns 403 when `out_path` resolves outside `settings.als_output_dir` (`tracks.py:175`), 400 for missing analysis / no stems / not COMPLETE.

---

## Recommend — `src/dance/api/routers/recommend.py`

| Method | Path                                  | Body / Query                                       | Response                          | 4xx |
|--------|---------------------------------------|----------------------------------------------------|-----------------------------------|-----|
| POST   | `/api/v1/recommend`                   | `RecommendRequest`                                 | `list[RecommendationOut]`         | 400 |
| GET    | `/api/v1/recommend/by-seed/{id}`      | `k` (query, default 10)                            | `list[RecommendationOut]`         |     |
| POST   | `/api/v1/recommend/text`              | `TextRecommendRequest`                             | `list[RecommendationOut]`         | 400, 503 |

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

## Ableton — `src/dance/api/routers/ableton.py`

All endpoints are fire-and-forget OSC sends except `/state` and `/load-track`.

| Method | Path                                  | Body                | Response              | 4xx |
|--------|---------------------------------------|---------------------|-----------------------|-----|
| POST   | `/api/v1/ableton/play`                | -                   | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/stop`                | -                   | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/tempo`               | `TempoRequest`      | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/fire`                | `FireClipRequest`   | `{"ok": true}`        |     |
| POST   | `/api/v1/ableton/volume`              | `VolumeRequest`     | `{"ok": true}`        |     |
| GET    | `/api/v1/ableton/state`               | -                   | `AbletonStateOut`     |     |
| POST   | `/api/v1/ableton/load-track`          | `LoadTrackRequest`  | `LoadTrackResult`     | 404, 503 |

`/state` returns the latest snapshot held by `AbletonBridge` — last observed tempo/beat/is_playing + per-track playing clip + volume. If Live isn't running, the fields are `null`.

`/load-track` creates empty named/colored audio tracks in Live (1 for mix + 1 per stem) and returns their indices. Cannot actually load samples (Live API limitation); the React UI typically follows up with `POST /files/reveal` so the user can drag the stems in. 503 when the OSC send raises `OSError`.

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
  "track_volumes": {"0": 0.85, "1": 0.85}
}
```

---

## CORS

`create_app` (`app.py:80`) allows origins `http://localhost:5173`, `5174`, and the `127.0.0.1` variants — Vite's default and `vite --port 5174`. Anything else is blocked; edit `app.py` if you serve the UI elsewhere.
