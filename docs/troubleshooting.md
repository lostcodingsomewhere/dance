# Troubleshooting

Friction points you'll hit, in roughly the order you'll hit them.

## Model weights download on first run

The first time you run `dance process`, two model archives stream from Hugging Face / torch.hub:

- **Demucs** (`htdemucs_ft`, the `dance.config.Settings.demucs_model` default) — ~80 MB. Cached under `~/.cache/torch/hub/checkpoints/`.
- **CLAP** (`laion/clap-htsat-unfused`) — ~500 MB. Cached under `~/.cache/huggingface/hub/`.

On a slow connection the first track can take 5-10 minutes before any audio work happens. The log line `Loading CLAP model laion/clap-htsat-unfused on ...` (or `Loading Demucs model ...`) appears once per process — if those lines hang, the download is stuck. Bandwidth-limit your terminal, retry, or set `HF_HUB_OFFLINE=1` after a manual fetch.

If you've enabled the deep tagger (`DANCE_DEEP_TAGGER_ENABLED=true`), the first `dance tag --deep` call also pulls `Qwen/Qwen2-Audio-7B-Instruct` (~8 GB) to the same HF cache.

To pre-warm offline:

```bash
python -c "from transformers import ClapModel, ClapProcessor; ClapModel.from_pretrained('laion/clap-htsat-unfused'); ClapProcessor.from_pretrained('laion/clap-htsat-unfused')"
```

## MPS / Apple Silicon quirks

`pick_device("auto")` in `src/dance/pipeline/utils/device.py:10` prefers MPS on Apple Silicon. Both Demucs and CLAP load to MPS first; if the cast raises (`RuntimeError` or `NotImplementedError`, common for some HF ops), they fall back to CPU and log a warning:

```
WARNING  MPS load failed (...); falling back to CPU
```

To verify what each stage chose, search the backend startup logs for `Loading CLAP model ... on mps` / `... on cpu` and `Loading Demucs model ... on mps`. There is no `dance llm-status` command — the logs are the source of truth.

To force CPU for a session:

```bash
DANCE_CLAP_DEVICE=cpu DANCE_DEMUCS_DEVICE=cpu dance process
```

The deep tagger's Qwen2-Audio is the most fragile on MPS — see `src/dance/llm/qwen_audio.py:144` for the special-case load order (CPU first, then `model.to("mps")`). If it crashes, fall back via `DANCE_DEEP_TAGGER_DEVICE=cpu`.

## OSC firewall on macOS

The first time the backend starts and the OSC listener binds to UDP `127.0.0.1:11001`, macOS may prompt:

> "Do you want the application 'python' to accept incoming network connections?"

Click **Allow**. If you missed the dialog or accidentally clicked Deny, open **System Settings -> Network -> Firewall -> Options** and remove the Python entry, then restart the backend to re-trigger the prompt. Loopback-only (`127.0.0.1`) traffic should not actually require permission, but in practice macOS sometimes blocks it anyway.

Symptom: the backend starts cleanly, `/api/v1/ableton/state` returns all `null`s, and AbletonOSC's status panel in Live shows no incoming/outgoing traffic. Check the firewall first.

## `spotdl` rate limiting / auth

`dance sync` uses spotDL, which authenticates against Spotify using a shared default app token. Under load (rapid syncs, large playlists), Spotify rate-limits that token aggressively:

```
spotdl.utils.spotify.SpotifyError: HTTP Error for GET ... 401 Unauthorized
```

Fix: register your own Spotify app at https://developer.spotify.com/dashboard and configure spotDL with your client ID/secret:

```bash
spotdl --client-id <id> --client-secret <secret> ...
```

Or set them globally for spotDL via its `~/.spotdl/config.json`. The `dance sync` wrapper does not currently surface these flags — set them on the spotDL config file and they're picked up.

## `Maximum allowed size exceeded` / `OverflowError` in librosa

You'll see scary stack traces during the first `analyze` or `analyze_stems` pass:

```
OverflowError: Maximum allowed size exceeded
  File ".../numba/core/typeinfer.py", ...
```

These are **Numba JIT compilation warnings, not errors**. librosa's first call into a Numba-accelerated function (`librosa.beat.beat_track`, etc.) triggers JIT compilation, and Numba's type inference logs spurious overflow attempts as it explores type promotions. The traces look identical to real exceptions.

The actual error (if any) appears **after** the JIT noise. If the stage's `Track.state` advances and `dance status` shows the count moving, the JIT noise was cosmetic — ignore it.

If you actually want to silence it: `export NUMBA_DISABLE_JIT=1`, but you'll lose ~5-10x throughput on the analyzers.

## Live "missing media" prompt when opening .als

The generated `.als` references stems by **absolute path** (`src/dance/als/writer.py` — `_add_file_ref`). Live shows a "Missing Media" dialog if any of those paths don't resolve.

Causes:

- You moved or renamed `stems_dir` between `dance export-als` and `open <file>.als`.
- You're opening the Set on a different machine than the one that generated it.
- The stems are on an external drive that's not mounted.

Fix: either re-run `dance export-als <track_id>` to regenerate with current paths, or let Live's "Locate Files" dialog hunt — it usually finds the files if the basename matches.

To audit a Set's referenced paths without opening Live:

```bash
gunzip -c "~/Music/Dance/Sets/Title - Artist.als" | grep -E '<Path Value=' | head
```

## Live rejects the .als entirely

If Live fails to load the Set (not "missing media" but a hard parse error), `src/dance/als/writer.py` is mis-injecting into the template (`src/dance/als/templates/blank_live12.xml`).

What to do:

1. Capture the exact Live error — Live 12 typically gives a line and column in the decompressed XML.
2. Unzip and inspect:
   ```bash
   gunzip -c bad.als > bad.xml
   ```
3. Read the indicated line; cross-check against the bundled template for what shape Live expects. Common gotchas:
   - Class elements (e.g. `<TimeSignature>`, `<FollowAction>`) emitted as leaves with a `Value` attribute.
   - Duplicate Pointee IDs after a deepcopy that wasn't renumbered (`_renumber_pointees` in `writer.py`).
   - Tempo only written to `MainTrack/.../Tempo/Manual` — Live reads the `AutomationEnvelope FloatEvent` anchor in preference. Update both.
4. If you upgraded Live and the bundled Live-12.4 template no longer loads, save a fresh blank Set as `Untitled.als` from your version, then:
   ```bash
   gunzip -c ~/Desktop/Untitled.als > src/dance/als/templates/blank_live12.xml
   ```

Tests (`tests/test_als_generator.py`) only validate well-formedness and shape — they cannot assert "Live will accept this." Only Live can.

## "track not found" / 404 on every API call

`get_session` in `src/dance/api/deps.py:37` reads `app.state.session_factory`. If the backend was started without the DB initialized (rare — `create_app` reads from `settings.db_url`), or if you pointed it at a different `DANCE_DATA_DIR` than `dance process` writes to, you'll see empty results everywhere.

Check the actual DB the backend is using by hitting `/api/v1/tracks?limit=1` and comparing with:

```bash
DANCE_LOG_LEVEL=DEBUG uvicorn dance.api:create_app --factory
# look for "create_engine sqlite:///..." in the logs
```

## Companion app shows blank "Now Playing"

`NowPlaying.tsx` currently keys off the most recent `SessionPlay` row (`companion-app/src/views/NowPlaying.tsx:21-24`). If you have no active DJ session, it'll be empty — start one via the Top Bar's "New Session" button or `POST /api/v1/sessions`.

Mapping a *Live-playing clip* back to a track requires a clip-to-track map the backend doesn't yet expose. See the inline comment at `NowPlaying.tsx:21` — this is a known Phase 2.4+ gap.

## WebSocket disconnects on every code change

Expected. Vite's HMR closes and reopens the page; `useAbletonState` reconnects with a 2 s backoff (`companion-app/src/hooks/useAbletonState.ts:35`). If the reconnect never succeeds, the backend isn't running — check `http://127.0.0.1:8000/health`.
