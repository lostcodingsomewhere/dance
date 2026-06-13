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

## Demucs separation crashes (SIGBUS / exit 138) on Python 3.14

On this machine's off-spec **Python 3.14** venv, Demucs stem-separation can die with a **bus error (SIGBUS)** — the process exits with code **138** (128 + signal 10) and the track makes no progress (no stems produced, the run bounces straight to the next track or to an `error` state). This is a native crash in the torch/Demucs stack under the non-standard interpreter, not a bug in the pipeline code.

**Workaround — run separation on CPU:**

```bash
DANCE_DEMUCS_DEVICE=cpu dance process
```

CPU separation is slower (a few minutes per track instead of ~1–3) but stable — it sidesteps the MPS path that triggers the crash. Also:

- **Reduce concurrent load** — separate fewer tracks at a time (e.g. `dance process -n 1`) so you're not running multiple Demucs processes against memory pressure at once.
- **Make sure ffmpeg is installed** (`brew install ffmpeg`) — a missing/half-broken ffmpeg can surface as an abrupt native exit during decode rather than a clean error.

If you need MPS speed, the durable fix is to run the pipeline on a spec-compliant Python 3.10 interpreter (see the Python-3.14 memory note); CPU separation is the right move when you just need the run to finish.

## OSC firewall on macOS

The first time the backend starts and the OSC listener binds to UDP `127.0.0.1:11001`, macOS may prompt:

> "Do you want the application 'python' to accept incoming network connections?"

Click **Allow**. If you missed the dialog or accidentally clicked Deny, open **System Settings -> Network -> Firewall -> Options** and remove the Python entry, then restart the backend to re-trigger the prompt. Loopback-only (`127.0.0.1`) traffic should not actually require permission, but in practice macOS sometimes blocks it anyway.

Symptom: the backend starts cleanly, `/api/v1/ableton/state` returns all `null`s, and AbletonOSC's status panel in Live shows no incoming/outgoing traffic. Check the firewall first.

## `spotdl` rate limiting / Spotify Web API restrictions (Nov 2024+)

`dance sync` uses spotDL, which calls Spotify's Web API to enumerate playlist tracks. Two failure modes, in escalating severity:

**1. Shared-default-token hangs** (mild — fixable by registering your own app).

The spotDL default `client_id = 5f573c9620494bae87890c0f08a60293` is shared by every spotDL install in the world. Spotify rate-limits it aggressively. Symptom: `dance sync` runs for 30+ minutes with zero log output and 0.1% CPU — spotDL is hung waiting on retries that never succeed. `lsof -p <pid>` shows many TCP connections in `CLOSE_WAIT`.

Fix: create your own Spotify dev app (https://developer.spotify.com/dashboard) and put your `client_id` / `client_secret` in `~/.spotdl/config.json`. Restart sync.

**2. Spotify deprecated `/v1/playlists/{id}/tracks` for new apps (CATASTROPHIC — no fix within spotDL).**

On **2024-11-27** Spotify restricted dozens of Web API endpoints to apps with "Extended Quota Mode" approval. New apps are stuck in Development Mode by default and the following endpoints return **HTTP 403 Forbidden, reason: None** regardless of auth flow (client-credentials OR user-auth with `playlist-read-private` scope):

- `Get a Playlist's Items` (the one spotDL calls on every sync)
- `Get Recommendations`
- `Get Audio Features / Audio Analysis`
- `Get Featured Playlists`
- Related-Artists / category playlists / etc.

Extended Quota Mode requires a formal Spotify review with stated commercial intent. Hobbyist requests have been routinely rejected since the policy change. **There is currently no way to make a new spotDL install talk to a new Spotify dev app.**

Diagnostic: direct curl against the API isolates the block.

```bash
TOKEN=$(curl -sS -X POST "https://accounts.spotify.com/api/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -u "<your_id>:<your_secret>" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# /playlists/{id}        -> HTTP 200 (works on new apps)
# /playlists/{id}/tracks -> HTTP 403 (broken on new apps since 2024-11-27)
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.spotify.com/v1/playlists/<id>/tracks?limit=5"
```

**Workaround in this repo: bypass Spotify entirely.** There is no `dance ingest-csv` command — the CSV path lives in two places:

- **`python scripts/yt_dlp_csv_import.py`** — the CSV → yt-dlp downloader. Export the playlist to CSV via https://exportify.net (which uses an old grandfathered Spotify app and still works), then feed the CSV to this script for the actual audio download from YouTube Music. The library dir ends up populated the same way `dance sync` would have done, and the rest of the pipeline runs unchanged.
- **The companion app's Spotify/CSV ingest flow** — the same import surfaced in the UI.

Once the library dir is populated (by either path, or by dropping local files — see [below](#loading-audio-from-a-local-folder-no-spotify-needed)), run `dance process` (or the new **`dance ingest`** wrapper, which chains sync → process → tag → build-graph → export-als) to advance everything through the pipeline.

> ⚠️ **YouTube has its own anti-bot wall (2025-2026) — but it's solvable.** Confirmed working on 2026-05-16. All five pieces below are required; missing any one returns "Only storyboard images are available" or HTTP 403 from googlevideo.com.
>
> **One-time setup:**
>
> 1. Latest yt-dlp: `pip install -U yt-dlp` (needs 2026.03+).
> 2. Node.js on a brew-installed path (NOT just nvm — yt-dlp's PATH lookup at process-spawn time misses nvm dirs): `brew install node`.
> 3. EJS challenge solver: `pip install yt-dlp-ejs`.
> 4. bgutil PO Token plugin + server:
>    ```bash
>    pip install bgutil-ytdlp-pot-provider
>    docker run --name bgutil-provider -d --restart unless-stopped \
>      -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider:latest
>    # verify: curl http://127.0.0.1:4416/ping  →  {"server_uptime":...,"version":"1.3.1"}
>    ```
> 5. Be logged in to YouTube on Chrome (any profile works; pick the freshest).
>
> **The working command** — every flag is load-bearing:
>
> ```bash
> yt-dlp \
>   --cookies-from-browser "chrome:Profile 2" \
>   --js-runtimes node \
>   --ffmpeg-location ~/.spotdl/ffmpeg \
>   --extract-audio --audio-format mp3 --audio-quality 0 \
>   --postprocessor-args "ffmpeg:-b:a 320k" \
>   --no-playlist \
>   -o "/Users/arya/Music/DJ/library/%(uploader)s - %(title)s.%(ext)s" \
>   "ytsearch1:Artist Title"
> ```
>
> **The one non-obvious flag is `--js-runtimes node`.** yt-dlp 2026 does NOT auto-discover JS runtimes — `YoutubeDL.py:876` only enables runtimes present in `params['js_runtimes']`, which defaults to empty. Without this flag, debug shows `JS runtimes: none` and `node (unavailable)` even when `node` is on PATH and `yt-dlp-ejs` is installed.
>
> **Verify the chain is healthy** — debug log should show:
>
> ```
> [debug] JS runtimes: node-X.Y.Z
> [debug] [youtube] [pot] PO Token Providers: bgutil:http-1.3.1 (external), ...
> [debug] [youtube] [jsc] JS Challenge Providers: ..., node, ...   ← no "(unavailable)" on node
> [youtube] [pot:bgutil:http] Generating a gvs PO Token for web_safari client via bgutil HTTP server
> [youtube] [jsc:node] Solving JS challenges using node
> ```
>
> YouTube rotates its anti-bot every few months. When it breaks: `pip install -U yt-dlp bgutil-ytdlp-pot-provider yt-dlp-ejs && docker pull brainicism/bgutil-ytdlp-pot-provider:latest && docker restart bgutil-provider`. For tracks you intend to mix repeatedly, buy them on Beatport / Bandcamp (stable + legal) — save yt-dlp for "fill out the long-tail library" use.

## Loading audio from a local folder (no Spotify needed)

For tracks you already own (purchased downloads, ripped CDs, existing library):

```bash
# Drop or symlink files into the library dir
ln -sf "/Users/arya/Music/Music/My Music/Daft Punk - One More Time.mp3" \
       "/Users/arya/Music/DJ/library/"

# Or, en masse:
cd "/Users/arya/Music/Music/My Music" && \
  find . -maxdepth 1 \( -name "*.mp3" -o -name "*.m4a" -o -name "*.flac" \) \
  -print0 | while IFS= read -r -d '' f; do
    ln -sf "$(pwd)/${f#./}" "/Users/arya/Music/DJ/library/${f#./}"
  done

# Now run the pipeline — dance process auto-ingests new files via dispatcher.ingest()
dance process
```

`IngestStage` (`src/dance/pipeline/stages/ingest.py`) is content-hash based (file size + first 1 MB + last 1 MB SHA256), so symlinks and renames don't cause duplicate rows. Files inside `library_dir` are tracked; files anywhere else are ignored.

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
