# Dance

Stem-performance brain for **Ableton Live**.

Spotify playlist → analyzed tracks with **stems, cues, loops, tags, and graph edges**, then auto-loaded into Ableton Live as ready-to-fire stem clips via OSC, with a companion React app (in `companion-app/`) for live recommendations during the set.

The repo has three pieces:
1. **Python pipeline** (`src/dance/`) — the brain. Spotify ingest → analysis → stems → regions → embeddings → recommendation graph. SQLite is the source of truth.
2. **FastAPI backend** (`src/dance/api/`) — read-mostly REST over the SQLite DB, plus a WebSocket for live Ableton state and an OSC passthrough that auto-loads stem clips into Live's session view (Live 12.0.5+).
3. **React companion app** (`companion-app/`) — Vite + TypeScript + Tailwind. Everything centers on one surface, the **plan grid**: five role columns — Drums · Bass · Vocals · Other · Song — each stacking your queued plan picks on top and recommendations below. Three views: **Booth** (live performance — SceneGrid mirror + the plan grid in *live* mode, where recs tail what's playing in Ableton and each card has ⤒A/⤒B to load it onto a deck), **Set** (the same plan grid in *plan* mode — recs scored against the rest of the plan; ＋ to queue), **Pipeline** (ingest + processing status + library inventory). One hybrid **⌘K** palette covers fuzzy artist/title search and CLAP vibe search in one surface, appending picks to the Song column.

## Documentation

- [docs/architecture.md](docs/architecture.md) — three-layer architecture, the Stage protocol + dispatcher, schema overview
- [docs/cli.md](docs/cli.md) — every `dance` subcommand and the state it touches
- [docs/api.md](docs/api.md) — REST + WebSocket reference (auto-generated `/docs` at runtime)
- [docs/tagging.md](docs/tagging.md) — CLAP zero-shot vs Qwen2-Audio, vocabulary, tuning
- [docs/als-export.md](docs/als-export.md) — generated Live Set contents, color palette, limitations
- [docs/abletonosc_setup.md](docs/abletonosc_setup.md) — installing AbletonOSC + what works / doesn't
- [docs/troubleshooting.md](docs/troubleshooting.md) — MPS quirks, OSC firewall, spotDL auth, Live errors
- [docs/dev.md](docs/dev.md) — adding stages, endpoints, migrations; pre-commit hygiene

## Architecture

```
Spotify playlist
      ↓
   ingest (file scan, hash, metadata)
      ↓
   analyze (Essentia/librosa: BPM, key, energy, mood)
      ↓
   separate (Demucs: drums, bass, vocals, other)
      ↓
   analyze_stems (per-stem RMS, presence, BPM, pitch, kick density)
      ↓
   detect_regions (sections + cue points + loop candidates, per track and per stem)
      ↓
   embed (CLAP embeddings for full mix + each stem)
      ↓
   build_graph (track-to-track edges: harmonic, tempo, embedding-neighbor, tag-overlap)
      ↓
   SQLite DB (consumed by companion app + Ableton via AbletonOSC)
```

Each stage is an independent `Stage` object registered with the dispatcher. Stages are state-driven — a stage runs on tracks whose `state` matches its `input_state`. No central orchestrator, no hardcoded order.

## Quick start (any Mac)

```bash
./bin/setup.sh
```

That does everything: Python venv, dev deps, Alembic migrations, companion-app npm install, scaffolds `~/.dance/.env` from [`.env.example`](.env.example), and tells you which Homebrew packages still need installing (`yt-dlp`, `ffmpeg`).

After it finishes:

```bash
# Terminal 1 — backend
source .venv/bin/activate
uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000

# Terminal 2 — companion app
cd companion-app && npm run dev   # http://localhost:5173
```

The MasterStrip has a small dot top-right that turns **red** if a required host tool is missing, **amber** if only optional stuff is (Spotify creds, YouTube cookies), **green** when everything's ready. Click it for a checklist with one-line install hints.

### What goes in `~/.dance/.env`

Edit your per-user file (the setup script copies a template into place). The minimum for "Add from Spotify" in Cmd-K to work:

```bash
DANCE_SPOTIFY_CLIENT_ID=...        # https://developer.spotify.com/dashboard
DANCE_SPOTIFY_CLIENT_SECRET=...
```

For reliable YouTube downloads, export cookies via the **Get cookies.txt LOCALLY** Chrome extension and save to `~/.dance/cookies.txt`. The setup script tells you when this is missing.

### Pipeline runbook

Once setup is done and host tools are installed:

```bash
dance config --spotify-playlist "https://open.spotify.com/playlist/<id>"
dance run --once          # sync + process
dance build-graph         # build recommendation edges
```

Or use the in-UI **Add from Spotify** flow in Cmd-K for ad-hoc tracks while building a set.

Ableton state is pushed over the WebSocket — install AbletonOSC first (see [docs/abletonosc_setup.md](docs/abletonosc_setup.md)).

## DJ flow

Both views share **one surface** — the **plan grid** (`RoleColumnsGrid`): five role columns (Drums · Bass · Vocals · Other · Song; "song" is the full-track anchor) that stack your queued plan picks on top and recommendations below. The same unified rec brain scores both; only the context and the available actions differ.

### Before the set — plan in the Set view

1. **Open the companion** at `http://localhost:5173`. First-time users with a legacy localStorage Stack get a one-shot prompt to import it as a named Set.
2. **Create or switch sets** from the **Set** view (top nav). Sets are persistent and named — "Warehouse Sat", "Wedding 90min" — and a Set *is* its plan (a per-role queue stored on the set).
3. **Queue picks per role.** Each role column shows recommendations scored against the rest of the plan and the plan's journey so far. Tap **＋** on a card to queue it onto that role's stack. Use ▶ to prelisten in headphones (cue) first; the ScoreBreakdown chips explain why a pick scored where it did.
4. **Or fill via ⌘K.** The palette is hybrid: typing surfaces fuzzy artist/title matches in the **Tracks** section first, CLAP vibe matches in the **Vibe** section below (8+ char queries). BPM/key/energy chips narrow further. Selecting a result appends it to the **Song** column of the active plan.

### During the set — the Booth

1. **SceneGrid** mirrors the APC40 — tap cells to fire, tap row labels to anchor, eyes here during a mix.
2. **The plan grid in live mode** sits below: your queued plan picks on top, plus recs that tail what's playing in Ableton (combo embedding + trailing-journey trend), live-rescored against the active stem combo. Tap **⤒A**/**⤒B** on a card to load that pick onto a deck; ▶ to prelisten on the cue bus.
3. **Auto-logging** — when a clip fires that was loaded via the companion, the play is recorded to the current `DjSession`. Session play count + end-session live in the MasterStrip's SessionChip.

The **MasterStrip** (top) shows live BPM, transport, energy arc, AbletonOSC heartbeat, ⌘K, and the three view tabs. Stacked-stem **TwoDeckStrip** waveforms (click + drag to seek), the **Crossfader**, **BoothColumnHeaders**, and the **CueStrip** round out the Booth.

## Commands

| Command | What it does |
|---|---|
| `dance config --show` | Show current configuration |
| `dance sync` | Download tracks via spotDL |
| `dance process` | Run pipeline on pending tracks |
| `dance list` | Browse tracks with filters |
| `dance run --once` | Sync + process, one pass |
| `dance run` | Daemon mode |
| `dance status` | Pipeline state counts |

## Configuration

Set via `~/.dance/.env` or environment variables (prefix `DANCE_`):

```bash
DANCE_SPOTIFY_PLAYLIST_URL=https://open.spotify.com/playlist/...
DANCE_LIBRARY_DIR=~/Music/DJ/library
DANCE_STEMS_DIR=~/Music/DJ/stems
DANCE_DATA_DIR=~/.dance
DANCE_SKIP_STEMS=false
DANCE_SKIP_EMBEDDINGS=false
DANCE_CLAP_MODEL=laion/clap-htsat-unfused
DANCE_DEMUCS_MODEL=htdemucs_ft
```

## Project layout

```
src/dance/
├── cli.py                  Click commands
├── config.py               Pydantic settings
├── core/
│   └── database.py         SQLAlchemy models
├── pipeline/
│   ├── dispatcher.py       Stage registry + runner
│   ├── stage.py            Stage protocol
│   ├── stages/             One module per stage (ingest, analyze, separate, ...)
│   └── utils/              Beat/phrase utilities, Camelot wheel
├── spotify/
│   └── downloader.py       spotDL wrapper
├── recommender/
│   ├── graph_builder.py    Builds track_edges
│   ├── recommender.py      Per-column recommend() entrypoint
│   ├── scoring.py          Unified rec brain — per-role scoring
│   ├── journey.py          Trend-aware "journey" vibe context
│   └── structure.py        Transition-fit / section scoring
└── alembic/                Schema migrations
```

## Development

```bash
pytest                  # tests (uses synthetic audio fixtures)
ruff check src/dance
mypy src/dance
alembic upgrade head    # apply migrations
```

## License

MIT
