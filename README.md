# Dance

Stem-performance brain for **Ableton Live**.

Spotify playlist → analyzed tracks with **stems, cues, loops, tags, and graph edges**, then auto-loaded into Ableton Live as ready-to-fire stem clips via OSC, with a companion React app (in `companion-app/`) for live recommendations during the set.

The repo has three pieces:
1. **Python pipeline** (`src/dance/`) — the brain. Spotify ingest → analysis → stems → regions → embeddings → recommendation graph. SQLite is the source of truth.
2. **FastAPI backend** (`src/dance/api/`) — read-mostly REST over the SQLite DB, plus a WebSocket for live Ableton state and an OSC passthrough that auto-loads stem clips into Live's session view (Live 12.0.5+).
3. **React companion app** (`companion-app/`) — Vite + TypeScript + Tailwind. Three views: **Booth** (live performance: set rail + now-playing card with structure timeline + recs with one-tap "Load → scene N"), **Crate** (vibe search, filters, pre-set Stack), **Pipeline** (ingest + processing status).

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

## Install

Backend:
```bash
pip install -e ".[dev]"
```

Companion app:
```bash
cd companion-app && npm install
```

## Quick start — pipeline

```bash
dance config --spotify-playlist "https://open.spotify.com/playlist/<id>"
dance run --once          # sync + process
dance build-graph         # build recommendation edges
```

## Quick start — companion app

Two processes:

```bash
# Terminal 1: backend
uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000

# Terminal 2: React UI
cd companion-app && npm run dev   # http://localhost:5173
```

Open `http://localhost:5173` on an iPad (landscape) or a desktop browser. Ableton state is pushed over the WebSocket — install AbletonOSC first (see [docs/abletonosc_setup.md](docs/abletonosc_setup.md) — includes a one-line patch needed to enable auto-loading stems into Live's session view).

## DJ flow (in the Booth)

1. **Open the companion** at `http://localhost:5173` in landscape. Booth view loads.
2. **First track**: vibe-search with ⌘K ("punchy techy with vocals") or browse the Crate. Click **Load → scene 1**. The backend auto-creates 5 reusable Deck tracks in Live (Mix / Drums / Bass / Vocals / Other) once per session, then uses `ClipSlot.create_audio_clip` to drop each stem into scene 1's clip slots. Live's status bar reads "Loaded {title} → scene 1".
3. **Fire scene 1** on the APC40 — all 4 stems play in sync. Use the 4 stem faders to mute/blend live.
4. **Companion auto-detects** the playing clip → flips NowCard badge from `○ HISTORY` to `● LIVE`, logs the play, and re-seeds Up Next with recs compatible with what's playing (`+0k 0b` chips = perfect key + BPM match).
5. **Stage the next song**: pick a rec, click **Load → scene 2**. Same 5 columns, new scene. Repeat.
6. **Mix**: when the current song's structure timeline shows "next: BUILDUP in 12 bars", hit scene-launch 2 on the APC. Live crossfades via your fader settings; companion auto-logs the new play.

The **SetRail** (left of the Booth) shows your **Scene Map** (which scenes have what loaded, with a ▶ on the currently-playing scene) above the **Played** history with energy arc. **MasterStrip** (top) always shows live BPM, transport, deck count, and a global ⌘K vibe-search.

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
│   └── graph_builder.py    Builds track_edges; exposes recommend()
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
