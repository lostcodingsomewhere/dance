# Dance

Stem-performance brain for **Ableton Live**.

Spotify playlist → analyzed tracks with **stems, cues, loops, tags, and graph edges**, then auto-loaded into Ableton Live as ready-to-fire stem clips via OSC, with a companion React app (in `companion-app/`) for live recommendations during the set.

The repo has three pieces:
1. **Python pipeline** (`src/dance/`) — the brain. Spotify ingest → analysis → stems → regions → embeddings → recommendation graph. SQLite is the source of truth.
2. **FastAPI backend** (`src/dance/api/`) — read-mostly REST over the SQLite DB, plus a WebSocket for live Ableton state and an OSC passthrough that auto-loads stem clips into Live's session view (Live 12.0.5+).
3. **React companion app** (`companion-app/`) — Vite + TypeScript + Tailwind. Three views: **Booth** (live performance — SceneGrid mirror, per-column rec banners, slide-out Set Rail with tail-recs), **Set** (full-pane editor for the active set + library browse), **Pipeline** (ingest + processing status + library inventory). One hybrid **⌘K** palette covers fuzzy artist/title search and CLAP vibe search in one surface.

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

## DJ flow

### Before the set — plan in the Set view

1. **Open the companion** at `http://localhost:5173`. First-time users with a legacy localStorage Stack get a one-shot prompt to import it as a named Set.
2. **Open the Set Rail** (⌘\\ or the violet edge pill on the right). If no active set, create one. Sets are persistent and named — "Warehouse Sat", "Wedding 90min" — and you can switch between them from the Set editor view.
3. **Fill the set via ⌘K**. The palette is hybrid: typing surfaces fuzzy artist/title matches in the **Tracks** section first, CLAP vibe matches in the **Vibe** section below (8+ char queries). BPM/key/energy chips narrow further. Each row has `+ Set` (add to the active set) and `Load` (push straight to Live).
4. **Reorder, annotate** in the **Set** view (top nav). Per-track notes ("cue at bar 33"), arrow nudge for position, two-pane library + tail-recs on the right.

### During the set — the Booth

1. **SceneGrid** mirrors the APC40 — tap cells to fire, tap row labels to anchor, eyes here during a mix.
2. **Per-column rec banners** below the grid show 3–5 candidates per stem column, live-rescored against the active combo. Tap **+** to soft-pin a stem candidate as a whole-song Mix-column rec.
3. **Set Rail** (⌘\\) slides in to surface your planned set + tail-recs scored against the trailing arc (embedding window + key walk + BPM band + energy slope). It auto-collapses 3 s after a clip fires so the grid stays sovereign. Tap a rail track → soft-pin to Mix recs; shift-tap → force-load into Live.
4. **Auto-logging** — when a clip fires that was loaded via the companion, the play is recorded to the current `DjSession`. End the set from the PlayedStrip footer.

The **MasterStrip** (top) shows live BPM, transport, energy arc, AbletonOSC heartbeat, ⌘K, and the three view tabs.

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
