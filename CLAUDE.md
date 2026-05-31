# CLAUDE.md

Instructions for Claude Code working on this repo. Keep brief — the docs already cover detail.

## What this repo is

**Dance** — a stem-performance brain for **Ableton Live**. Spotify playlist → analyzed tracks with stems, cues, loops, tags, and graph edges → loaded into Ableton via generated `.als` files and an OSC bridge. A React companion app (in `companion-app/`) shows live recommendations during a set.

Three pieces:
1. **Python pipeline** (`src/dance/`) — Spotify ingest → analysis → stems → regions → embeddings → recommendation graph. SQLite is the source of truth.
2. **FastAPI backend** (`src/dance/api/`) — read-mostly REST + WebSocket for live Ableton state + OSC passthrough.
3. **React companion app** (`companion-app/`) — Vite + TS + Tailwind. Three views: **Booth** (live performance: SceneGrid mirror + MasterVisualizer stacked-stem waveforms + ComboStrip + per-column rec banners + CueStrip + PlayedStrip + slide-out SetRail with tail-recs), **Set** (full-pane editor for the active set + library browse), **Pipeline** (ingest/processing status; also library inventory via the Done-column filter). Sets persist in the DB (`sets`/`set_tracks`, one `is_active` at a time). Hybrid **⌘K** palette combines fuzzy artist/title search with CLAP vibe results. All recs scoped per-column and re-scored against the active stem combo. See [`docs/dj_ux_flow.md`](docs/dj_ux_flow.md) for the full surfaces map; [`docs/proposals/set-rail-and-search-consolidation.md`](docs/proposals/set-rail-and-search-consolidation.md) for the Set Rail design.

Full architecture: [`docs/architecture.md`](docs/architecture.md). Repo layout: [`docs/dev.md`](docs/dev.md).

## Hardware context (the user owns this gear)

| | |
|---|---|
| Mac | MacBook Pro M2 Pro 14" (2023, 16 GB) |
| DAW | **Ableton Live Standard 12.4** — locked at this version for `.als` template compatibility |
| Controller | Akai APC40 mk2 (only MIDI device) |
| Audio interface | Focusrite Scarlett 4i4 4th gen — outs 1/2 = master to speakers, outs 3/4 = cue bus to headphones (swapped in from a 2i2 on 2026-05-18) |
| Speakers | Edifier R1700BT |
| Headphones | Bose wired (DJ headphones deferred) |

The `.als` template (`src/dance/als/templates/blank_live12.xml`) was generated from this specific user's Live 12.4 install. If they upgrade Live, the template needs refreshing — see [`docs/troubleshooting.md`](docs/troubleshooting.md) under "Live rejects the .als entirely".

Full hardware rationale: [`HARDWARE.md`](HARDWARE.md). Receipt: [`ORDER.md`](ORDER.md).

## Workflow rules (from user memory)

These come from the user's persistent memory — they apply project-wide.

1. **Propose-then-implement for schema / architecture work.** Before writing code that changes the SQLAlchemy models, the Stage protocol, the `.als` writer's emission shape, the OSC message contract, or the API schemas: write a short markdown proposal under `docs/proposals/` that surfaces decisions (alternatives, trade-offs, migration cost). Wait for explicit "go ahead" before implementing. Trivial fixes, new endpoints that follow existing patterns, and new pipeline stages that don't touch the protocol can proceed without a proposal.
2. **Real-data verification before claiming done.** Mocked tests pass while third-party APIs drift silently. Before reporting "phase complete" on anything that touches Spotify, Demucs, CLAP, AbletonOSC, or Live's `.als` loader: actually run it end-to-end on a real track with real model weights, then report what happened.
3. **Ableton-first.** When the audio engine and the analyzer disagree on truth (BPM, key, region boundaries), Ableton's interpretation wins — that's what the user hears. Persist the analyzer value but surface it as a hint, not an authority.
4. **Stems are first-class.** Every per-track field has a per-stem analog where it makes sense (RMS, presence, BPM-coherence, embedding). Don't add features that only work on the full mix.

## Code conventions

- **Python 3.10+**, target `py310`. Ruff line length 100, rules `E F I N W UP` minus `E501`. mypy strict-ish (`warn_return_any`, `warn_unused_ignores`).
- **Pre-commit** (no hook installed; run manually):
  ```bash
  ruff check src/dance tests && mypy src/dance && pytest -q
  ```
  - **ruff is pinned to `>=0.12,<0.13`** in `pyproject.toml` for reproducibility.
    If your venv has a newer ruff (e.g. the off-spec Python 3.14 venv ships 0.15),
    `pip install 'ruff>=0.12,<0.13'` first.
  - **The tree is NOT ruff-clean** — ~145 pre-existing lint findings
    (UP035/UP045/I001/F401…) exist at every ruff version, and the gate was never
    actually enforced. So **scope ruff to the files you changed**, e.g.
    `ruff check <your files>`, not `ruff check src/dance tests` (which dumps ~145
    unrelated errors). Cleaning the whole tree is a separate, deliberate task.
  - **Do NOT run `ruff format` wholesale** (`ruff format src/dance tests`). This
    code was hand-formatted and never `ruff format`-ed — *any* ruff version
    rewrites dozens of files. Format only files you changed; prefer
    `ruff format --check <file>` to inspect before writing.
- **Tests use synthetic audio.** `tests/audio_fixtures.py:24` — no real audio files in the repo, no downloads in CI. See [`docs/dev.md`](docs/dev.md) "Synthetic audio fixture".
- **One file per pipeline stage.** `src/dance/pipeline/stages/<name>.py`. The dispatcher auto-discovers via `_register_default_stages()`. Pattern: [`docs/dev.md`](docs/dev.md) "Adding a new pipeline stage".
- **One file per API resource.** `src/dance/api/routers/<resource>.py`. Pattern: [`docs/dev.md`](docs/dev.md) "Adding a new API endpoint".
- **SQLAlchemy models live in one file by design** — `src/dance/core/database.py`. Don't split.
- **Schema changes go through Alembic.** Two partial-unique indexes for `audio_analysis` are created via raw DDL in `init_db()` because SQLAlchemy can't model `UNIQUE ... WHERE` portably — autogenerate won't see them.

## When the user says X, do Y

| User says | What to do |
|---|---|
| "let's DJ" / "load this playlist" | `dance config --spotify-playlist <url>` → `dance run --once` → `dance build-graph` → `dance export-als --all`. See [`LEARNING.md`](LEARNING.md) for the full runbook. Note: an exported `.als` is opened in Live **only to load the deck columns** — stems are then **live-loaded through the app** (⌘K / rec promote → OSC). The `.als` clips themselves are not library-linked by design; see [`docs/proposals/als-deck-name-prefix-mismatch.md`](docs/proposals/als-deck-name-prefix-mismatch.md). |
| "the als is broken" / "Live won't open the set" | See [`docs/troubleshooting.md`](docs/troubleshooting.md) "Live rejects the .als entirely". Don't refactor `writer.py` from scratch — it's template-based on purpose. |
| "let's tag tracks" | `dance tag` (CLAP zero-shot, fast) or `dance tag --deep` (Qwen2-Audio, 10–30 s/track). See [`docs/tagging.md`](docs/tagging.md). |
| "let's plan a set" / "make a set" | Open the companion app → ⌘\\ to open the Set Rail → "create empty set" → fill via ⌘K. Sets persist; rail surfaces tail-recs scored against the trailing arc. See [`docs/proposals/set-rail-and-search-consolidation.md`](docs/proposals/set-rail-and-search-consolidation.md). |
| "tail recs seem off" / "tune the rec scoring" | Weights live in [`src/dance/recommender/tail.py`](src/dance/recommender/tail.py) as `_W_EMBED/_W_KEY/_W_BPM/_W_ENERGY`. Run a real-set check before changing — see [`docs/proposals/set-rail-verification.md`](docs/proposals/set-rail-verification.md). |
| "the model isn't downloading" / "MPS crashed" | See [`docs/troubleshooting.md`](docs/troubleshooting.md). MPS fallback is automatic. |
| "show me what's running" | Backend: `curl http://127.0.0.1:8000/health`. UI: open `http://localhost:5173`. OSC: AbletonOSC status panel in Live. |

## Don't

- **Don't commit secrets.** `~/.dance/.env` is OUT of repo; it has DANCE_* config that may include private playlist URLs.
- **Don't move stems after generating a `.als`.** Live references stems by absolute path. Move → "Missing Media" dialog. Regenerate the `.als` after a move.
- **Don't bypass the `als_output_dir` safety guard.** `out_path` must resolve inside `settings.als_output_dir`. The API endpoint relies on this — see `src/dance/als/generator.py` → `AlsOutsideDirError`.
- **Don't change `.als` writer emission without re-verifying in Live.** Tests only assert XML shape, not "Live will accept this." Generate a fresh `.als`, open it in Live, confirm 5 tracks load + audio plays + tempo correct.
- **Don't add Bluetooth / network audio paths.** Wired-only is a locked decision (see [`HARDWARE.md`](HARDWARE.md)).
- **Don't reintroduce Traktor.** The repo pivoted away from Traktor explicitly. Stem-in-Ableton is the locked architecture.

## Quick-reference

```bash
# Setup
source .venv/bin/activate
pip install -e ".[dev]"

# Pipeline
dance status                              # what state are tracks in
dance sync                                # download from configured playlist
dance process [-n 5] [--skip-stems]       # advance tracks through stages
dance tag [--deep]                        # CLAP zero-shot or Qwen2-Audio
dance build-graph                         # build recommendation edges
dance export-als <id> | --all             # generate Live Set(s)

# Servers
uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000
(cd companion-app && npm run dev)         # http://localhost:5173

# Tests
pytest                                    # ~25 s, 197 tests, synthetic audio
```

Everything else in [`README.md`](README.md) and `docs/`.
