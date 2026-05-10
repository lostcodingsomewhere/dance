# CLI reference

Every command lives in `src/dance/cli.py`. Installed as the `dance` entry point (see `pyproject.toml:76`).

```bash
dance --help                 # top-level group
dance <command> --help       # per-command flags
dance --verbose <command>    # DEBUG logging
```

All commands read configuration from `~/.dance/.env` and `DANCE_*` env vars (see `src/dance/config.py:19`). On startup, `main()` (`cli.py:50`) ensures the data dirs exist and calls `init_db()`.

---

## `dance config`

Show or set top-level config.

```bash
dance config --show
```

Pass `--spotify-playlist <url>` / `--library-dir <path>` to print where the value would be set — actual persistence is via env vars or `~/.dance/.env`. The command itself does not write a config file.

**State changes:** none.

---

## `dance sync`

Run spotDL against the configured playlist URL, downloading any tracks not already in `library_dir`. Wraps `dance.spotify.downloader.SpotifyDownloader.sync_playlist()`.

```bash
dance sync
dance sync --dry-run     # list-only, no download
```

Fails fast if `DANCE_SPOTIFY_PLAYLIST_URL` is unset. Output: `Downloaded: N  Skipped: N  Failed: N`.

**State changes:** no row state mutations. New files land in `library_dir` and are picked up by the next `dance process` ingest pass.

---

## `dance process`

Scan `library_dir` for new files (ingest), then run every registered pipeline stage until no track changes state.

```bash
dance process                       # process everything pending
dance process --limit 10            # cap at 10 tracks per stage per pass
dance process --track-id 42         # only one track
dance process --skip-stems          # skip Demucs separation (auto-advances state)
dance process --skip-embeddings     # skip CLAP embedding
```

**State changes (per track):**
```
filesystem -> pending -> analyzed -> separated -> stems_analyzed -> regions_detected -> complete
```
See `docs/architecture.md` for the stage table. A skipped stage auto-advances its tracks to `output_state` (`dispatcher.py:221`) so downstream stages still run.

---

## `dance list`

Read-only browse of the `tracks` table joined to `audio_analysis`.

```bash
dance list                          # 50 most recent
dance list --bpm-range 125-130
dance list --key 8A
dance list --energy 7
dance list --state error
dance list --limit 200
```

**State changes:** none.

---

## `dance run`

Convenience: `sync` then `process`, one pass or in a loop.

```bash
dance run --once          # one sync + process pass, exit
dance run                 # daemon: every settings.sync_interval_minutes (default 30 m)
dance run --once --skip-sync   # process only
```

Ctrl+C in daemon mode exits cleanly.

**State changes:** same as `sync` + `process`.

---

## `dance status`

Pipeline state counts grouped by `Track.state`.

```bash
dance status
```

Output is one row per `TrackState` enum value plus `total`. Use this to confirm `dance process` made progress.

**State changes:** none.

---

## `dance tag [TRACK_ID]`

Run the local tagger over `COMPLETE` tracks. See `docs/tagging.md` for what the modes do.

```bash
dance tag                       # CLAP zero-shot over all untagged COMPLETE tracks
dance tag --track-id 42         # one track
dance tag --limit 20            # cap
dance tag --retag               # re-run on tracks that already have tags from this source
dance tag --deep                # use Qwen2-Audio (needs DANCE_DEEP_TAGGER_ENABLED=true)
```

By default, tracks that already have tags from the chosen source (`inferred` for CLAP, `llm` for Qwen) are skipped (`cli.py:336`). `--retag` overrides; `--track-id` always re-tags the named track regardless.

Re-tag semantics: the chosen tagger deletes its own existing tags on the track and writes a fresh batch. Tags from other sources (e.g. `manual`) are untouched.

**State changes:** none on `Track.state`. Writes `tags` + `track_tags` rows.

---

## `dance build-graph`

(Re)build the `track_edges` recommendation graph. Library-level operation, not a per-track stage.

```bash
dance build-graph                            # rebuild all kinds, all tracks
dance build-graph --track-id 1 --track-id 2  # incremental: edges touching these tracks
```

Builds: `harmonic_compat`, `tempo_compat`, `embedding_neighbor`, `tag_overlap`. Each kind deletes its existing rows (scoped to `track_ids` if provided) and re-inserts.

Run after `dance process` completes a meaningful batch, otherwise the recommender query returns stale neighbors.

**State changes:** rewrites `track_edges`. No `Track.state` mutation.

---

## `dance export-als [TRACK_ID]`

Generate Ableton Live Set (`.als`) files. See `docs/als-export.md` for what's inside.

```bash
dance export-als 42                          # single track, default filename
dance export-als 42 --out ~/Music/Dance/Sets/Custom.als
dance export-als --all                       # every COMPLETE track
```

The output path must live under `settings.als_output_dir` (default `~/Music/Dance/Sets`). Anything outside is rejected with `AlsOutsideDirError` (`als/generator.py:46`).

Errors per-track (missing analysis, no stems, not COMPLETE) print a red line and continue when `--all`.

**State changes:** none. Writes a file to disk.
