# CLI reference

Every command lives in `src/dance/cli.py`. Installed as the `dance` entry point (see `pyproject.toml:76`).

```bash
dance --help                 # top-level group
dance <command> --help       # per-command flags
dance --verbose <command>    # DEBUG logging
```

All commands read configuration from `~/.dance/.env` and `DANCE_*` env vars (see `src/dance/config.py:19`). On startup, `main()` (`cli.py:50`) ensures the data dirs exist and calls `init_db()` — except for `dance doctor`, which is read-only and skips both.

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
dance process --wait                # queue behind a run that's already going
```

**One run at a time.** Every dispatcher (terminal or API) holds `<data_dir>/dispatcher.lock` for its whole run. If another run holds it, `dance process` / `dance run --once` / `dance ingest` print the holder's pid and exit with status **75**; pass `--wait` to start as soon as it finishes. `dance run` in daemon mode skips that pass instead. The API's `POST /pipeline/process` answers 409 and its watch loop skips its turn. A lock left behind by a killed run never blocks: the kernel drops it with the process.

Tracks the Spotify ingest pre-created (`file_hash` `pending:<spotify_id>`) are skipped until their download lands. `dance status` counts them.

**State changes (per track):**
```
filesystem -> pending -> analyzed -> separated -> stems_analyzed -> regions_detected -> complete
```
See `docs/architecture.md` for the stage table. A skipped stage auto-advances its tracks to `output_state` (`dispatcher.py:221`) so downstream stages still run.

---

## `dance reprocess`

Re-run the pipeline from a stage on tracks that already went past it — how a fix to a stage (e.g. the analyzer) reaches a library that is already `complete`. `dance process` only moves tracks forward, so it never would.

```bash
dance reprocess --from analyze --all --dry-run       # list what would be rewound
dance reprocess --from analyze --all --limit 3       # try it on 3 tracks first
dance reprocess --from analyze --all --skip-embeddings
dance reprocess --from analyze_stems -t 12 -t 40     # specific tracks
```

`--from` takes a stage name: `analyze`, `separate`, `analyze_stems`, `detect_regions`, `embed`. Exactly one of `--all` / `--track-id` is required.

`--from separate` does **not** re-run Demucs on a track whose four stems are still on disk: `separate` keeps existing stems, so for those tracks it only re-runs the stages after it. The plan says how many selected tracks that applies to. To redo a bad separation, delete that track's stem files first.

What it does (logic in `src/dance/pipeline/reprocess.py`):

1. Selects tracks whose state is **at or after** the stage's `input_state` (pipeline order = `TrackState` declaration order). Tracks still upstream are left alone — rewinding them "back" to a later stage would skip work they never had. Lowest ids first, capped by `--limit`.
2. Leaves out tracks with `duplicate_of` set unless `--include-duplicates`, and tracks in `error` unless `--include-errors` (their failure point isn't recorded in their state).
3. `--dry-run` prints the selection and the skip reasons, then exits without touching anything.
4. Otherwise rewinds the selection to the stage's `input_state` (clearing `error_message` and the heal-ledger entry), then runs the normal parallel dispatcher — the same path as `dance process` (no library scan) — scoped to the selection: tracks it didn't select (new downloads in `pending`, tracks parked mid-pipeline) are neither processed nor auto-advanced, and wait for the next `dance process`. `separate` short-circuits when all four stems are already on disk, so `--from analyze` does not re-run Demucs. The dispatcher is re-run while passes still make progress, so a single command carries the selection back to `complete`.
5. `--skip-embeddings` auto-advances `embed` instead of re-running CLAP — for the selected tracks only. Embeddings depend only on the audio, so this is safe after an analysis fix.

Afterwards run `dance build-graph` (tempo / harmonic edges use BPM and key) and `dance export-als --all` (Live Sets carry the tempo and warp markers).

The rewind and the run hold the dispatcher lock, like `dance process`: if another run (terminal or the API worker) holds it, `reprocess` exits with status 75 naming its pid and changes nothing — add `--wait` to queue behind it. `--dry-run` doesn't take the lock.

**State changes:** selected tracks → stage `input_state` → … → `complete` (or `error`). Rewrites the stage's outputs for those tracks (`audio_analysis` rows are upserted; auto regions/beats/phrases are replaced; user regions are kept).

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

## `dance ingest`

The whole new-track lifecycle in one command: `sync` → `process` → `build-graph` → `tag` → `export-als --all`, with an ffmpeg preflight up front and a one-line summary at the end (playable tracks, `.als` count, anything still stuck).

```bash
dance ingest                # full chain
dance ingest --skip-sync    # re-run the chain on the existing library, no Spotify
```

Skips the sync step (with a hint) when no playlist is configured. Equivalent to the five-command "let's DJ" runbook.

**State changes:** same as the five commands it chains.

---

## `dance dedupe`

Mark redundant copies of the same recording via `tracks.duplicate_of` — reversible, never deletes rows or audio. Metadata proposes candidate groups; every pair is then confirmed by comparing the audio before it is marked. See `docs/proposals/library-duplicates.md`.

```bash
dance dedupe                   # DRY RUN: list confirmed / unconfirmed pairs, write nothing
dance dedupe --apply           # mark the confirmed copies
dance dedupe --undo            # DRY RUN of clearing every marker
dance dedupe --undo --apply    # clear every duplicate_of marker (full revert)
```

**State changes:** with `--apply`, sets or clears `tracks.duplicate_of`. No `Track.state` mutation.

---

## `dance doctor`

Read-only health check of everything dance depends on outside its own code. Each line is `ok` / `info` / `WARN` / `FAIL` with a `fix:` hint; the command exits 1 if anything fails.

```bash
dance doctor           # human-readable
dance doctor --json    # {"ok", "counts", "checks": [{key, label, status, summary, details, fix}]}
```

| Check | What it looks at |
|---|---|
| Database migrations | Alembic revision of `settings.db_url` vs the scripts' head (lists pending revisions, unknown revisions, multiple heads, unversioned DBs, schema drift). Opens SQLite with `immutable=1`; never upgrades or stamps. |
| ffmpeg / ffprobe | On `PATH` (the `process`/`sync` preflight uses PATH) and actually runs. |
| yt-dlp, YouTube cookies, Spotify search | Same logic as `GET /api/v1/health/deps`. |
| Ableton Live | Every `/Applications/Ableton Live*.app` version vs the `.als` template's `Creator`/`Revision`; warns about Lite/Intro and multiple installs sharing `com.ableton.live`, and prints the `open -a` line to use. |
| AbletonOSC | Upstream commit of the Remote Script clone, and which OSC addresses dance sends (derived from `dance/osc/client.py`) have no handler in the installed source — named against the patch docs. Static scan; Live still needs a full restart after patching. |
| CLAP / Demucs weights | Presence in the Hugging Face / torch.hub caches (never downloads). |
| Backend API | `GET 127.0.0.1:8000/health`, 0.5 s timeout; "not running" is `info`, not a failure. |

**State changes:** none — it skips the usual `init_db()` / directory creation.

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
