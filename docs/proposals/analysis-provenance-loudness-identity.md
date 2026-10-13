# Analysis provenance, loudness and track identity: one migration

**Status:** proposal, not implemented. Needs your go-ahead (CLAUDE.md rule 1: SQLAlchemy models, Alembic, API response schemas).
**Date:** 2026-10-13
**Touches:** `core/database.py` (`Track`, `AudioAnalysis`), one Alembic revision, `alembic/env.py`, `api/schemas.py` (additive fields only), the writes in `analyze.py` / `analyze_stems.py`, the download finalizer, one backfill script. No Stage-protocol, `.als` or OSC change.
**Related:** [`warp-guard.md`](warp-guard.md), [`library-duplicates.md`](library-duplicates.md), branches `claude/analysis-fixes` (BPM/duration fixes + `dance reprocess`) and `claude/acquisition` (scored, verified downloads)

---

Nine small columns that three separate improvements need. They are bundled
because every migration costs you a manual backup plus `alembic upgrade head`
(`init_db()` only runs `create_all`, which never adds a column to an existing
table, `database.py:832-836`), and because the expensive part of the backfill
is the library re-decode that `dance reprocess` is about to do anyway.

All library numbers below come from the real DB, read only through
`file:/Users/arya/.dance/dance.db?immutable=1` on 2026-10-13: head
`e4a91c7b2d05`, 282 tracks, 1,646 `audio_analysis` rows.

## The problem, measured

### 1. Bad analysis rows can't be told apart from good ones

- **All 282 full-mix rows have `bpm_confidence = 0.7`**, the hard-coded librosa
  fallback (`analyze.py:183`). Essentia never ran: `analyze.py:174` calls
  `len()` on RhythmExtractor2013's confidence, which is a scalar. Reproduced
  with the installed essentia: `TypeError: object of type 'float' has no len()`.
  The 282 tracks share 26 distinct BPMs. `claude/analysis-fixes` fixes this.
- After that fix and a `dance reprocess`, the 0.7 marker no longer identifies anything. On stem
  rows `bpm_confidence` also changes **meaning**: today it is a beat-count
  heuristic (1,111 of 1,112 live stem rows read 0.4), afterwards it is tempo
  coherence in [0, 1]. No row records which code wrote it, so if a reprocess
  stops partway, no query can tell the old rows from the new ones.
- Rule 3 has no guard. Nothing writes an Ableton or manual BPM to the DB yet. When
  warp-guard Option C lands and something does, the next
  `dance reprocess --from analyze` overwrites it: the upsert at
  `analyze.py:116-131` sets `bpm` unconditionally, and `detect_regions`
  re-derives every `beats` row from that value.

### 2. The energy features measure mastering level, not energy

`energy_overall = mean(RMS) / 0.2`, clipped to 1 (`utils/audio.py:32`). That is
linear in gain. On a synthetic track, +6 dB moved `energy_overall` from 0.349 to
0.696 and integrated loudness by exactly 6.00 LU. Across the library:

| | full mix | drums stem |
|---|---|---|
| `energy_overall` saturated at 1.0 | **237 / 282** | 41 / 278 |
| `energy_peak` saturated at 1.0 | **265 / 282** | 251 / 278 |
| `floor_energy` 9 or 10 | **257 / 282** (91%) | |

So the energy term in tail recs (`tail.py:142`) mostly compares tracks that all
score 9 or 10. Because the stored values are clipped, they can't be renormalised
after the fact. No loudness figure is stored anywhere either, so every clip
loads at its mastering level. How far apart those levels are in this library
is unmeasured (verification step 4 measures it).

### 3. The DB forgets what Spotify said and where the audio came from

- **`duration_seconds` stores Spotify's claim** for app-ingested rows
  (`routers/pipeline.py:508`, `:694`), because analyze only fills it when it is
  empty (`analyze.py:94`). I compared it with each track's real length, taken
  from its stems' RMS curves (`rms_curve_length × hop`):

  | ingested | rows | off by > 2 s | > max(10 s, 3.5%) | > 30 s |
  |---|---|---|---|---|
  | 2026-05-17 | 196 | 0 | 0 | 0 |
  | 2026-05-24 / 05-26 | 82 | 27 | **22** | 14 |

  The worst is #300 *Cycles*: Spotify says 185.8 s and the file is 414.1 s. All
  14 rows off by more than 30 s must have come from the unfiltered yt-dlp retry
  (`csv_importer.py:223`), because the first attempt filters at ±30 s. Two
  consumers read the wrong value today. The `.als` writer places the end warp
  marker and loop end from it (`writer.py:479-480` via `generator.py:159,167`),
  so *Cycles* would loop over the first 186 s of a 414 s file. The warp check
  computes expected beats from it (`routers/ableton.py:484`, `:564-569`).
  `claude/analysis-fixes` correctly makes `duration_seconds` the measured
  length. That also deletes the only stored copy of the Spotify duration, so a
  wrong-version download would show up only as a log warning.
- **The download source is not recorded anywhere.** yt-dlp runs with `--quiet` and no
  `--print` (`csv_importer.py:191-201`). Example: #206 *Surrender* and #207
  *Surrender - TH;EN Remix* are listed by Spotify at 200 s and 210 s, but both files are 375.5 s.
  Their sizes differ by exactly 21 bytes, which is the length difference of
  their ID3 artist and title strings. Every stem of both has the same
  3,755-frame RMS curve. So the same upload was downloaded twice, and a stored
  source URL would have shown that with one query.
- **There is no ISRC**, the only identifier for a recording that works across
  catalogs ([ISO 3901, 12 characters](https://en.wikipedia.org/wiki/International_Standard_Recording_Code)).
  Spotify removed `external_ids` for development-mode apps in
  [February 2026](https://developer.spotify.com/documentation/web-api/references/changes/february-2026)
  and [reverted that in March](https://developer.spotify.com/documentation/web-api/references/changes/march-2026).
  The playlist reader doesn't request the field (`search.py:181-182`).
  Aside, not verified against this account: that same reader calls
  `/playlists/{id}/tracks` (`search.py:180`), which the February changelog lists
  as removed in favour of `/playlists/{id}/items`.

## Proposed columns

### `audio_analysis`: on every row, mix and stems alike (rule 4)

| column | type | null | meaning |
|---|---|---|---|
| `analyzer_version` | `String(32)` | yes | Which code wrote the row, e.g. `analyze/2` or `analyze_stems/2`, with a `+librosa` suffix when the tempo fallback produced `bpm`. NULL means written before this migration, which today is all 1,646 rows. Follows the precedent of `TrackEmbedding.model_version` and `StemFile.model_used`. |
| `grid_source` | `String(16)` | no, default `'analyzer'` | `analyzer` \| `ableton` \| `manual`. Validated by a Python `GridSource` enum, as `RegionSource` is; no CHECK constraint. Stem rows copy it from the mix row, the same way they copy `bpm`. |
| `analyzer_bpm` | `Float` | yes | The rule-3 hint: the analyzer's latest estimate whenever `grid_source != 'analyzer'`, NULL otherwise. `bpm` remains the tempo every consumer already reads. |
| `loudness_lufs` | `Float` | yes | BS.1770 integrated loudness. Essentia reports −70.0 when everything is gated, which identifies a silent stem. |
| `true_peak_dbtp` | `Float` | yes | Maximum over both channels of the 4× oversampled \|x\|. Floored at −120, because digital silence gives −inf and JSON can't carry it. |
| `loudness_range_lu` | `Float` | yes | LRA. It comes out of the same essentia call at no extra cost. |

### `tracks`

| column | type | null | meaning |
|---|---|---|---|
| `isrc` | `String(12)` | yes | Spotify's `external_ids.isrc`, uppercased, no hyphens. **Not unique**: single and album releases share an ISRC, and so do rows marked `duplicate_of`. |
| `source_url` | `Text` | yes | Where the audio was downloaded from (yt-dlp `webpage_url`, spotDL `download_url`). NULL means unknown or a local file. |
| `expected_duration_seconds` | `Float` | yes | Spotify's `duration_ms / 1000` at ingest, never overwritten by analysis. NULL for rows not from Spotify. The name pairs it with `duration_seconds`. |

Rules that come with them. These are stage logic only, with no protocol change:

- **Grid guard.** When the existing mix row has `grid_source != 'analyzer'`, the
  analyze stage writes its estimate to `analyzer_bpm`, leaves `bpm` alone, and
  logs both values.
- **ISRC binding.** Set `isrc` only from the Spotify entry the row was created
  from, or from a match confirmed by duration. That is the rule
  `claude/acquisition` (`c69b40e`) already applies to `spotify_id`. A name-only
  match must not copy it, because a remix or extended mix has its own ISRC.
- **Wrong version** means `|duration_seconds − expected_duration_seconds| > max(10 s, 3.5%)`,
  the tolerance both branches already use.

### Measuring loudness: what the probes showed

All of these were run with the installed essentia 2.1-beta6-dev on synthetic
audio:

- `LoudnessEBUR128` accepts **stereo only**. A mono vector is rejected with
  `cannot convert argument VECTOR_REAL to VECTOR_STEREOSAMPLE`. It works at
  22.05, 44.1 and 48 kHz: a 1 kHz sine at −20 dBFS reads −19.96, −19.99 and
  −19.99 LUFS. It does not compute true peak
  ([docs](https://essentia.upf.edu/reference/std_LoudnessEBUR128.html)).
- **Don't measure the mono decode the stages already hold.** Downmixing to mono
  and duplicating to L/R reads 0.00, −1.25 and −3.01 LU low for identical,
  partly correlated and uncorrelated channels. Decode once in stereo at the
  native rate and downmix that for the existing features, so the file is still
  decoded only once. A mono file is duplicated to L/R, which is how Live plays
  it.
- **True peak: use `scipy.signal.resample_poly(x, 4, 1)`, not essentia's
  `TruePeakDetector`.** At its default quality TruePeakDetector takes 2.51 s per
  mono minute. scipy gives the same answer in 0.31 s (0.04 dBTP for both, on a
  planted intersample peak). TruePeakDetector at `quality=4` misses that peak
  entirely: it returns the −1.84 dB sample peak. scipy is already a dependency.
- **Cross-check** on a synthetic 24-bit stereo WAV: essentia and scipy give
  I = −14.74 LUFS and TP = +0.02 dBTP; `ffmpeg -af ebur128=peak=true` gives
  −14.7 LUFS and 0.0 dBTP.
- **Cost** on a synthetic 6-minute stereo file on the M2 Pro: loudness takes
  3.7 s and true peak about 3.7 s. That is about 37 s per track for the mix plus
  4 stems, or about 1.5 h of wall time for 282 tracks at stage concurrency 2.
  It is added to a reprocess that runs anyway.

## Options

**A. One migration with all nine columns (recommended).** You migrate once, and
the reprocess that is due anyway fills the analysis columns in a single pass
over the library.

**B. One migration per consumer, as each one lands.** That means three manual
backup-and-upgrade rounds. If the reprocess runs before the loudness columns
exist, the library has to be decoded a second time later.

**C. Don't store anything; measure at load time.** Measuring loudness and true
peak on a 6-minute mix takes about 7 s, which is too slow for the interactive
load-track path. Provenance can't be recomputed at all.

Alternatives I rejected for individual columns:

- **`analyzer_version` on `tracks`.** Different stages write the rows at
  different times, so a partial reprocess leaves mixed versions within one track.
- **A JSON `provenance` blob.** The one query that matters,
  `WHERE analyzer_version IS NULL`, can't use it.
- **An analyzer-owned `bpm` plus a new `grid_bpm`.** Every BPM reader (writer,
  warp check, recommenders, graph) would need a `coalesce`. Keeping `bpm` as the
  effective tempo moves only the hint.

## Alembic notes

- **The upgrade doesn't rebuild either table.** Every column is nullable or has a constant
  server default, so batch mode emits plain `ALTER TABLE … ADD COLUMN`. I ran
  the exact operations on a scratch copy of the real DB (copied via the
  immutable URI, then deleted): 9 ALTERs, no `_alembic_tmp_` table, 6 ms. All
  243,005 rows in 15 tables were unchanged, and the partial unique indexes and
  the `duplicate_of` FK were intact. **Don't add a CHECK, FK or UNIQUE constraint.** Any of
  them forces a move-and-copy; verified with `create_check_constraint`, which
  produced CREATE `_alembic_tmp_audio_analysis` / INSERT / DROP / RENAME.
- **The downgrade does rebuild.** Each `drop_column` triggers move-and-copy of
  both tables. With foreign keys off it took 0.1 s and every count survived.
- **Rebuilding `tracks` (as `e4a91c7b2d05` already does) is safe only by
  accident.** I populated a scratch DB at `b7d4e2f1a9c3` and upgraded it:

  | Alembic connection | result |
  |---|---|
  | as shipped (`env.py:64` uses `engine_from_config`, no pragma) | all counts survive |
  | `PRAGMA foreign_keys=ON` forced | **reports success**, while stem_files 12→0, audio_analysis 15→0, and regions, embeddings, beats and track_edges all →0 |
  | the same, plus one `session_plays` row (FK with no `ON DELETE`) | `IntegrityError` |

  `DROP TABLE` performs an implicit DELETE that fires `ON DELETE CASCADE`
  ([sqlite.org/foreignkeys.html §5](https://www.sqlite.org/foreignkeys.html)).
  If `env.py` were ever changed to reuse `get_engine()`, whose connect listener
  turns foreign keys on (`database.py:770`), the next table rebuild would wipe
  every child table without reporting an error.
- **Migrations are not atomic with this driver.** pysqlite doesn't issue BEGIN
  before DDL
  ([SQLAlchemy: transactions with the sqlite3 driver](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html)).
  On a scratch copy, a migration that failed in its second batch left
  `audio_analysis.loudness_lufs` added and an `_alembic_tmp_tracks` table
  behind, so the next attempt would fail with "table already exists". With
  SQLAlchemy's documented recipe (`isolation_level = None` on connect, then emit
  `BEGIN` on begin), the same failure left nothing behind.
- **The real DB already fails `foreign_key_check`: 252 rows.** They are stem
  analysis rows for `stem_files` ids 257–508, which no longer exist: 63 tracks,
  all analysed 2026-05-17 between 18:18 and 18:22. Those 63 tracks have fresh
  stems and fresh analysis. Any check therefore has to compare before and after
  rather than expect zero. Deleting the 252 rows is a one-line DELETE and out of
  scope here.

**Recommended in the same change:**

1. In `env.py`, add a connect listener that issues `PRAGMA foreign_keys=OFF`
   before any BEGIN (SQLite ignores the pragma inside a transaction, §2 of the
   page above) and applies the BEGIN recipe. After `run_migrations()`, fail if
   `PRAGMA foreign_key_check` reports a violation that wasn't there before.
2. Add `tests/test_alembic_populated.py`. Walk every revision from initial to
   head and back, with rows in every table that exists at each step. Assert
   that per-table counts and `foreign_key_check` are unchanged after each
   `+1` and `-1`. Run the walk a second time with a global listener forcing
   `foreign_keys=ON`, to prove that env.py's OFF wins. `docs/dev.md:150` already
   claims `test_alembic.py` walks up and down, but it only upgrades an empty DB.
3. Give `grid_source` `server_default="analyzer"` in the model too, so DBs
   created by `init_db()` match migrated ones. Base the new revision on main's
   head at the time it's implemented; other branches may add revisions first.

## Backfill

| column | how | cost |
|---|---|---|
| `grid_source` | The migration default `'analyzer'`, which is true for every existing row. | 0 |
| `analyzer_bpm` | Left NULL; it only matters once something writes an Ableton or manual BPM. | 0 |
| `analyzer_version`, loudness ×3 | `dance reprocess --from analyze --all --skip-embeddings` (from `claude/analysis-fixes`). Until it runs, NULL means "needs reprocess". | the reprocess, plus about 37 s per track |
| `isrc`, `expected_duration_seconds` | One-off script: `GET /tracks/{id}` for each of the 142 `spotify_id`s, reading `external_ids.isrc` and `duration_ms`. **Don't copy `duration_seconds` instead**: for the 57 rows whose `spotify_id` was attached by name match on 2026-05-17, it is the file's tag duration. | ~2 min with acquisition's 429 backoff |
| `isrc`, `source_url` from file tags | The same script reads tags read-only. spotDL 4.4.3 embeds `TSRC` (ISRC) and `COMM` (download URL) in mp3s (`.venv/…/spotdl/utils/metadata.py:601-604`). How many of the 2026-05-17 files carry them is unmeasured; the dry run prints it. | seconds |
| `source_url` for yt-dlp rows (2026-05-24 / 05-26) | Unrecoverable: yt-dlp ran without `--print` or `--embed-metadata`, so these stay NULL. New downloads write `AcquireResult.chosen` (`claude/acquisition`) in `_finalize_track_after_download` (`routers/pipeline.py:761`). | 0 |

**Order:** migrate, then run the backfill script, then reprocess. The library is
decoded once.

## API and UI exposure

Additive optional fields only, defaulting to `None`, so older clients are
unaffected. The companion's TS types change in the same commit.

- **`AnalysisOut`** gains `loudness_lufs` and `true_peak_dbtp`. The Set view
  can then show a level jump between two tracks before gain normalisation
  exists.
- **`TrackOut`** gains `expected_duration_seconds` and `source_url`. They feed a
  Pipeline-view chip ("file 414 s · Spotify 186 s") with a link to the upload.
  This is the only UI that ships with the migration.
- **Not exposed:** `analyzer_version` (`dance status` prints "N rows predate
  analyzer v2" instead), `grid_source` and `analyzer_bpm` (there is no
  write-back UI yet), `isrc`, stem loudness and LRA. `DeckCellOut` is unchanged.

## What this unblocks (each a separate change)

1. **Clip-gain normalisation on load.** One gain is applied to a track's four
   stems and its mix, so the stems still sum to the mix:
   `gain_db = min(target − mix_lufs, ceiling − mix_true_peak)`. In the `.als`,
   `SampleVolume` is linear amplitude. Live's bundled *Sample Reference.als*
   has 2,679 clips: 2,664 at `1`, and the rest at values such as `1.99526238`
   (+6 dB), `0.7943282127` (−2 dB) and `10` (+20 dB), all exactly
   10^(dB/20). The writer emits `1` today (`writer.py:531`). Over OSC, stock
   AbletonOSC has `/live/clip/set/gain`, but LOM `Clip.gain` is a normalised
   0–1 value whose dB mapping isn't documented
   ([LOM Clip](https://docs.cycling74.com/apiref/lom/clip/)). It needs
   calibrating against `gain_display_string`, which stock AbletonOSC doesn't
   expose, so that half needs an OSC-contract proposal. Either half has to be
   re-verified in Live.
2. **Energy features that don't depend on mastering.** Compute energy on
   loudness-normalised audio, and use stem LUFS minus mix LUFS as a measure of
   how prominent each stem is (rule 4). This is an analyzer change: bump
   `analyzer_version`. It also changes rec behaviour, so it needs the real-set
   check described in `set-rail-verification.md`.
3. **Wrong-version and duplicate-upload queries.** These become plain SQL
   against `expected_duration_seconds` and `source_url`.
4. **Writing Live's grid back (warp-guard Option C).** It sets `bpm` and
   `grid_source='ableton'`, and the grid guard keeps it through every later
   reprocess.

## What NOT to add

- **No `bpm_method` or other per-field provenance columns.** The version suffix
  covers the one runtime branch, the tempo fallback.
- **No constraints and no indexes.** Constraints force a rebuild, and Python
  enums are the project's convention. 282 rows don't need indexes, and adding
  one later doesn't require a rebuild either.
- **No loudness curves.** `rms_curve` already exists. Short-term LUFS at 10 Hz
  would add about 3,600 floats per row.
- **No stored gain or target columns.** Gain is derived at load time from LUFS
  and a setting, so a stored value would go stale whenever the target changes.
- **No ReplayGain or loudness tags written into library files.** dance never
  modifies library audio, and `file_hash` covers the first 1 MB
  (`ingest.py:67-94`), which includes the ID3 header.
- **No downbeat or grid-offset column** until warp-guard Option C step 5 shows
  that a repaired grid persists.
- **No download-history table.** One URL per row is enough; earlier attempts
  are recorded in job messages.

## Decisions for you

1. **Option A, a single migration?** Recommended.
2. **`loudness_range_lu`: include it or drop it?** Recommend including it: it
   costs nothing during this pass, and adding it later means decoding the whole
   library again.
3. **`grid_source` and `analyzer_bpm` now, while they're dormant, or later with
   the write-back?** Recommend now: it costs no extra migration, and it protects
   any grid written before the next reprocess.
4. **Harden `env.py` in the same change** (explicit FK OFF, the BEGIN recipe,
   the `foreign_key_check` gate)? Recommend yes.
5. **The Spotify backfill makes 142 `GET /tracks/{id}` calls with your
   credentials.** Is that OK? Development mode now requires Premium on the app
   owner's account
   ([migration guide](https://developer.spotify.com/documentation/web-api/tutorials/february-2026-migration-guide)).
6. **The 252 orphaned stem-analysis rows:** delete them in this change or
   separately?

## Migration cost

- **Code:** about 10 model lines, one revision (~40 lines), ~20 lines in
  `env.py`, two schema classes plus TS types, ~40 lines of stage writes, a
  backfill script of ~120 lines, and two new test files.
- **Your part:** stop the backend, back up with
  `sqlite3 ~/.dance/dance.db ".backup ~/.dance/backups/dance-pre-<rev>.db"`
  (`.backup` handles WAL; a plain `cp` can miss committed pages), run
  `alembic upgrade head`, run the backfill, then reprocess.
- **Ongoing:** about 37 s more per new track, on top of Demucs.

## Verification plan

1. **Tests.** `test_alembic_populated.py` (above). A stage test on synthetic
   audio: a 1 kHz sine at −20 dBFS in stereo reads −20.0 ±0.1 LUFS, a mono file
   is duplicated to L/R, and silence gives −70.0 with the true-peak floor. A
   grid-guard test: a row with `grid_source='ableton'` keeps its `bpm` through a
   reprocess and gains `analyzer_bpm`.
2. **Migration on a copy of the real DB:**
   ```bash
   sqlite3 'file:/Users/arya/.dance/dance.db?immutable=1' ".backup /tmp/dance-mig/dance.db"
   alembic -x url=sqlite:////tmp/dance-mig/dance.db upgrade head
   ```
   Then compare per-table counts. `foreign_key_check` should show 252 rows
   before and after, and `integrity_check` should return `ok`. Repeat after
   `alembic … downgrade -1`.
3. **Real data (rule 2).** Copy 3 library files to a temp dir: one mastered hot,
   one quiet, one with a silent vocal stem. Analyse them into a temp DB, as
   `claude/analysis-fixes` did. Their `loudness_lufs` and `true_peak_dbtp`
   should be within 0.1 of `ffmpeg -i <file> -af ebur128=peak=true -f null -`.
4. **Library spread.** After the reprocess, run
   `SELECT min(loudness_lufs), max(loudness_lufs), avg(loudness_lufs) FROM audio_analysis WHERE stem_file_id IS NULL`.
   The clip-gain consumer needs this to pick a target.
5. **Identity.** After the backfill, the wrong-version query lists the 22 rows
   counted above, including *Cycles*, *Surrender* and *Vision Blurred*.
