# Analysis provenance, loudness and track identity — one migration

**Status:** proposal — not implemented (CLAUDE.md rule 1: SQLAlchemy models + Alembic)
**Date:** 2026-10-13
**Touches:** `audio_analysis` (6 columns), `tracks` (3 columns + 1 index), one Alembic
revision, `src/dance/alembic/env.py`. No API schema, OSC or `.als` change.

---

## Why one proposal

Three improvements each need a few small columns, and all of their backfills
ride on the `dance reprocess --from analyze` that the BPM fix (5b9a65f, merged
today in d2ab400) already requires. One revision, one upgrade, one reprocess.

No real-library data was read to write this. Library numbers below are quoted
from the commits that measured them; everything else was measured on synthetic
audio and synthetic databases in a scratch directory (see **Verification plan**).

## The problems

### 1. Analysis rows don't say which analyzer wrote them

Today the only way to find rows from the broken analyzer is a float fingerprint:

- Old full-mix rows: `bpm_confidence = 0.7`, always. The Essentia path raised a
  `TypeError` every time, and the librosa fallback hard-coded 0.7
  (`analyze.py` at 5b9a65f^, `_analyze_bpm`). 5b9a65f: "282 tracks shared 26 BPMs".
- Old stem rows: 0.4 or 0.7 (`analyze_stems.py` at 5b9a65f^ :199), or 0.0 with
  `bpm = 0` for empty audio. In practice they are nearly all 0.4, because the
  test was `beats_per_bar >= 8` and a 4/4 bar has 4 beats.
- New rows hold continuous values: Essentia's confidence / 5.32, ≤ 0.4 for the
  librosa fit, or 0.1 (`utils/tempo.py:58-73`). On stem rows `bpm_confidence`
  now means something else, tempo coherence (`analyze_stems.py:179`).

That works once, and only by luck. The next analyzer change (for example
loudness-normalised energy, below) leaves no fingerprint at all.

**Rule 3 has no column.** If Live's grid or a hand-typed BPM ever goes into
`audio_analysis.bpm`, the next `dance reprocess --from analyze` (f2bb065)
overwrites it, because `upsert` sets every column it is given
(`pipeline/utils/db.py:18-30`). Regions already solve this with `source`:
`_clear_auto_data` deletes only `auto` rows (`detect_regions.py:144-157`). The
tempo grid has nothing like it.

### 2. No loudness measurement

`energy_overall` is mean frame RMS / 0.2, clipped to 1 (`utils/audio.py:23-34`).
It measures how loud the master is, and it saturates. Here is the same synthetic
track at six master levels, run through `analyze.py`'s own energy code on 60 s:

| integrated loudness (LUFS) | −18.0 | −14.0 | −12.1 | −10.2 | −8.4 | −6.7 |
|---|---|---|---|---|---|---|
| `energy_overall` | 0.38 | 0.61 | 0.76 | 0.96 | **1.00** | **1.00** |
| `energy_peak` | 0.59 | 0.93 | **1.00** | **1.00** | **1.00** | **1.00** |

`floor_energy` is 50 % `energy_overall` + 30 % `energy_peak`
(`analyze.py:277-297`), so above about −10 LUFS it stops telling tracks apart.

Clip-gain normalisation on load has nothing to compute from. The `.als` writer
emits `SampleVolume` = 1 (`als/writer.py:531`), and Live's `Clip.gain` is a
0.0–1.0 value with no documented dB mapping ([LOM Clip][lom]).

### 3. Track identity gets overwritten or was never kept

- **Spotify duration.** Ingest stores Spotify's `duration_ms` in
  `duration_seconds` (`api/routers/pipeline.py:528, 714`). Since d2ab400 the
  analyze stage overwrites it with the decoded length (`analyze.py:172-190`).
  That is right for Live, but it destroys the reference that d1b9222 (on
  `claude/acquisition`) used to find "22/278 files >10 s off Spotify, 11
  >60 s" (Cycles 414 s vs 186 s). Once the library is reprocessed, the
  wrong-version check has nothing to compare against.
- **ISRC.** It is never stored. The playlist `fields=` filter leaves out
  `external_ids` (`spotify/search.py:181-182`). spotDL writes `TSRC` into the
  mp3s it makes (spotDL 4.4.3 `utils/metadata.py:238`), but
  `extract_metadata` reads only title, artist, album, date and length
  (`ingest.py:96-130`). Spotify marked `external_ids` removed in its
  [February 2026 changelog][sp-feb], then reverted that in [March 2026][sp-mar],
  so the field is available.
- **Download source.** Never stored. Main's in-app path downloads YouTube's first
  search result (`csv_importer.py:189`) and keeps no URL. The acquisition branch
  writes the URL into a tag comment only. spotDL puts its YouTube URL in the
  mp3 `COMM` frame (`metadata.py:251-252`).

## Proposed columns

All are nullable or defaulted, with no FKs and no CHECKs. Enums are plain
strings, like `Region.source` (`database.py:467`).

| table | column | type | null | written by | meaning |
|---|---|---|---|---|---|
| `audio_analysis` | `analyzer_version` | INTEGER | yes | analyze / analyze_stems | the writing stage's hand-bumped constant; NULL = unknown |
| | `grid_source` | VARCHAR(16) | no, default `'analyzer'` | analyze; later Live/UI | `analyzer` \| `ableton` \| `manual`: who owns `bpm` |
| | `analyzer_bpm` | FLOAT | yes | analyze | the analyzer's own estimate, always written (the rule-3 hint) |
| | `loudness_lufs` | FLOAT | yes | analyze / analyze_stems | integrated loudness, ITU-R BS.1770 / EBU R128 |
| | `loudness_range_lu` | FLOAT | yes | same | LRA, free from the same call |
| | `true_peak_dbtp` | FLOAT | yes | same | max over channels, 4× oversampled |
| `tracks` | `isrc` | VARCHAR(12), non-unique index | yes | ingest, backfill | ISRC of the recording the row was *requested as* |
| | `source_url` | TEXT | yes | acquisition, ingest, backfill | URL the audio bytes came from (not the Spotify URL) |
| | `expected_duration_seconds` | FLOAT | yes | ingest, backfill | Spotify's duration at request time; never overwritten |

Naming follows the existing `duration_seconds`. Semantics worth pinning down:

- **`analyzer_version`.** Version 1 is the pre-5b9a65f analyzer, tagged by the
  migration (below). Version 2 is 5b9a65f; it never stamped anything, so its
  rows stay NULL. Version 3 is the first stamped version and adds loudness.
  `ANALYZE_VERSION` and `ANALYZE_STEMS_VERSION` sit next to their stages; bump
  them whenever stored numbers change meaning. Library versions are logged, not
  stored.
- **`grid_source`.** It lives on the full-mix row. Stem rows copy it together
  with `bpm` (`analyze_stems.py:106-145` already copies `bpm`). When the mix row
  is not `analyzer`, analyze writes `analyzer_bpm` and leaves `bpm` alone.
  `detect_regions` rebuilds beats from `bpm` (`detect_regions.py:109`), so beats
  follow whoever owns it. On day one every row is `analyzer`, because nothing
  writes `ableton` or `manual` yet. The likely writers are warp-guard Option C
  (`scripts/warp_probe.py --repair`) and the grid harness on
  `claude/grid-metrics`. The guard has to exist before the first of them lands,
  or the next reprocess clobbers their grids.
- **Loudness.** Measure what Live plays: a stereo decode at the file's own rate
  (`es.AudioLoader`). A mono file is measured as dual-mono, which reads 3.01 dB
  higher than a single channel would (measured: −22.99 vs −26.00 LUFS).
  Essentia returns exactly −70.0 for digital silence (measured), so store NULL
  at or below −70, which also covers silent stems. On gated stems such as
  vocals, `loudness_lufs` is the level of the audible part. `presence_ratio`
  already says how much of the track that part is.
- **`isrc` vs the file.** An ISRC names one recording, and edits and remixes get
  their own codes ([ISRC][isrc]). A radio-edit download filed under an extended
  mix's ISRC is exactly the d1b9222 failure. The file *is* that recording only
  when |`duration_seconds` − `expected_duration_seconds`| ≤ max(10 s, 3.5 %),
  the threshold already in `analyze.py:30-31`. Any code that uses `isrc` to mean
  "same recording" must check both. The index is non-unique: `duplicate_of`
  copies share an ISRC, and Spotify lists one recording under several track IDs
  (album and single). djbooth's backlog reached the same rule ("Never copy
  entry-level Spotify ISRC … to every variant"). Normalise to upper case without
  hyphens, and store NULL unless the value matches `^[A-Z]{2}[A-Z0-9]{3}[0-9]{7}$`.

## Options considered

**Provenance.**
- *A0: an `analyzed_at` cutoff* (no schema change). Rejected: the cutoff is when
  this machine ran the new code, not a property of the code, and it cannot say
  that stem rows were re-analysed while mix rows were not.
- *A1: one integer per stage.* Recommended.
- *A2: a provenance string* like `stem_files.model_used`
  (`htdemucs_ft|s2|pcm24|demucs4.0.1`, `database.py:278-280`). Richer, but every
  new Essentia dev build would mark the whole library stale. The string fits
  separation, where the model *is* the output; it does not fit analysis.

**Grid authority.**
- *G0: nothing.* Breaks rule 3 as soon as Live's grid is adopted.
- *G1: `grid_source` + `analyzer_bpm`.* Recommended. Every consumer keeps
  reading `bpm` unchanged.
- *G2: a `grids` table* (track, source, bpm, first beat, markers JSON). This is
  the right shape once variable-tempo warp markers are stored, but today it
  would be a new table with an FK and nothing writing to it. Revisit together
  with warp-guard Option C.

**Loudness placement.** Compute it inside `analyze` and `analyze_stems`. A
separate stage would need new `TrackState` values, which means changing the
enum and the dispatcher. For true peak, use `scipy.signal.resample_poly(x, 4, 1)`
(scipy is already installed as a librosa dependency) rather than
`es.TruePeakDetector`: they agree, and scipy is 8× faster (table below).

## Evidence (scratch, synthetic)

Loudness, with essentia 2.1-beta6-dev, scipy 1.17.0 and ffmpeg 8.1.1, on the
M2 Pro while other jobs were running:

| signal | essentia I | ffmpeg `ebur128` I | sample peak | TP essentia | TP scipy ×4 | TP ffmpeg |
|---|---|---|---|---|---|---|
| stereo 1 kHz sine −23 dBFS, 20 s ([EBU Tech 3341][ebu3341] test-signal shape; BS.1770 gives −23.0) | −22.99 | −23.0 | | | | |
| same, −33 dBFS | −32.99 | −33.0 | | | | |
| same, 48 kHz | −22.99 | | | | | |
| 6-min synthetic techno, 320k mp3 | −23.85 | −23.8 | | | | |
| fs/4 sine at 45°, true peak −0.01 dBTP by construction | | | −3.02 | +0.11 | +0.11 | +0.6 |
| 7 kHz sine, true peak −0.50 dBTP | | | −0.50 | −0.38 | −0.32 | +0.2 |

Essentia and scipy over-read the true peak by at most 0.18 dB; ffmpeg's
`ebur128` over-reads by 0.6–0.7 dB. Integrated loudness agrees with ffmpeg
within 0.05 LU.

Cost on a 6-minute stereo file, set against what analyze already spends:

| step | seconds |
|---|---|
| today's full-mix analyze (load 1.84 + BPM 3.30 + key 0.12 + energy 1.23) | 6.5 |
| mix loudness: stereo decode 0.50 + `LoudnessEBUR128` 1.59 + TP (scipy) 1.69 | 3.8 |
| …TP with `es.TruePeakDetector` instead | 13.62 |
| one stem: PCM24 read 0.15 + 1.59 + 1.69 | 3.4 (×4 = 13.7) |

That totals about 17.5 s per track. For the 244 visible tracks
(`library-duplicates.md`) it comes to about 71 min of CPU, or about 36 min at
the stages' `concurrency = 2`.

Migration, through dance's real Alembic chain, on a populated synthetic DB: 40
tracks, 160 stems, 200 analysis rows, 4,000 beats, plus edges, tags, plays and
set rows.

| scenario | result |
|---|---|
| S1: `e4a91c7b2d05` (rebuilds `tracks`), shipped env.py | all rows survive |
| S2: same, with `PRAGMA foreign_keys=ON` on the migration connection | `IntegrityError` at `DROP TABLE tracks`, because `session_plays`/`set_tracks` reference `tracks` with no `ON DELETE`. Nothing lost. |
| **S2b**: same, with no `session_plays`/`set_tracks` rows | **reports success.** `tracks` keeps 40 rows; `stem_files`, `audio_analysis`, `regions`, `track_embeddings`, `track_edges`, `track_tags`, `beats`, `phrases` all drop to **0** |
| S3: proposed revision with FKs forced **on** | 9 × `ALTER TABLE … ADD COLUMN` + 1 `CREATE INDEX`, no `_alembic_tmp_` table, all rows survive; the fingerprint tags 100 of 200 rows `analyzer_version = 1` |
| S4: proposed downgrade as batch `drop_column`, FKs on | fails the same way as S2 |
| S5: same as S4, shipped env | rebuilds `tracks` **and** `audio_analysis` (77 statements); safe only because FKs are off |
| S6: downgrade via native `ALTER TABLE … DROP COLUMN`, FKs on | 9 statements, no rebuild, all rows survive |
| S7: a revision that adds a column and then raises, shipped env | the column **stays**, `alembic_version` does not move, and re-running fails with `duplicate column name: isrc` |
| S8: S7, and S2b's chain, with the env.py hardening below | nothing left behind; the full chain keeps every row |

S2b is SQLite behaving as documented: with FKs on, `DROP TABLE` first runs an
implicit `DELETE` that fires `ON DELETE CASCADE` ([foreignkeys §5][fk]). S7 is
pysqlite behaving as documented: it never opens a transaction before DDL, so
each `ALTER` commits on its own ([SQLAlchemy][sa-txn]).

## Alembic notes

- **Stay on the plain ALTER path.** In batch mode, Alembic rebuilds the table
  for any operation other than `add_column` and `create_index`/`drop_index`, and
  also when a `server_default` is a SQL expression (`alembic/ddl/sqlite.py:49-75`
  in 1.18.4). So: no FKs, no CHECKs, and `server_default="analyzer"` as a plain
  string. That emitted `ADD COLUMN grid_source VARCHAR(16) DEFAULT 'analyzer'
  NOT NULL`. ADD COLUMN takes the same time at any table size; a CHECK would make
  SQLite scan every row ([ALTER TABLE][alter]).
- **Backfill that needs no audio goes in the revision.** That is
  `UPDATE audio_analysis SET analyzer_version = 1 WHERE (stem_file_id IS NULL
  AND bpm_confidence = 0.7) OR (stem_file_id IS NOT NULL AND (bpm_confidence IN
  (0.4, 0.7) OR bpm = 0))`, plus `analyzer_bpm = bpm` on mix rows. Anything
  that needs audio or the network stays out, following e4a91c7b2d05's precedent.
- **Downgrade with native `DROP COLUMN`** (SQLite ≥ 3.35; the venv links
  3.53.1). Drop `ix_tracks_isrc` first: SQLite refuses to drop an indexed
  column.
- **Harden `env.py`** (`run_migrations_online`, `env.py:61-79`). Today's safety
  rests on two accidents: env.py builds its own engine with `engine_from_config`,
  which skips dance's `PRAGMA foreign_keys=ON` listener (`database.py:772-779`),
  and SQLite's compile-time default leaves FKs off. Three changes, verified in S8:
  1. Set `PRAGMA foreign_keys=OFF` explicitly on connect. The PRAGMA is ignored
     inside a transaction ([foreignkeys §2][fk]), so a revision cannot do this
     itself.
  2. Get real transactional DDL from SQLAlchemy's documented recipe: set
     `isolation_level = None` on connect and emit `BEGIN` on the `begin` event
     ([SQLAlchemy][sa-txn]).
  3. Run `PRAGMA foreign_key_check` before and after the migration and fail if
     the count grew. Comparing counts, rather than demanding zero, means
     violations already in the DB cannot block migrations.
- **New test, `tests/test_alembic_populated.py`.** Populate every table at
  `b7d4e2f1a9c3`, in two variants: with plays/set rows and without (the S2b
  shape). Upgrade to head, `downgrade -1`, `upgrade head`. Assert that per-table
  row counts are unchanged and `foreign_key_check` is empty. A class-level
  `event.listens_for(Engine, "before_cursor_execute")` also catches env.py's own
  engine, so the test can assert that the new revision never creates an
  `_alembic_tmp_` table. Today nothing checks that data survives: despite
  `docs/dev.md:148` saying it walks "up and down", `tests/test_alembic.py` only
  upgrades an empty DB.

## Backfill

| column | source | how |
|---|---|---|
| `analyzer_version`, `analyzer_bpm`, `grid_source` | fingerprint / `bpm` / default | in the revision |
| `loudness_*`, `true_peak_dbtp`, `analyzer_version = 3` | audio | `dance reprocess --from analyze --all --skip-embeddings` (already needed for the BPM fix; Demucs short-circuits), plus a new `--stale` flag that selects rows whose version is NULL or below the stage constant |
| `isrc`, `expected_duration_seconds` | Spotify `GET /tracks/{id}` (listed as "still available" in [Feb 2026][sp-feb]); else spotDL's `TSRC` tag; else spotDL's sync file (`songs[].isrc`, `duration`, `song_id`) | new `dance backfill-identity`, dry-run by default, `--apply` (same pattern as `dance dedupe`) |
| `source_url` | `COMM` frame of spotDL mp3s | same command. Rows from main's in-app yt-dlp path stay NULL; that URL was never kept |

Plain mutagen 1.47 reads spotDL-style tags back: `EasyID3['isrc']`,
`ID3.getall('WOAS')` and `ID3.getall('COMM')` all worked on a tagged synthetic
mp3.

**For `expected_duration_seconds`, the order matters.** Rows not yet
re-analysed still hold their ingest-time duration, and for app-ingested rows
that is Spotify's. The reprocess destroys it. So before running any `dance
reprocess --from analyze`, take this snapshot (read-only, plus your own backup
file; no approval needed):

```bash
mkdir -p ~/.dance/backups
sqlite3 -readonly -csv -header ~/.dance/dance.db \
  "SELECT id, spotify_id, duration_seconds, file_path FROM tracks" \
  > ~/.dance/backups/durations-pre-reprocess.csv
```

Then:

1. Stop the backend and run `sqlite3 ~/.dance/dance.db ".backup
   ~/.dance/backups/dance-pre-provenance.db"`. `.backup` includes the WAL; a
   plain `cp` of a WAL database can miss committed pages.
2. `alembic upgrade head`.
3. `dance backfill-identity --apply`.
4. The reprocess.

The snapshot and `.backup` commands were both checked on a scratch WAL
database: the snapshot sees rows still sitting in the `-wal` file, and the
backup passes `integrity_check`.

To size the work first (read-only):

```bash
sqlite3 'file:/Users/arya/.dance/dance.db?immutable=1' "SELECT CASE WHEN stem_file_id IS NULL
  THEN 'mix' ELSE 'stem' END, SUM((stem_file_id IS NULL AND bpm_confidence = 0.7) OR
  (stem_file_id IS NOT NULL AND (bpm_confidence IN (0.4, 0.7) OR bpm = 0))), COUNT(*)
  FROM audio_analysis GROUP BY 1"
```

## API / UI exposure

**None in this change.** `AnalysisOut`, `StemAnalysisOut` and `TrackOut`
(`api/schemas.py:39-62, 118-126`) stay as they are. Each consumer adds its own
fields in its own proposal: clip gain on load, a "wrong version?" badge
(expected vs measured duration), a grid-owner marker. `dance doctor` gets one
read-only line: "N analysis rows older than the current analyzer → `dance
reprocess --from analyze --stale`".

## What NOT to add

- Momentary/short-term loudness curves. `rms_curve` already exists, and at 10 Hz
  a curve is about 3,600 floats per row × 5 rows per track.
- Sample peak, PLR, ReplayGain values and per-stem gain offsets; all of them can
  be derived. Clip gain must be **one value per track** (target − mix
  `loudness_lufs`) applied to all four stems alike; normalising stems
  individually would re-balance the mix.
- A `source` enum (spotdl / yt-dlp / local): the host in `source_url` already
  says it.
- Match score, video ID, upload title: they stay in the acquisition job message
  and the tag comment.
- A unique constraint on `isrc`, or FKs / CHECKs on any of these columns.
- Spotify `popularity` / `available_markets`: both removed from the API in
  [February 2026][sp-feb].
- New API fields (see above).

## Recommendation

Add all nine columns in **one** revision, using plain ALTERs and a downgrade
that uses native DROP COLUMN. Ship the `env.py` hardening and the populated-DB
test **in the same PR**. The revision itself is safe even with FKs on (S3), but
the next migration that carries an FK or CHECK will rebuild a table, and S2b
shows that going wrong silently. Compute loudness inside the two analyze stages,
using scipy for true peak. Stamp `analyzer_version = 3`. Put the `grid_source`
guard in now, before anything writes a non-analyzer grid. Take the duration
snapshot **today**, before anyone reprocesses the real library.

## Decisions for the user

1. **All three groups in one revision**, or loudness first and identity later?
   (Recommended: one. Each extra revision means another upgrade and another
   reprocess.)
2. **Add `grid_source` + `analyzer_bpm` now**, before anything writes `ableton`
   or `manual`, or together with warp-guard Option C? (Recommended: now.
   Otherwise the first grid adopted from Live is lost to the next reprocess.)
3. **`analyzer_version` as an integer** (A1) or a provenance string (A2)?
4. **True peak on stems too** (≈ 6.8 s/track), or on the mix only? Rule 4 argues
   for stems, since a soloed stem gets the track's gain as well.
5. **`env.py` hardening in this PR**, or separately?
6. **Identity source:** a Spotify re-fetch (needs credentials; authoritative), or
   tags + the spotDL sync file only (works offline)?
7. **ISRC on a probable wrong-version file:** store it and rely on the
   documented duration check (recommended), or leave it NULL until the
   durations agree?

## Migration cost

- **Schema:** 1 revision of 13 statements (9 ADD COLUMN, 1 CREATE INDEX, 3
  backfill UPDATEs on `audio_analysis`). No table is copied, and ADD COLUMN
  takes the same time at any table size.
- **Code (estimate, lines):** models 25; `pipeline/utils/loudness.py` 60;
  stamping + grid guard 25; ingest 15 (`pipeline.py:528, 714` also write
  `expected_duration_seconds`, `extract_metadata` reads `TSRC`/`COMM`,
  `fields=` adds `external_ids`); `dance backfill-identity` 120;
  `reprocess --stale` 30; `env.py` 25; tests 250.
- **Data:** one reprocess. That is about 36 min of loudness CPU on top of the
  analysis it re-runs anyway, plus one Spotify call per track with a
  `spotify_id` (at most 357). Get Several Tracks is gone, so there is no
  batching.

## Verification plan

1. **Unit tests, synthetic:**
   - stereo 1 kHz −23 dBFS sine reads −23.0 ± 0.1 LUFS; silence stores NULL;
     dual-mono vs one channel differs by 3.01 dB
   - the fs/4 sine at 45° reads −3.02 dBFS sample peak and ≈ 0 dBTP true peak
   - a mix row with `grid_source='ableton'` and `bpm = 125.0`, run through the
     stage on a 128-BPM click track, keeps `bpm` at 125.0 and gets
     `analyzer_bpm` ≈ 128
   - the fingerprint SQL tags hand-made v1 rows and leaves v2 rows alone
   - `tests/test_alembic_populated.py`, as described above
2. **Real data (rule 2)**, on a `.backup` copy in a throwaway install with HOME
   and `DANCE_*` redirected, as f2bb065 did:
   - record per-table counts, run `alembic upgrade head`, and compare; the
     counts must be identical
   - run `dance reprocess --from analyze -t <3 ids> --skip-embeddings`; check
     each mix's `loudness_lufs` against `ffmpeg -af ebur128=peak=true`
     (|ΔI| ≤ 0.2 LU) and confirm `analyzer_version = 3`
   - run `dance backfill-identity` as a dry-run on the copy: the ISRC count,
     and how many rows exceed the duration threshold
   - check that `dance doctor` reports head

   Live is not involved until the clip-gain consumer, which changes `.als`
   emission and needs its own in-Live check.

Every number above comes from scratch scripts kept outside the repo
(`loudness_probe.py`, `migration_probe.py`, `tags_and_cost_probe.py`), which
use synthetic audio and synthetic DBs only.

[lom]: https://docs.cycling74.com/apiref/lom/clip/
[sp-feb]: https://developer.spotify.com/documentation/web-api/references/changes/february-2026
[sp-mar]: https://developer.spotify.com/documentation/web-api/references/changes/march-2026
[isrc]: https://en.wikipedia.org/wiki/International_Standard_Recording_Code
[ebu3341]: https://tech.ebu.ch/publications/tech3341
[fk]: https://www.sqlite.org/foreignkeys.html
[alter]: https://www.sqlite.org/lang_altertable.html
[sa-txn]: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#enabling-non-legacy-sqlite-transactional-modes-with-the-sqlite3-or-aiosqlite-driver
