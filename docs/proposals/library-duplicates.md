# 41% of the library is duplicate recordings

**Status:** DONE (audio-verified marking applied); disk reclaim still open
**Date:** 2026-08-01

---

## What was measured

The library was built up over three ingest runs — 2026-05-17, 2026-05-26,
2026-06-13 — which pulled overlapping sets of songs under slightly different
filenames. Different bytes → different `file_hash` → ingest's hash-based dedup
never fired, and each copy became its own `Track` row.

```
complete tracks                                  353
groups sharing a normalized title+artist          82   (225 tracks)
redundant copies                                 145   = 41% of the library
  of which the SAME recording (dur within 2s)    114 pairs
  of which genuine variants (extended vs radio)   29 pairs
```

A single song, three times, from three runs:

```
#67  Dot Major;Kitty Amor - Navi - Kitty Amor Remix.mp3   211.8s   2026-05-17
#246 Dot Major, Kitty Amor - Navi - Kitty Amor Remix.mp3  211.8s   2026-05-26
#374 Dot Major - Navi - Kitty Amor Remix.mp3              211.8s   2026-06-13
```

### What it costs

| | |
|---|---|
| Rec-list pollution | **5 of 40** slots across the five role columns were a literal repeat of another card in the same list (bass 2, song 2, other 1) |
| Disk | ~**25.8 GB** of the 63.6 GB stems directory (~184 MB/track) |
| Compute already spent | ~**16.7 hours** of Demucs separation on copies |
| Future | every `dance process` run re-separates them; every `export-als --all` writes duplicate `.als` files |

## Fixed: the rec surface

`recommender/dedup.py` collapses duplicate recordings in a rec list, applied
in `_ColumnRecommenderImpl.run` after the sort and before `[:k]` — so the copy
the scorer preferred survives and the freed slots go to real alternatives.

Verified against the live library: **5 repeated slots → 0**, lists still 8 deep.

Two design decisions worth recording, both driven by the data rather than
intuition:

- **The key is the normalized title, not title+artist.** Artist looks like the
  obvious second field but actively breaks the match here: the same release is
  credited three ways (`Dot Major;Kitty Amor` / `Dot Major, Kitty Amor` /
  `Dot Major`), so keying on it collapsed only 2 of those 3 copies. Measured
  before choosing: title+duration collapses 165 pairs library-wide, of which
  exactly **one** has artist strings that are not substrings of each other —
  `Layton Giordani, Linney, Sarah de Warren` vs
  `Layton Giordani/Sarah de Warren/Linney`, the same three people with a
  different separator. Zero genuine false positives.
- **Duration, not artist, is the guard**, at a tight ±5 s. That is what keeps
  `Manifesto (Extended Mix)` (348 s) visible alongside `Manifesto` (224 s) —
  different tools for different moments in a set. Hiding an extended mix behind
  a radio edit that happened to score higher would be worse than showing two
  cards.

## Not fixed: the durable cleanup

Dedup at the rec layer hides the symptom. It does not reclaim the 25.8 GB, stop
future runs re-separating copies, or de-duplicate `⌘K` search, `export-als
--all`, or the recommendation graph (`track_edges`, currently 61,508 rows built
over a library that is 41% redundant — so a meaningful share of those edges
point at a copy).

Three options, needing a decision:

### A. A `duplicate_of` column on `tracks` (schema change → this proposal)

`duplicate_of: int | None` referencing the canonical track. Ingest sets it;
a `dance dedupe` command backfills. Every consumer filters on it.

- Honest, reversible, keeps the files on disk until the user says otherwise.
- Cost: a migration, plus touching every candidate query.
- Question this raises: what is canonical? For DJing the **longest** duration
  is usually right (extended mix), but for a re-rip pair the durations are
  identical and the tie-break should probably be file size / bitrate.

### B. A `dance dedupe` command that reports, and deletes only on `--apply`

No schema change. Prints the groups, the disk it would reclaim, and the
canonical pick; `--apply` removes the redundant `Track` rows and their stems.

- Cheapest, and reclaims the disk immediately.
- Destructive and irreversible — and the redundant rows may be referenced by
  `set_tracks`, `session_plays`, and `track_edges`. Those need re-pointing at
  the canonical id, not orphaning.

### C. Fix ingest so it stops happening, and leave the existing mess

Match on (normalized title, duration) at ingest time in addition to
`file_hash`. Stops the bleeding; does nothing about the 145 copies already
there.

## Recommendation

**C then A.** Fix ingest first so a fourth run doesn't add a fourth copy of
everything — that is small, safe, and purely additive. Then A, because the
41 GB is not urgent and a reversible marker is much easier to live with than a
delete that has to rewrite `session_plays` and `track_edges` to avoid orphans.

B is tempting for the disk reclaim, but deleting rows that plays and graph
edges point at is exactly the kind of thing to do deliberately, not in passing.

**Do not run any of this immediately before a session.** The rec-layer fix
already removes the day-to-day annoyance; the rest is housekeeping.


---

## Update 2026-08-01 — metadata was not enough, and the audio proved it

Challenged on whether the "duplicates" were really the same song, the marking
was reverted and every candidate pair compared **by audio** before anything was
hidden. Two things came out of it:

**CLAP embeddings are the wrong tool for this.** Cosine over the stored
full-mix embeddings ranked the most identical pair in the library (#218/#354,
chroma 0.994) as the *least* similar (cosine 0.379). CLAP is semantic and
window-sensitive; it does not answer "is this the same recording".

**Fixed-offset comparison lies.** Copies from different sources carry different
leading silence, so sampling "the middle" of each lands on different bars. The
"Take Off" pair scores 0.715 unaligned, 0.822 aligned. Every comparison now
searches for the offset first.

Aligned chroma over all 125 candidates is strongly bimodal:

```
  0.60-0.70  # 1
  0.70-0.75    0
  0.75-0.80  # 1
  0.80-0.85  ### 3
  0.85-0.90  ###### 6
  0.90-0.95  ### 3
  0.95-1.00  ############################################ 111   (median 0.995)
```

So the threshold sits in real empty space, not on a guess. **111 marked, 14
left visible.** Those 14 share a title and a duration but are demonstrably
different audio — including `LA FAMA` at 0.696 and, notably, `Natural Blues`
at 0.852, which this document previously cited as an obvious duplicate. A
metadata-only cleanup would have hidden all 14.

`dance dedupe` now verifies audio before marking and is dry-run by default;
`--undo --apply` reverses it completely. Source audio is the right thing to
compare — stems are derived from it, so same recording means equivalent stems.

Final state: 357 rows intact, 111 marked, **244 visible complete tracks**,
`stem_files` and `session_plays` untouched.
