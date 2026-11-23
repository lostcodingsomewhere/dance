# Cue pads, a prepared Set, and version families

**Status:** PROPOSAL, not implemented. Waiting for your go-ahead (CLAUDE.md rule 1: it changes the `.als` writer's emission shape, the OSC load contract, and API schemas).
**Date:** 2026-11-23
**Touches:** `src/dance/pipeline/stages/detect_regions.py`, `src/dance/core/database.py` (one enum value), `src/dance/als/writer.py` + `generator.py`, `src/dance/osc/bridge.py` (`push_track_to_live`), `src/dance/api/routers/{ableton,sets,tracks}.py` + `schemas.py`, `src/dance/recommender/dedup.py`, `companion-app` (`SceneGrid`, `TwoDeckStrip`, `lib/roles.ts`), `cli.py`.
**Stacks on:** [`als-clip-elements.md`](als-clip-elements.md) option A, [`warp-guard.md`](warp-guard.md) (who owns the grid), and the library re-run that the tempo fix already requires.
**Related:** [`../vision.md`](../vision.md), [`../dj_ux_flow.md`](../dj_ux_flow.md), [`stem-deck-pair.md`](stem-deck-pair.md), [`hardware-and-live-controls.md`](hardware-and-live-controls.md), [`analysis-provenance-loudness-identity.md`](analysis-provenance-loudness-identity.md), [`set-plan-revision.md`](set-plan-revision.md)

---

## What this is for

The style you described on 2026-11-23: remix live by layering stems from different
songs, lean on an original and its remixes to mix and transition, set the night up
quickly ahead of time, and play it from the APC40 once it is set up.

Three things in the build stand in the way of that, and each needs the others:

1. **A pad is a whole stem from 0:00.** There is no pad for a hook, a drop, or eight bars of drums.
2. **The plan never reaches Live.** Prep produces lists of track ids. Every clip is still created during the set, which is when Live's auto-warp goes wrong.
3. **Nothing knows a remix belongs to its original.**

`LEARNING.md` says no new features before the first play-out. Steps 1 to 5 below are
features by that rule. The case for doing steps 1 and 2 first is in the numbers: a
first play-out on today's pads would test a layout this proposal replaces.

## The problem, measured

Read-only queries on `~/.dance/dance.db`, 2026-11-23, over the 244 complete tracks
not marked `duplicate_of`. Presence comes from `audio_analysis.presence_intervals`
per stem. Sections were cut with the pre-fix tempos, so these move a little after the
re-run.

**How long a pad is silent after you press it.** Time from 0:00 until the stem first
plays for a sustained stretch (present for at least half of an 8 s window):

| stem | median | 75th pct | 30 s or more | never sustained |
|---|---|---|---|---|
| drums | 2.0 s | 23.6 s | 44 of 237 (18%) | 7 |
| bass | 18.8 s | 41.0 s | 102 of 239 (42%) | 5 |
| vocals | 30.0 s | 74.8 s | 113 of 226 (50%) | 18 |
| melody (`other`) | 5.6 s | 27.5 s | 60 of 243 (24%) | 1 |

Today's load creates one clip from the full stem file and sets warp, nothing else
(`bridge.py:2402`). A vocal pad gives you a median of 30 s of nothing, about 15 bars
at 124 BPM.

**The data to fix it already exists.** `detect_regions` stores 3,256 sections on
these tracks (median 12 per track, 10th to 90th percentile 8 to 21) and 14,131
eight-bar loops. Nothing in `src/dance/osc` reads a `Region` row. The `.als` export
turns track-level regions into arrangement locators and drops per-stem ones
(`als/markers.py:67`).

**A section is not always a useful pad.** Share of (stem, section) pairs where the
stem is present for under 20% of the section: drums 16%, bass 23%, melody 11%,
**vocals 46%** (1,493 of 3,256). Vocals are absent in 554 of 1,051 sections labelled
`drop`. A grid that put every stem on the same five sections would leave close to
half the vocal pads silent.

**Five rows cannot hold a song's sections.** 3 of 244 tracks have five sections or
fewer. Merging neighbouring sections with the same label gives a median of 7, and 34
tracks at five or fewer. The pads need a chosen five.

**The plan and the hand-off.** A plan is `{role: [track_id]}` (`core/set_queues.py`).
It names whole tracks, has no order across roles, and no action loads or exports it:
`dance export-als` writes one track per file. The one set in the database holds one
pick.

**Loading during the set is the unreliable path.** On 5 tracks measured on
2026-08-01, 4 had at least one stem at the wrong tempo after Live's auto-warp, and
the real guess lands 13 to 15 s after the load (`warp-guard.md`).

**Versions.** Grouping by base title with at least one shared artist finds 26 groups
of two or more distinct recordings, 54 of the 244 rows. Only 3 hold an original next
to a named remix or club mix (*Surrender*, *Shallow Water*, *SOMEWHERE ELSE*). The
other 23 are the same recording twice, mostly as a long and a short cut and in a few
cases under two spellings. The download matcher
rejects any remix, VIP or edit the playlist did not name
(`spotify/matching.py:122`, `HARD_VERSION_MARKERS`), so a family only exists when you
added each version yourself.

**How working DJs do the same thing.** Guides to DJing in Live agree on the prep:
warp every track beforehand and correct it by ear, cut a song into several clips
with different start points, and keep the night in one Set
([DJ TechTools](https://djtechtools.com/2012/04/18/how-to-start-djing-in-ableton-part-1/),
[Sound On Sound](https://www.soundonsound.com/techniques/djing-ableton-live),
[MusicRadar](https://www.musicradar.com/tuition/dj/how-to-prepare-a-track-for-djing-with-ableton-live-566696)).
Stems of one song share one set of warp markers, set once
([Ableton forum](https://forum.ableton.com/viewtopic.php?p=1177751)). The best-known
APC40 DJ template is reported to give each channel five cue pads
([DJ TechTools forum](https://forum.djtechtools.com/t/ill-gates-template-for-pad-controllers/25595)).
DJs on decks with built-in stems work each loaded song with hot cues, loops and stem
mutes.

## Terms

- **Cue:** a start point inside a stem, on a bar line, with a section label.
- **Cue pad:** a clip that starts its stem at one cue.
- **Page:** five consecutive scene rows, which is what the APC40 shows at once.
- **Prepared Set:** one `.als` holding every clip for the night, already warped.
- **Family:** the recordings of one song, meaning the original, its remixes, and its long and short cuts.

## Part 1: What a pad is

### Options

**A. A whole stem from the top (today).** A row is a song. Up to eight songs can
sound at once, one per column. Costs are in the table above.

**B. A column holds one stem of one song, cut at five cues. ← recommended.** The
five rows of a page are that stem's five cues in time order. A deck is four
neighbouring columns and usually holds one song. Any single column can be given a
stem from another song, which keeps the vision's "drums from one track, vocals from
another". This is the hot-cue model that deck DJs and the APC40 templates use, laid
on the pad grid.

**C. Every pad is a part you pick by hand in prep.** A row is a moment of the set
and holds whatever parts you placed there. This is how producers perform their own
material. It is the most expressive and the slowest to prepare, because every pad
is a decision. It can be built on B's clips later.

### What B looks like

Original on deck A, its remix on deck B, with the column order from decision 5:

```
            DECK A (original)                 DECK B (remix)
            Drums  Bass   Vocals Melody       Drums  Bass   Vocals Melody
  row 1     intro  intro  v.1    intro        intro  intro  v.1    intro
  row 2     build  build  hook   build        build  build  hook   build
  row 3     DROP   DROP   v.2    DROP         DROP   DROP   v.2    DROP
  row 4     break  break  hook   break        break  break  bridge break
  row 5     drop2  drop2  outro  drop2        drop2  drop2  hook   drop2
  faders    1      2      3      4            5      6      7      8
```

Remix drums under the original's hook is pad (row 3, Drums B) plus pad (row 2,
Vocals A), with faders 1 and 5 crossing.

### Which five cues

Stored as `Region` rows with a new `region_type` of `pad`, one set per stem and one
for the mix. Picked in `detect_regions`, which already holds everything needed.

- **Drums, bass, melody and the mix share one list per track**, so their rows line
  up: the intro, the first build-up, the loudest drop, the longest breakdown after
  it, and the last drop (the outro when there is only one drop).
- **Vocals get their own list**: the five longest vocal entrances, an entrance being
  where the stem turns present after at least two bars of silence, moved back to the
  bar line. Fewer than five when the song has fewer. Instrumentals get none.

These rules are a starting point to tune by ear. `source` is `auto`; a pad you move
by hand becomes `manual`, and `_clear_auto_data` already leaves manual rows alone on
a re-run (`detect_regions.py:144`).

### What happens when the section ends

`vision.md` locks "stems loop by default" with an override for vocals. Three choices:

| | Drums, bass, melody | Vocals | Song (mix) |
|---|---|---|---|
| **Hold per role ← recommended** | loop the section, cut to 4, 8 or 16 bars | play on | play on |
| Everything plays on | play on | play on | play on |
| Everything holds | loop | loop | loop |

Hold per role matches the vision and suits layering two versions, because a remix's
drums keep their groove while the original's vocal runs its verse and chorus.
"Everything plays on" behaves like DJ decks and surprises less on a first set, but
two arrangements then run into each other's breakdowns. It is a clip property set in
one place, so the default can be a setting and flipped after hearing both.

### Pad colour

The pads are the only display on the controller. A column's position already says
which role it is, so colour is free for something else.

- **Recommended:** hue for the section type (intro and outro, build-up, drop,
  breakdown), and a dim shade when the stem is silent for the pad's first eight bars.
- **Today:** one colour per role.

The app keeps role tints on screen. The exact palette indices need a look at the
hardware.

### Column order

Today the columns interleave roles: Drums A, Drums B, Bass A, Bass B, and so on
(`bridge.py:872`, `writer.py:85`, `roles.ts:41`). That puts a role's two decks on
neighbouring faders.

With B, the most common press is "start this deck at cue 3", which is four pads in
one row. Interleaved, those are pads 1, 3, 5 and 7. Grouped by deck they are four
neighbouring pads for one hand, faders 1 to 4 are deck A, and a role swap is the
same finger on each hand. The scene button fires both decks' cue, which is rarely
wanted, and the stock APC40 script cannot fire half a row.

- **Recommended with B:** group by deck.
- **Cost:** it reverses the 2026-06-13 choice. It is cheap now and expensive once
  your hands have learned an order. Three exercises in the course already assume the
  grouped order (`docs/course/index.html` lines 849, 883, 902).

### Paging

A song takes one page per deck load. Loads fill the page the app shows as current
and move to the next page when the target column is occupied, as
`_free_stem_slot` does for rows today (`bridge.py:1806`). The app shows a page strip
and highlights the page that is sounding. It cannot know which rows the hardware
window is on. Arrows move the window one row; a five-row jump should come from the
Bank button or the Shift overview and has to be confirmed on the device.

## Part 2: The prepared Set

### Options

**A. Keep creating clips during the set (today).** Every load waits on Live's
auto-warp and can come back wrong.

**B. `dance export-set` writes the whole plan as one `.als`. ← recommended.** Every
clip carries its warp grid and its cue markers in the file, so nothing is analysed
at the gig. This is the path where the grid travels with the clip, which
`als-clip-elements.md` identified as the one way to skip auto-warp.

**C. The app builds the Set over OSC before the gig.** It reuses the load path. A
20-song plan is 80 stems at 13 to 15 s of analysis each, up to about 20 minutes. It
also depends on the unverified warp-marker handlers, and the result lives in an
unsaved Set.

### Layout, from the plan as it is

No change to the plan's shape in this proposal.

- **Songs fill pages.** The Song queue in order: first song on deck A of page 1,
  second on deck B, third on deck A of page 2. Queue an original then its remix and
  they share a page.
- **Single-stem picks go on spare pages after the songs**, two per role per page.
  Firing a spare pad replaces that role on its deck, and it keeps playing when you
  page back.
- **The Song A and Song B tracks** hold each song's full mix with the same five cues,
  playing on. They are tracks 9 and 10, outside the eight pad columns, so reaching
  them from the controller means banking right.
- The first row of each page is named with both titles. The Set's tempo is the first
  song's.

A 20-song plan is about 20 × 4 stems × 5 cues = 400 stem clips over 80 files.

### The grid

One function returns a track's tempo and the time of its first bar line, and both
the writer and the bridge call it. Every clip of a song gets the same markers, which
is the usual practice for stems. The values come from the analyser after the re-run.

Two risks, both stated in `als-clip-elements.md` and both still open:

- Whether Live keeps a saved grid or re-runs auto-warp has not been measured. Its
  verification steps 1 and 2 answer that and must come first.
- Bar lines are assumed on every fourth detected beat (`detect_regions.py:165`). A
  wrong guess puts every cue of that song the same number of beats off. Step 2's
  listening pass will show how often this happens.

### Warp mode per role

Both paths pin every clip to Beats (`writer.py:526`, `bridge.py:1841`). Beats suits
drums. When an original and its remix differ in tempo, a vocal stretched in Beats
mode sounds chopped.

- **Recommended:** Beats for drums and bass, Complex Pro for vocals, Complex for
  melody and the mix. The CPU cost with eight stems playing is not measured.

### Loading during the set still works

A live load builds the same five clips per column over OSC, from the same cue and
grid functions. It needs the grid set through the warp-marker handlers
(`vendor/abletonosc` patch `0005`, installed, never run against Live). Five
`create_audio_clip` calls per stem may each trigger analysis. If they do, a
`duplicate_clip_to` handler is one more small patch; the installed script does not
expose it.

## Part 3: Version families

### Options

**A. Compute the family when asked. ← recommended.** A pure function beside
`title_key` in `recommender/dedup.py`: same base title (brackets, "- X Remix" tails
and featuring credits removed), at least one shared credited artist, and a different
title or a length more than 5 s apart. No schema change. It is fast at this library's
size and finds the 26 groups above. The shared-artist test is what keeps Duke
Dumont's *Losing Control* apart from Odd Mob's.

**B. Store a family key on `tracks`.** An Alembic revision, and a value that goes
stale whenever a title is corrected. Worth it only if the library grows past what a
scan per request can serve.

**C. External ids** (ISRC, MusicBrainz works). ISRC differs per version and gives no
link between them. MusicBrainz covers club remixes patchily.

### Where it shows

- `GET /tracks/{id}/versions`, and a `version_count` on track payloads.
- **Plan grid:** other versions of anything already in the plan lead the Song
  column's suggestions with a "version" chip, and one action queues a version right
  after its original so they share a page.
- **⌘K:** results grouped by family.
- **Booth:** with a song on a deck, its versions lead the Song column with a load
  button for the other deck.

### Getting the versions in

With three true families in the library, the read side has little to show. A "find
versions" action would search Spotify for the same title and artist, list results
whose titles carry a version word (`api/dedup.py` `VERSION_WORDS`), and queue the
ones you choose as explicitly requested, which is what lets the matcher accept a
remix. It needs Spotify's track search. The playlist reader in the same module still
calls an endpoint Spotify removed (`spotify/search.py:216`), so that module needs
work first and this is its own step.

## What changes, by contract

| Contract (rule 1) | Change |
|---|---|
| SQLAlchemy models | One value added to `RegionType`: `pad`. `region_type` is `String(16)` with no constraint, so no Alembic revision. |
| Stage protocol | None. `detect_regions` emits more rows. |
| `.als` emission | Several clips per deck track with their own start and loop markers, warp mode per role, named scenes, more than one song per Set. A new export beside the per-track one, which stays. |
| OSC load contract | A load creates five clips per column and sets `start_marker`, `loop_start`, `loop_end`, `looping`, `color_index` (all handled by the installed script). Clip names stay `{title} ({kind} {side})` so `/decks/resync` keeps working (`ableton.py:701`); the cue is the row within the page. |
| API schemas | Deck cells gain a cue number and label. New `POST /sets/{id}/als` and `GET /tracks/{id}/versions`. `load-track` accepts a page. |
| Plan shape | Unchanged. |
| Column order (if decision 5 is yes) | `_DECK_KINDS`, `DECK_ORDER`, `TWO_DECK_COLUMN_ORDER`, the template's track order, docs and course. |

## Order of work

Each step lands on its own and is checked in Live before the next one starts.

0. **Prerequisites, already written up.** Back up the database. Run
   `dance reprocess --from analyze --all --skip-embeddings`, `dance embed --all`,
   `dance tag`, `dance build-graph` (take the provenance proposal's duration snapshot
   first if that proposal is going ahead). Run `als-clip-elements.md` verification
   steps 1 and 2, then land its option A.
1. **Cues in data.** The picker, the `pad` rows, the API field, and five cue flags on
   the waveform. Nothing changes in Live. Check: read the cues for ten songs you
   know and listen to each start point.
2. **`dance export-set`.** Pages, cue clips, the shared grid, warp modes, and the
   column order if chosen. Check: the Live session in the verification plan.
3. **The Booth reads pages.** `SceneGrid` shows the current page with cue labels,
   resync adopts a prepared Set, deck play takes a cue.
4. **Live loads build the same clips.** Only after patch `0005` is seen working.
5. **Families, read side.** Then "find versions" once a set has been played.

## Decisions for you

1. **What a pad is:** A, B or C. I recommend B.
2. **Section end:** hold per role, everything plays on, or everything holds. I
   recommend hold per role, as a setting.
3. **The five-cue rule**, including separate cues for vocals. I recommend yes to both.
4. **Pad colour:** section type with a dim shade for silent stems, or role colour as
   today. I recommend section type.
5. **Column order:** grouped by deck, or interleaved by role as today. I recommend
   grouped if decision 1 is B. Otherwise leave it.
6. **Prepared Set:** export (B), build over OSC (C), or neither. I recommend B.
7. **Single-stem picks in the Set:** spare pages after the songs, in place of a
   song's own stem on its page, or left out. I recommend spare pages.
8. **Warp mode per role**, or Beats everywhere as today. I recommend per role, judged
   by ear at a 6% tempo change.
9. **Families:** computed (A), stored (B), or external ids (C). I recommend A.
10. **"Find versions":** build it with step 5, or after the first played set. I
    recommend after.
11. **Scope of the go-ahead.** Step 0 includes landing `als-clip-elements.md` option
    A and re-running the library. Say if a yes here covers those.

## Migration cost

- **Backend:** the picker in `detect_regions.py` (about 120 lines); a multi-clip,
  multi-song path in `writer.py` and `generator.py` (about 200, on top of the
  conformance patch); `push_track_to_live` creating five clips per column (about 80);
  the family function and endpoint (about 90); CLI and API wiring (about 60). These
  are estimates.
- **Frontend:** `SceneGrid` paged with cue labels, cue flags in `TwoDeckStrip`, a
  version chip and grouping in `RoleColumn` and `CommandBar`. About 300 lines.
- **Data:** none migrated. Pads appear when regions are regenerated, which the
  step 0 re-run does for every track. Existing `.als` exports stay as they are.
- **Docs:** `vision.md` items 3 and 6, `dj_ux_flow.md`, `als-export.md`,
  `live-compat.md` (matrix rows and a log line), `CLAUDE.md` (grid description and
  the "let's DJ" row), and the course's fader map.
- **You:** one session of about 45 minutes with the APC40 and the interface
  connected, plus the 20 to 30 minutes `als-clip-elements.md` already asks for.

## Verification plan

**Tests.** End to end, through real entry points. Each is written first and seen to
fail on today's code.

- `dance export-set` and `POST /sets/{id}/als` on a library built by the real
  pipeline from the synthetic audio fixture (Demucs and CLAP stubbed, as
  `tests/test_pipeline_e2e.py` does). Read the written file back and assert what
  Live would receive: pages in plan order, five clips per column, start and loop
  positions in beats on bar lines, the same grid on every clip of a song, the warp
  mode per role.
- `POST /ableton/load-track` against a UDP listener standing in for Live, which is
  the one system outside ours. Assert the messages for five clips and that a playing
  cell is never replaced.
- `GET /tracks/{id}/versions` on a seeded library holding an original, its remix, an
  extended cut, and a same-title track by another artist.
- The pipeline run on a synthetic track with a silent vocal intro, with the cues
  read back through `GET /tracks/{id}/regions`. No vocal cue starts in the silence.
- One UI flow in the existing vitest harness: load a song on deck A, see five cue
  cells on the page, fire one, the right slot is requested.

**In Live 12.4 Standard** (`open -a "Ableton Live 12 Standard"`, after `dance doctor`):

1. Export a set of four songs that includes one of the three true families. It
   opens with no dialog and shows two pages.
2. On each deck, fire all five rows in turn. Every pad starts on a bar line and at
   the cue you expect. Note any song whose cues sit a beat or two off.
3. A drums pad loops cleanly at its length. A vocal pad plays on.
4. Read every clip's `warping`, `warp_mode`, `start_marker`, `loop_start` and
   `length` over OSC one second and twenty seconds after opening. Nothing changes
   between the two reads.
5. Raise the master tempo by 8 BPM. Every stem follows. Compare a vocal in Complex
   Pro against Beats.
6. On the APC40: the section colours are distinguishable, dim pads read as dim, four
   neighbouring pads start a deck, and you can move a page at a time.
7. Record the result in `docs/live-compat.md`.

## Not in scope

- An explicit page-by-page arrangement in the plan. The derived layout comes first;
  a new plan shape would get its own proposal.
- Moving or renaming cues in the app. The `manual` source is reserved for it.
- Hold and release, loop length, or loading from the controller. Each needs a MIDI
  listener beside the OSC bridge.
- A grid check with a click and nudge keys, and reading warp fixes back from a Set
  you saved in Live (`als/reader.py` could do the second).
- A filter or EQ on each deck track (`hardware-and-live-controls.md` section 2).
- The stem model, loudness matching, and identity fields. Those have their own
  proposals.
- Flagging two rows that hold the same audio under different titles, such as ids 79
  and 176.

Per CLAUDE.md, implementation waits for your go-ahead.
