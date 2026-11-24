# Cue pads, a prepared Set, and version families

**Status:** APPROVED 2026-11-23, in progress. All eleven decisions are settled in "Decisions for you". Nothing is verified in Live yet.
**Date:** 2026-11-23
**Touches:** `src/dance/pipeline/stages/detect_regions.py`, `src/dance/core/database.py` (one enum value), `src/dance/als/writer.py` + `generator.py`, `src/dance/osc/bridge.py` (`push_track_to_live`), `src/dance/api/routers/{ableton,sets,tracks}.py` + `schemas.py`, `src/dance/recommender/dedup.py`, `companion-app` (`SceneGrid`, `TwoDeckStrip`, `lib/roles.ts`), `cli.py`.
**Stacks on:** [`als-clip-elements.md`](als-clip-elements.md) option A, [`warp-guard.md`](warp-guard.md) (who owns the grid), and the library re-run that the tempo fix already requires.
**Related:** [`../vision.md`](../vision.md), [`../dj_ux_flow.md`](../dj_ux_flow.md), [`stem-deck-pair.md`](stem-deck-pair.md), [`hardware-and-live-controls.md`](hardware-and-live-controls.md), [`analysis-provenance-loudness-identity.md`](analysis-provenance-loudness-identity.md), [`set-plan-revision.md`](set-plan-revision.md)

---

## Progress

As of 2026-11-23. Sources are the run logs of that day and the checks named here.

**Step 0, prerequisites**

- Database backed up to `~/.dance/dance.db.bak-pre-reanalyze-20261123-162954`
  (357 tracks, 29,116 regions, integrity check ok).
- Re-analysis: 244 of 244 non-duplicate tracks, 0 errors, 730 s. The 111 rows marked
  duplicate and the tracks in the error state were not re-run, so 113 of 357 mix
  analyses still carry the old values.
- Re-embedding: 715 s. All 1,750 embedding rows now carry the current scheme
  (`win-nonsilent-mean-v1`), where all 1,765 had none before, so recommendations
  and vibe search have their embedding signal back. `dance tag` tagged the 1 track
  that had no tags. `dance build-graph` built 10,938 neighbour edges.
- `als-clip-elements.md` step 1 (baseline) is done, read over OSC with nobody at the
  screen. Live 12.4.3 kept the warp grid written in the file on all five A-side
  clips, and the warp-marker read handlers from patch `0005` are active. Details in
  `docs/live-compat.md`. This settles the open question of whether a Set file can
  carry its own timing.
- Step 2 (Save As round-trip) is done. Live kept both warp markers and showed which
  element names and file reference form it writes.
- Option A is landed and checked in Live over OSC: the new export loads, clips carry
  their stem colour, quantization follows the Set, and the grid is kept after 22 s.
  Still open: a round-trip of the new export, and eyes and ears on it.

**Step 1, cues in data**

- Done in code: the picker, `pad` rows, `GET`, `PUT` and `DELETE` pad routes, six
  end-to-end tests. `dance export-als` leaves pads out of its locators.
- Done in the app: a "Pads" button on each queued pick in the Set view opens an
  editor with the song's waveform, five numbered flags coloured by section, a stem
  selector, click-to-move and reset. One flow test. Checked in a browser against
  the library copy: a move and a reset both went through and redrew.
- Run on a private copy of the library, 244 tracks: 243 got five pads and one track
  with two sections got two. Pad 3 is a drop on 168 of 244. Pad 1 starts more than
  5 s into the file on 47 of 244, because the beat detector starts where the beat
  does. Vocal pads start silent on 578 of 1,217 (47%), and on 42 of 244 tracks every
  vocal pad starts silent.
- Spread, same copy: the largest stretch of a song with no pad is a median of 39% of
  its length, and over half the song on 37 of 243 tracks. A variant that anchors on
  the first drop in place of the loudest scored the same (40%, 43 of 243), so the
  limit is the section labels, which call long stretches "breakdown". Moving pads by
  hand covers it for now. Better section labels are their own piece of work.
- Merged to `main` on 2026-11-23 (`97595cf`). The real library was backed up to
  `~/.dance/dance.db.bak-pre-pads-20261123-172103`, migrated to `a1c5d9e3f7b2`, and
  given its pads: 244 of 244 tracks, 0 errors, 97 s.
- Not done: listening to the start points on ten songs you know.

**Column order (decision 5)**

- Done in code on 2026-11-23: the ten deck tracks are grouped by deck in the bridge
  (`_DECK_KINDS`), the `.als` writer (`DECK_ORDER`) and the app
  (`TWO_DECK_COLUMN_ORDER`). One export test and one app test cover the order. The
  docs and the course describe the new fader map.
- Live 12.4.3 returned the ten track names in the new order from an exported Set.
- Not done: checking the fader and pad map on the APC40.

**Step 2, `dance export-set`**

- Done in code on 2026-11-23: the command and `POST /sets/{id}/als`, pages of five
  scenes, cue clips with their start and loop markers, the grid written into the
  file, warp mode per stem, section colours, single-stem picks on spare pages. Four
  end-to-end tests.
- Checked in Live 12.4.3 over OSC on a copy of the library (an original, its remix,
  a third song and one vocal stem): 15 scenes, clips on the 80 expected slots, loop
  lengths and start points as written, warp modes 0, 6 and 4. Details in
  `docs/live-compat.md`.
- Done in the app on 2026-11-23: an "Open in Live" button in the Set view. One click
  writes the plan to a new file and opens it in Live Standard. Checked on a copy of
  the library: Live switched to the new plan each time. Not checked: what Live does
  when the Set it has open has unsaved changes. A new file per click is there so
  that a save in that case cannot land on the file just written.
- Not done: the Live session in the verification plan. Nobody has looked at the
  screen, listened, or touched the APC40. The palette indices for pad colours are a
  guess. The live-load path still uses Beats for every stem.

**Step 3, the Booth reads pages**

- Done in code on 2026-11-23. The Booth grid shows one page of five rows with a page
  strip, and marks every page that has something playing. A pad carries a cue chip
  (number and section, dimmed where the stem is silent). Each deck has a row of cue
  buttons that start the loaded stems of its song from that cue. With nothing
  playing, a deck panel shows the song on the page in front of you.
- The backend adopts a Set when Live opens it. AbletonOSC sends `/live/startup` each
  time Live loads a Set, and the bridge listens for it, subscribes again to tempo,
  beat and transport, finds the deck tracks and reads the clips. No resync press
  after **Open in Live**.
- A cue comes from Live. The scan reads every clip's `start_marker`, and a cell is
  cue *n* only when that marker sits on pad *n* of the song and the cell is on row
  *n* of its page. A clip nudged in Live, a whole stem dropped on a row, or a pad
  moved in the app after the export has no cue. This is the rule "the cue is the row
  within the page" from the contract table, checked against what Live plays.
- Two songs with one title are told apart by the clip's audio file, read from Live.
  Before, the lower track id won.
- Cue buttons fire only the cells the Booth knows at that scene
  (`fire-deck?loaded_only=true`). On a page of spare stems a whole-deck fire would
  press the stop button of every empty slot on that deck.
- Three end-to-end tests against a UDP stand-in that answers from the exported file,
  and one app flow test.
- Checked with Live 12.4.3 on a copy of the library: one click on **Open in Live**
  and the Booth held 55 of 55 clips on 2 pages with their cues. Details in
  `docs/live-compat.md`.
- Not done: firing a cue from the app in Live, a Set larger than 55 clips, the
  APC40. `load-track` does not take a page yet, so a live load still puts one whole
  stem on the next free row, which shows without a cue unless pad 1 starts at 0:00.
  That is step 4.
- Left as it was: the deck ▶ button still fires all five tracks of the deck at the
  scene, empty slots included.

**Step 4, live loads build the same clips**

- Built on 2026-11-24 and checked against Live 12.4.3 over OSC the same day. Nobody
  has listened to it or used the APC40 on it. It is on the branch, not in `main`.
- A song loaded from the app during a set lands on a page as five cue pads per
  column, as a prepared Set would hold it: the same start and loop markers, warp
  mode per stem, section colours, and the song's own tempo grid. One difference
  is kept on purpose: a prepared Set's mix clips are deactivated in the file,
  and a live-loaded mix clip is left active as before, because the bridge mutes
  the two Mix tracks itself and `session-1.md` plays the mix by unmuting the
  track. `POST /ableton/load-track` takes the page the Booth shows and answers
  with the page, deck and first row the load landed on. Every load button in the
  app sends the page on screen, and the Booth follows the load to its page.
- Placement. A load takes the first page, starting at the one asked for, where its
  deck columns are empty. Empty is checked against Live as well as the app's own
  map, so a clip dropped in by hand counts. A load never deletes or replaces a
  clip. With no deck named, deck A is tried before deck B on each page. A single
  stem takes only its own column.
- The grid. Live's auto-warp cannot be trusted with a stem (measured below), so
  the load writes the grid itself. It makes the first pad of a column, waits for
  Live to answer for it, removes Live's warp markers, pins the start of the file
  and a point one second before its end to the song's tempo, activates the clip,
  copies it to the other four rows, and sets each row's markers. For 30 seconds
  after a load the bridge reads every new clip's grid back (once a second for
  12 s, then less often) and rewrites any that changed. In the Live runs below
  that check never had anything to do. `POST /ableton/warp-check/{scene}` does
  the same for the scene's page and reports only pads Live did not answer for or
  would not take the grid on. Its old length comparison now runs only on whole
  stems the app loaded without a grid.
- Loads take turns, because Live answers nothing while it makes the clips of the
  load before. When Live does not answer at all, nothing is loaded and nothing is
  added to the Booth. A column Live does not make is left out and named in the
  warnings. Pads are recorded only once Live has answered for their first clip.
- A song with no pads yet loads each stem whole on the first row of the page, on
  its own grid, with a warning. A song with no tempo analysis loads as before.
- `dance doctor` fails when the installed `add_warp_marker` handler is the form
  Live refuses, and counts the warp-marker handlers as needed by the app.
- Tests: seven end-to-end tests against a stand-in for Live, one app flow test,
  one `dance doctor` test through the command. The stand-in handles messages in
  the order sent and follows what Live did in the runs below: its warp-marker
  rules, the 4-beat loop a clip gets when looping is first switched on, the
  unlooped start that only moves through `loop_start`, a clip that is
  deactivated while Live analyses its file, and the time Live takes to make a
  clip.
- What Live 12.4.3 did (2026-11-24, on a copy of the library, read over OSC):
  - Left alone, Live warps a new clip three ways, none of them usable. A stem it
    has never analysed gets a placeholder grid at the Set's tempo and then Live's
    own grid within 9 s (2 of 2 stems; it read a 138 BPM drum stem at 105 BPM
    with beat 1 at 1.63 s). A stem it auto-warped on an earlier day gets that
    guess back as the clip is made (2 of 2 stems, each seen twice; one was half
    tempo, 64.17 for 128.0). A stem it first met inside an exported Set keeps
    the placeholder (2 of 2).
  - `add_warp_marker` refuses a dict ("No registered converter was able to
    produce a C++ rvalue of type NApiHelpers::TWarpMarker") and takes a
    `Live.Clip.WarpMarker`, which is what patch `0005` builds. The hand-written
    handler that was installed on this Mac passed a dict. Removing markers
    works; Live keeps one marker of its own after the last and will not remove
    it.
  - A grid written straight after a clip is made survives Live's analysis of a
    never-analysed file (11 files: one watched directly for 26 s, ten through
    the bridge's 30-second check, which found nothing to rewrite). A copied clip
    keeps the grid.
  - Switching `looping` on gives a clip that never looped a loop of beats 0 to 4,
    so `loop_end` has to be set before `loop_start`. An unlooped clip's start
    moves only through `loop_start`; a `start_marker` write is ignored. A clip's
    end stays where Live's first grid put it, which can be before a cue, so the
    end is raised first.
  - While Live analyses a new file the clip reads as deactivated. A copy made
    then stays deactivated after the analysis unless its source was active when
    it was copied. The first two loads of never-analysed files showed this on
    all 24 of their copied pads; with the first pad activated before copying,
    the next ten never-analysed files came out active (50 of 50 pads).
  - Making a clip from a file with a remembered auto-warp stops Live answering
    for about 0.6 s per file the first time in a session. A whole-song load of
    such a song took 3.4 s, a never-analysed song 0.5 to 1.3 s, and a song whose
    files Live already had open 0.7 s.
  - Nine whole-song loads and two single-stem loads, 235 clips, were read back
    from Live and compared with what was intended (name, file, loop and start
    markers, warp mode, colour, active, grid): all match. Two of the loads were
    made by clicking ⤒ in the Booth in a browser, and the Booth moved to the
    page they landed on. 24 pads of the first two loads were reactivated by hand
    because those loads predate the fix above. Live's script logged no error
    from any of the loads.
  - `/live/api/reload`, sent earlier that day to pick up the new handler, broke
    AbletonOSC in the running Live until Live was restarted: the reload cleared
    every handler and then stopped on the installed crossfader listener, which
    is not tracked the way the reload expects.
- Not done: listening to a live-loaded pad, the APC40, firing a cue from the app
  in Live, a load while something is playing (whether Live's pause while it
  makes clips is audible is unknown), and `move_warp_marker`.
- Found on the way: the AbletonOSC installed on this Mac is an older hand-edited
  copy. `scripts/abletonosc_patch.py --check` reports two patches in conflict and
  two pending, so the script cannot bring it in line with the repo. Only its
  warp-marker block was replaced, with the handler text of patch `0005` (backup
  in `~/.dance/backups/abletonosc-20261124-100603`).

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

### What a pad stands for

A pad could stand for three things. Options A, B and C below all sit in the first
two rows. The third row is out of scope.

| | What a press does | Closest to | In this proposal |
|---|---|---|---|
| **Part of a song** | starts one stem at a named place in its song, on the next bar, and the stem plays on from there | hot cues on a DJ deck | vocals and Song |
| **Loop taken from a song** | starts a 4, 8 or 16 bar loop of one stem, which repeats until another pad in its column replaces it | a loop pack built from your own library | drums, bass and melody |
| **Instrument** | sounds a hit or a chop at once, for as long as the pad is held, on top of whatever is playing | finger drumming | left out |

With option B and "hold per role", every pad is found by its place in the song: its
row is the intro, the drop and so on. The rhythm stems then loop where they land.
Finding pads by song part is what lets an original and its remix line up, because
"remix drums under the original's hook" needs both versions cut at the same named
places.

**Decided 2026-11-23: pads are found by song part.**

`vision.md` leans further towards loops in its opening line, "play stems as
independent loops, not as pieces of songs", and until this decision it counted a
set a success when a listener could not identify the source tracks. Playing an
original against its remix pulls the other way, because the listener is meant to
recognise the song. `vision.md` now has a section on originals and their remixes,
and that success line is replaced.

The loop reading would have changed two rules below. Each stem would have had its
own five best loops in place of its track's five sections, so rows would have
stopped lining up across a deck. The case for grouping columns by deck would also
have been weaker.

The instrument reading is left out for these reasons:

- The APC40 mk2 pads do not sense velocity (MusicRadar and MusicTech reviews).
- The stock script uses the grid for clip launch. I found no source for a note mode
  on this model. Akai's APC64 and APC mini mk2 have one.
- A hit played alone exposes separation artefacts that a full loop covers. This is
  an expectation and has not been tested.

A lighter form can be added later inside the clip model. Live clips have a Gate
launch mode, which sounds the clip while the pad is held, and a launch quantization
of their own. The installed AbletonOSC already writes `launch_mode`,
`launch_quantization` and `legato` (`abletonosc/clip.py`, `properties_rw`).

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

**Decided 2026-11-23: B.** The app picks the five cues and any of them can be moved
by hand.

### What B looks like

Original on deck A, its remix on deck B, with the column order from decision 5:

```
            DECK A (original)                 DECK B (remix)
            Drums  Bass   Vocals Melody       Drums  Bass   Vocals Melody
  row 1     intro  intro  intro  intro        intro  intro  intro  intro
  row 2     build  build  build  build        build  build  build  build
  row 3     DROP   DROP   DROP   DROP         DROP   DROP   DROP   DROP
  row 4     break  break  break  break        break  break  break  break
  row 5     drop2  drop2  drop2  drop2        drop2  drop2  drop2  drop2
  faders    1      2      3      4            5      6      7      8
```

Remix drums under the original's vocal is pad (row 3, Drums B) plus the Vocals A
pad for the part you want, with faders 1 and 5 crossing.

### Which five cues

Stored as `Region` rows with a new `region_type` of `pad`, one set per stem and one
for the mix. Picked in `detect_regions`, which already holds everything needed.

**Decided 2026-11-23: every stem and the mix share one list of five per track**, so
all rows line up: the intro, the first build-up, the loudest drop, the longest
breakdown after it, and the last drop (the outro when there is only one drop).

Vocals share the list too. The measured cost is that vocals are absent in 46% of
sections (1,493 of 3,256 on the 244 tracks), so close to half the vocal pads start
with no singing. Three things soften that:

- Vocals play on, so a vocal pad that starts in silence brings the singing in when
  the song reaches it.
- Each cue carries a flag for "this stem is silent for the first eight bars", which
  the app shows and the pad colour can show (decision 4).
- A vocal pad can be moved by hand on its own, without moving the other stems' pads.

The alternative, not chosen, was a separate vocal list of the five longest vocal
entrances.

These rules are a starting point to tune by ear. `source` is `auto`; a pad you move
by hand becomes `manual`, and `_clear_auto_data` already leaves manual rows alone on
a re-run (`detect_regions.py:144`).

Moving a pad is in scope. In the Set view a song's waveform shows its five cue
flags. Drag a flag to another bar line, or click a section to give it to a pad. A
reset puts the app's pick back. The API has no route that writes a region today, so
this adds one.

### What happens when the section ends

`vision.md` locks "stems loop by default" with an override for vocals. Three choices:

| | Drums, bass, melody | Vocals | Song (mix) |
|---|---|---|---|
| **Hold per role ← decided 2026-11-23** | loop the section, cut to 4, 8 or 16 bars | play on | play on |
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

- **Decided 2026-11-23:** hue for the section type (intro and outro, build-up, drop,
  breakdown), and a dim shade when the stem is silent for the pad's first eight bars.
- **Today, and not chosen:** one colour per role.

The app keeps role tints on screen. The exact palette indices need a look at the
hardware.

### Column order

Until 2026-11-23 the columns interleaved roles: Drums A, Drums B, Bass A, Bass B, and
so on. That put a role's two decks on neighbouring faders.

With B, the most common press is "start this deck at cue 3", which is four pads in
one row. Interleaved, those are pads 1, 3, 5 and 7. Grouped by deck they are four
neighbouring pads for one hand, faders 1 to 4 are deck A, and a role swap is the
same finger on each hand. The scene button fires both decks' cue, which is rarely
wanted, and the stock APC40 script cannot fire half a row.

- **Decided 2026-11-23:** group by deck. Deck A is faders 1 to 4, deck B is 5 to 8.
- **Changed in code 2026-11-23:** `_DECK_KINDS` (`bridge.py`), `DECK_ORDER`
  (`writer.py`) and `TWO_DECK_COLUMN_ORDER` (`roles.ts`) list drums, bass, vocals and
  other for deck A, then the same four for deck B, then the two mix tracks. The
  bridge finds deck tracks by name, so a Set that still has the old track order keeps
  loading, with the old fader map, until it is exported again or its deck tracks are
  created again. The new order has not been checked in Live yet.
- **Cost:** it reverses the 2026-06-13 choice. It is cheap now and expensive once
  your hands have learned an order. Three exercises in the course assumed the grouped
  order before the change (`docs/course/index.html` lines 849, 883, 902).

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

**B. `dance export-set` writes the whole plan as one `.als`. ← decided 2026-11-23.** Every
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
  page back. Decided 2026-11-23, because a single stem is queued to keep handy and
  the plan does not say which song it goes under. Putting it in a song's own column
  was the other choice, and it needs a plan that pairs a stem with a song.
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

- **Decided 2026-11-23:** Beats for drums and bass, Complex Pro for vocals, Complex
  for melody and the mix. The CPU cost with eight stems playing is not measured, so
  the first Live session checks it by ear and on the CPU meter.

### Loading during the set still works

A live load builds the same five clips per column over OSC, from the same cue and
grid functions. It needs the grid set through the warp-marker handlers
(`vendor/abletonosc` patch `0005`). Five `create_audio_clip` calls per stem would
each be auto-warped, so the load creates one clip, writes its grid, and copies it
with `/live/clip_slot/duplicate_clip_to`, which stock AbletonOSC already has.
(Corrected 2026-11-24. This paragraph used to say the patch was installed and that
the copy handler was missing. See the step 4 notes under Progress.)

## Part 3: Version families

### Options

**A. Compute the family when asked. ← default taken 2026-11-23.** A pure function beside
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
| SQLAlchemy models | One value added to `RegionType`: `pad`. One nullable column added, `regions.slot`, which says which of the five pads a row is. It came with hand-picking: a moved pad has to keep its number when the app's picks are rebuilt around it. Alembic revision `a1c5d9e3f7b2`, additive. |
| Stage protocol | None. `detect_regions` emits more rows. |
| `.als` emission | Several clips per deck track with their own start and loop markers, warp mode per role, named scenes, more than one song per Set. A new export beside the per-track one, which stays. |
| OSC load contract | A load creates one clip per column, writes its warp grid (`remove_warp_marker`, `add_warp_marker`), copies it to the other four rows (`duplicate_clip_to`), and sets `start_marker`, `loop_start`, `loop_end`, `looping`, `color_index` on each. Clip names stay `{title} ({kind} {side})` so `/decks/resync` keeps working (`ableton.py:701`); the cue is the row within the page. |
| API schemas | Deck cells gain a cue number and label. New `POST /sets/{id}/als` and `GET /tracks/{id}/versions`. New `GET /tracks/{id}/pads`, and `PUT` and `DELETE /tracks/{id}/pads/{stem}/{n}` to move a pad and to reset it. `{stem}` is `drums`, `bass`, `vocals`, `other`, `mix`, or `all` for the whole song. The silent flag is worked out on read from the stem's presence data. `load-track` accepts a page. |
| Plan shape | Unchanged. |
| Column order (decision 5, decided) | `_DECK_KINDS`, `DECK_ORDER`, `TWO_DECK_COLUMN_ORDER`, the template's track order, docs and course. |

## Order of work

Each step lands on its own and is checked in Live before the next one starts.

0. **Prerequisites, already written up.** Back up the database. Run
   `dance reprocess --from analyze --all --skip-embeddings`, `dance embed --all`,
   `dance tag`, `dance build-graph` (take the provenance proposal's duration snapshot
   first if that proposal is going ahead). Run `als-clip-elements.md` verification
   steps 1 and 2, then land its option A.
1. **Cues in data.** The picker, the `pad` rows, the API field, five cue flags on
   the waveform, and moving a flag by hand. Nothing changes in Live. Check: read the
   cues for ten songs you know, listen to each start point, and move one.
2. **`dance export-set`.** Pages, cue clips, the shared grid, warp modes, and the
   column order if chosen. Check: the Live session in the verification plan.
3. **The Booth reads pages.** `SceneGrid` shows the current page with cue labels,
   resync adopts a prepared Set, deck play takes a cue.
4. **Live loads build the same clips.** Only after patch `0005` is seen working.
5. **Families, read side.**
6. **"Find versions".** Repair the Spotify module first, then the search and the
   queue action. Decided 2026-11-23 to build it in this round, because the library
   holds three original and remix pairs and the style needs more.

## Decisions for you

1. **What a pad is.** Decided 2026-11-23: B, found by song part, with `vision.md`
   edited to match. The app's five cues are the default and any pad can be moved by
   hand.
2. **Section end.** Decided 2026-11-23: hold per role. Drums, bass and melody loop
   their section, vocals and Song play on. It stays a setting.
3. **The five-cue rule.** Decided 2026-11-23: one list of five per track, shared by
   every stem including vocals.
4. **Pad colour.** Decided 2026-11-23: section type, with a dim shade for silent
   stems.
5. **Column order.** Decided 2026-11-23: grouped by deck, and changed in code the
   same day.
6. **Prepared Set.** Decided 2026-11-23: B, one exported `.als` for the whole plan.
7. **Single-stem picks in the Set.** Decided 2026-11-23: spare pages after the
   songs.
8. **Warp mode per role.** Decided 2026-11-23: per role, judged by ear at a 6% tempo
   change.
9. **Families.** A, computed. Taken as the default on 2026-11-23 without a separate
   question, because nothing you see or hear depends on it. Easy to reverse.
10. **"Find versions".** Decided 2026-11-23: build it in this round, as the last
    step.
11. **Scope of the go-ahead.** Decided 2026-11-23: go ahead, covering the library
    re-run, `als-clip-elements.md` option A, and the Live check.

## Migration cost

- **Backend:** the picker in `detect_regions.py` (about 120 lines); a multi-clip,
  multi-song path in `writer.py` and `generator.py` (about 200, on top of the
  conformance patch); `push_track_to_live` creating five clips per column (about 80);
  the family function and endpoint (about 90); CLI and API wiring (about 60). These
  are estimates.
- **Frontend:** `SceneGrid` paged with cue labels, cue flags in `TwoDeckStrip`, a
  version chip and grouping in `RoleColumn` and `CommandBar`. About 300 lines.
- **Data:** one additive migration (`alembic upgrade head`, for `regions.slot`).
  Pads appear when regions are regenerated, so after the step 0 re-run the library
  needs `dance reprocess --from detect_regions --all --skip-embeddings`, which took
  about half a second a track on 2026-11-23. Existing `.als` exports stay as they are.
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
  read back through `GET /tracks/{id}/pads`. Every stem has the same five
  positions, and the vocal cue at the intro carries the silent flag.
- `PUT /tracks/{id}/pads/{stem}/{n}` on a pipeline-built track, then the pipeline
  run again from `detect_regions`. The moved pad is still where it was put, and
  `DELETE` brings the app's pick back.
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
- Renaming cues in the app.
- Hold and release, loop length, or loading from the controller. Each needs a MIDI
  listener beside the OSC bridge.
- Pads played as an instrument, meaning hits and chops that sound while held. See
  "What a pad stands for".
- A grid check with a click and nudge keys, and reading warp fixes back from a Set
  you saved in Live (`als/reader.py` could do the second).
- A filter or EQ on each deck track (`hardware-and-live-controls.md` section 2).
- The stem model, loudness matching, and identity fields. Those have their own
  proposals.
- Flagging two rows that hold the same audio under different titles, such as ids 79
  and 176.

The go-ahead was given on 2026-11-23.
