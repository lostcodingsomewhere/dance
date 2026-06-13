# Learning journal

Where you (arya) track the DJing-with-stems learning curve, and where Claude looks to understand "where am I?" between sessions.

Update freely — keep entries dated, keep them short, keep the *what surprised me* honest. Future you (and future Claude) will thank present you.

---

## How to actually get there (the approach, not the tooling)

The build keeps growing; the playing doesn't. Four corrections, in priority order:

1. **Pick a real gig date 4–6 weeks out — now — and make it the forcing function.** Invert the self-graded "5 consecutive recorded sets you'd play for a stranger" trigger (Phase 5 below): that bar is infinitely deferrable because *you* grade it. A booked date you can't move grades it for you. House party, a friend's birthday, a bar's open-decks night — anything with a date and other humans. Work backward from it.
2. **Tools freeze: no new app features until first play-out.** The companion app and pipeline are good enough to DJ on. Every new feature is procrastination dressed as progress. Until you've played out once, the only code changes allowed are *bug fixes that block playing* — nothing additive. After the gig, reassess.
3. **Record-and-listen is a per-session ritual, not a Phase-4 gate.** Don't wait until "Phase 4: 15-min recorded set" to start recording. Record *every* session from today and listen back to at least part of it. In the bedroom phase you have no crowd, so the recording is your only honest feedback signal. Make it a habit before it's a milestone.
4. **Check Phase 0 off first.** Before anything clever: open a generated `.als`, hit play, and confirm sound comes out of the Edifiers and the Bose cues independently. "Hear sound out of the speakers" is the gate everything else stands on — don't assume it; verify it at the start of the next session and tick the box.

---

## The runbook (start here every session)

### First-time setup (one-off)

```bash
# Hardware (current rig — Scarlett 4i4 4th gen, 4 outs):
#   1. Plug Scarlett 4i4 into Mac (USB-C). Mac picks it as default audio out.
#   2. Plug APC40 mk2 into Mac (USB-C). Ableton sees it as a Control Surface.
#   3. Speakers: Scarlett outs 1/2 (TRS, back) → Hosa CPR-203 → Edifier RCA.
#   4. Cue: Scarlett outs 3/4 (or the front headphone jack assigned to mirror
#      3/4 via Focusrite Control) → GPM-103 adapter → Bose.
#      Front Phones knob ~9 o'clock to start. Direct Monitor disabled.
#
# Ableton:
#   Preferences → Audio → Driver: CoreAudio,
#     Audio Input: Scarlett 4i4 USB, Audio Output: Scarlett 4i4 USB.
#   Preferences → Audio → Output Config: enable both 1/2 and 3/4.
#   Mixer: set Master → Audio To: 1/2. Set Cue Out: 3/4. Click Solo button →
#     "Cue" mode (toggles between Solo-in-Place and PFL/Cue). With Cue mode
#     on, a soloed track routes to 3/4 (headphones) instead of muting the
#     rest of the mix.
#   Preferences → Link, Tempo & MIDI → Control Surface row: Akai APC40 mkII
#     (Input/Output: APC40 mkII).
#
# Dance:
source .venv/bin/activate
dance config --show               # confirms playlist URL, dirs, db
```

> **Independent cue is live:** with the 4i4, soloing a deck or clip in Live
> routes audio to the Bose only — the Edifiers keep playing the master
> untouched. This is the foundation for "preview a candidate before
> committing to the master" workflows in the companion app.

### Every-session runbook ("I want to DJ now")

```bash
# 1. Confirm pipeline state
dance status

# 2. If new tracks were added to the playlist:
dance run --once                  # sync + process (auto-advances all stages)

# 3. After process finishes:
dance tag                         # CLAP zero-shot tags (~50 ms/track)
dance build-graph                 # rebuilds track_edges (recommendations)
dance export-als --all            # one .als per track in ~/Music/Dance/Sets/

# 4. Spin up the companion app (two terminals)
uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000
cd companion-app && npm run dev   # http://localhost:5173

# 5. Open a Set — ONLY to load the deck columns into Live
open "/Users/arya/Music/Dance/Sets/<some track>.als"
# → Live launches with the 10-column 2-deck layout (Drums/Bass/Vocals/Other/Mix
#   × A/B) + master tempo set. The app adopts these columns automatically.
# → From here, DON'T mix from the .als's own clips. Live-load stems through the
#   app instead: ⌘K to search, or promote a rec — each load drops the stems onto
#   a deck column over OSC. That path is what links cells to the library and
#   drives per-combo rec re-scoring. (Static .als clips are columns-only by
#   design — see docs/proposals/als-deck-name-prefix-mismatch.md.)
# → Fire stems from the app's grid / the APC40.
```

### Shutting down

```bash
# Ctrl-C the uvicorn and `npm run dev` terminals. Nothing else holds state.
# Live's "Save Live Set As..." → keep your performance .als somewhere outside ~/Music/Dance/Sets/
# (Anything inside ~/Music/Dance/Sets/ may be overwritten by the next export-als --all.)
```

---

## Where I am in the learning curve

Edit this table as you go. Targets are soft.

| Phase | What I'm learning | Status | Notes |
|---|---|---|---|
| 0. Rig works at all | Sound out of Edifiers, APC40 lights up, Bose cues independently | ⬜ today | First-light test: open generated `.als`, hit play, hear it. |
| 1. APC40 → Session View basics | Launching clips, scene rows, stop buttons, faders→volume, knobs→EQ | ⬜ | Live's manual: APC40 mkII Reference (built-in PDF in Live). |
| 2. Stem-mix a single track | Drop drums in, fade vocals out, EQ-sweep the bass | ⬜ | Use one of our exported `.als` files — 5 stem tracks already there. |
| 3. Two-track transition | Track A → Track B using a recommended pair from Up Next | ⬜ | Companion app `/up-next` view drives the candidate list. |
| 4. 15-min set, recorded | Live → Record button → bounce → listen back | ⬜ | Listening back is where mistakes become obvious. |
| 5. First gig | House party, friend's basement, anywhere | ⬜ | Trigger: 5 consecutive recorded sets you'd play for a stranger. |

---

## Open questions (let Claude answer next session)

- Does CLAP-based "Up Next" actually feel right, or does it surface harmonically-wrong picks? → If wrong, tune `recommender/graph_builder.py` weights, or add Camelot-wheel hard constraint.
- Should the APC40's left-hand fader bank map to stem volumes (current plan) or to send levels for FX (alternative)? → Decide after 4 sessions.
- Do exported `.als` files need device chains pre-loaded (EQ8, Glue Compressor) or is the click-and-drag template approach fine? → Try both, pick the friction-lower one.
- What's the right way to "promote a recorded set into a permanent Live project"? → Write a script that copies the `.als`, freezes clips, flattens to arrangement?

---

## Session log

Append to the top. Use this template:

```
### YYYY-MM-DD — short title
**Played:** N tracks / M minutes / "first time trying X"
**Worked:** what felt good
**Broke:** what surprised me / what didn't work
**Next time:** one thing to try
```

### 2026-06-13 — Docs/course fix pass (drift cleanup, no DJing)

**Played:** No DJing — documentation + course correction pass while a real pipeline run was in flight elsewhere.
**Worked:**
- Course (`docs/course/index.html` + the companion-app mirror, kept byte-identical): fixed the library path (`~/Music/Dance/library` → `~/Music/DJ/library`); rewrote Move 1 (The Fade) to the 2-deck fader map (1–4 = Deck A, 5–8 = Deck B, Mix off-fader) and the hardware A/B crossfader; corrected the 4th rec signal (Energy match → Tag overlap) and added that live banners re-score on the active stem combo (embedding + key + BPM); added a Cue-is-additive heads-up; added the one-lead-vocal rule and an "anchor is home" strategy framing; promoted record-and-listen-back into a per-session habit from Lesson 1.1; added a weekly "recs-off rep" exercise; added an "ingest stalls" note (ffmpeg, ~580 MB first-run weights, SIGBUS on Py3.14).
- Repo docs: updated `dj_ux_flow.md` + `vision.md` to the 8-column 2-deck grid (Mix moved to per-deck header chips; crossfader now used) and the actual rendered Booth surfaces (TwoDeckStrip/Crossfader/BoothColumnHeaders, no ComboStrip/MasterVisualizer/PlayedStrip); rewrote `als-export.md` to the 10-track 2-deck shape with the "clips are scaffold decoys, live-load through the app" framing up front; fixed `troubleshooting.md` (real CSV path `scripts/yt_dlp_csv_import.py` + new `dance ingest` wrapper; added a Demucs SIGBUS/exit-138 entry); flagged `proposals/hardware-fx-runbook.md` SUPERSEDED (wrong FX return-track names); updated `CLAUDE.md` Booth surfaces.
**Broke:** Nothing run live — docs only. The course had drifted badly from the shipped 2-deck layout; the single-deck "5 columns on faders 1–5" map was the most misleading bit.
**Next time:** Stop editing docs and DJ. Pick the gig date (see "How to actually get there"), check Phase 0 off, record the session.

### 2026-05-22 — Booth UX hardening + visualizers + deck-state persistence

**Played:** No DJing yet — UX iteration pass.
**Worked:**
- SceneGrid at top of Booth (was buried below banners), tap-to-stop on cells + row labels (was fire-only), bigger emerald playing-state visual + 🔁 loop badge.
- VU meter in MasterStrip (sum of deck-column meters via AbletonOSC subscription — a separate output_meter_level per deck column).
- Master Stacked Stems visualizer below the grid: one row per stem role (drums/bass/vocals/melody), each rendering the *actual stem audio* waveform with a wrapping playhead derived from Live's master beat clock.
- CueStrip surfaces what's auditioning in headphones (parallel to ComboStrip's master surface), with `→ Load … to master` one-click commit.
- BPM editor became a slider popover with genre-anchored ticks (chill / hip-hop / house / techno / trance / d&b), explicit Apply button so glancing drags can't change tempo mid-set.
- Cell-level loads: rec card buttons are now column-specific (`Load drums` / `Load bass` / etc.) and only load that stem; a separate `Load song` button on stem cards loads all 4 stems anchor-style.
- "OSC" → "Live" relabel + heartbeat green-dot tooltip; dropped the confusing "X decks loaded · out-of-sync" chip.
**Broke:**
- Bridge restart was wiping `_deck_cells` even though Live still had the clips loaded → SceneGrid showed empty. Fixed two ways: (a) bridge persists state to `{settings.data_dir}/deck_state.json` atomically on every mutation + restores on `start()`, and (b) new `POST /api/v1/ableton/decks/resync` scans Live's deck-column clip slots, parses clip names back to `Track` ids via DB lookup, and adopts them into bridge state. A small `↻ resync` button in MasterStrip triggers the latter.
- `playing_slot_index` wasn't subscribed per-deck-column → `AbletonState.playing_clips` stayed empty so the FE never saw firing cells (VU meter worked but visuals showed "silent"). Fixed by subscribing in `_subscribe_deck_columns` alongside the meter subscription.
**Next time:**
1. Wire up the Scarlett 4i4 physically when it arrives (outs 1/2 master, 3/4 cue) and confirm preview audio leaves on 3/4 only.
2. Try a 30-min real set — see what breaks and what needs rethinking.
3. Set/crate refactor — ✅ shipped 2026-05-23 as the Set Rail consolidation. Sets are DB-persisted and named, the Crate view is retired (Pipeline is the inventory surface, Cmd-K is the search), and the active set drives tail-rec scoring against its trailing arc. See [`docs/proposals/set-rail-and-search-consolidation.md`](docs/proposals/set-rail-and-search-consolidation.md).

### 2026-05-18 — Scarlett 2i2 → 4i4 swap (cue routing unlocked)

**Played:** No DJing yet — hardware change session.
**Worked:** Pulled the Scarlett 4i4 upgrade forward from "9-12 months out" because the cue/preview workflow in the companion app makes no sense without an independent stereo cue out. Sweetwater 45-day window made it cheap to swap inside the original buy.
**Why now:** The proposed "preview before committing" UI (small ▶ on every rec / anchor card) requires the candidate to be auditioned in headphones without leaking to the master. A 2-output interface physically can't do that.
**Broke:** Nothing yet — Kyle's RMA paperwork is the next dependency.
**Next time:**
1. Wire 4i4 per the updated runbook: outs 1/2 → Edifier, outs 3/4 → Bose via GPM-103.
2. In Ableton Preferences → Audio → Output Config, enable both pairs; set Master → 1/2, Cue Out → 3/4. Toggle the mixer's solo button into Cue (PFL) mode.
3. Solo a clip → confirm it plays in headphones only.
4. Then green-light the FE cue/preview build (decoupled from this hardware day; lands as its own PR).

### 2026-05-16 — Bootstrap day (audio source meltdown + recovery)

**Played:** Library bootstrap, no DJing yet.
**Worked:**
- All hardware docs + project docs in place (HARDWARE / ORDER / CLAUDE / LEARNING).
- 66 of your own tracks symlinked into `~/Music/DJ/library/` (from `~/Music/Music/My Music/` + Beatport purchases). Pipeline ingests these natively.
- Pipeline analyze stage running ~5 s/track on M2 Pro — projecting ~6 min for analyze, ~30 min for Demucs separation, total ~45 min to a fully-loaded library.
- Hardware swap: messaged Kyle to upgrade Scarlett 2i2 → 4i4 (for independent cue).

**Broke:** Spotify ingest. The path was supposed to be simple — `dance config --spotify-playlist <url>` then `dance sync`. It is not simple anymore.

Three escalating blocks, **3 hours of fighting**:

1. **spotDL default token hang.** The shared `client_id=5f573c9620494bae87890c0f08a60293` is rate-limited to oblivion. Created our own Spotify dev app on the dashboard → fixed.
2. **Spotify Web API endpoint deprecation (2024-11-27).** New apps cannot access `/v1/playlists/{id}/tracks` regardless of auth (client-credentials OR user-auth with `playlist-read-private` scope). Returns HTTP 403, no workaround within spotDL. Pivoted to **Exportify** (https://exportify.net) for CSV export — works because Exportify's old dev app is grandfathered.
3. **YouTube anti-bot wall.** With the CSV in hand, `yt-dlp` was supposed to fetch each track from YouTube Music. YouTube's 2025-2026 escalation now requires: latest yt-dlp + Node.js + `yt-dlp-ejs` + `bgutil-ytdlp-pot-provider` (Docker server on :4416) + browser cookies. Even with **all of that**, we got "Only storyboard images available" — the n-challenge signature solver is intermittently broken.

**Pivot that actually worked:** Symlinked your existing 66 tracks. You already owned what you needed. (Lesson: check what you have *before* fighting third-party APIs for 3 hours.)

**Permanent docs added to repo:**
- [`docs/troubleshooting.md`](docs/troubleshooting.md) "spotdl rate limiting / Spotify Web API restrictions (Nov 2024+)" + "Loading audio from a local folder (no Spotify needed)"
- bgutil + EJS plugins kept installed for any future yt-dlp retry — Docker container `bgutil-provider` runs on `localhost:4416`. Restart with `docker start bgutil-provider`.

**Next time (in order):**
1. Open one of the exported `.als` files in Live, hit play, confirm sound out of Bose headphones.
2. Set up APC40 → Session View mapping (Live's APC40 mkII Reference PDF, built into Live's help menu).
3. Stem-mix a single track — fade vocals in/out, EQ-sweep the bass, listen.
4. **Don't open YouTube/Spotify fight again.** When you want more tracks: buy 5-10 you actually intend to mix on **Beatport** (320 kbps MP3 or AIFF, ~$2-3/track, instant). Drop in `~/Music/DJ/library/`, run `dance process`. Done.

---

## Reference (don't rewrite, just link)

| Thing | Where |
|---|---|
| Why this hardware | [`HARDWARE.md`](HARDWARE.md) |
| Receipt / warranty windows | [`ORDER.md`](ORDER.md) |
| What a generated `.als` contains | [`docs/als-export.md`](docs/als-export.md) |
| Every `dance` CLI command | [`docs/cli.md`](docs/cli.md) |
| API + WebSocket reference | [`docs/api.md`](docs/api.md) |
| AbletonOSC install | [`docs/abletonosc_setup.md`](docs/abletonosc_setup.md) |
| When things break | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
