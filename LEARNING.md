# Learning journal

Where you (arya) track the DJing-with-stems learning curve, and where Claude looks to understand "where am I?" between sessions.

Update freely — keep entries dated, keep them short, keep the *what surprised me* honest. Future you (and future Claude) will thank present you.

---

## The runbook (start here every session)

### First-time setup (one-off)

```bash
# Hardware (current — headphones-only, no speakers yet):
#   1. Plug Scarlett 2i2 into Mac (USB-C). Mac picks it as default audio out.
#   2. Plug APC40 mk2 into Mac (USB-C). Ableton sees it as a Control Surface.
#   3. Bose plugged into Scarlett 1/4" front jack via GPM-103 adapter.
#      Scarlett "Phones" knob ~9 o'clock to start; "Monitor" knob can stay at 0
#      (no speakers in the chain).
#   4. Edifier speakers will join the chain once they're at the desk —
#      Scarlett TRS outs (back) → Hosa CPR-203 → Edifier RCA inputs.
#      Until then, headphones are the only output.
#
# Ableton:
#   Preferences → Audio → Driver: CoreAudio, Audio Input: Scarlett 2i2 USB, Audio Output: Scarlett 2i2 USB
#   Preferences → Link, Tempo & MIDI → Control Surface row: Akai APC40 mkII (Input/Output: APC40 mkII)
#
# Dance:
source .venv/bin/activate
dance config --show               # confirms playlist URL, dirs, db
```

> **Headphones-only mode (current):** Cueing (the "play this track silently in
> headphones while the other is on the main outs") doesn't work without
> speakers — the headphone jack mirrors the main outs on a 2i2. That's fine
> for learning APC40 muscle memory and stem-mixing a single track. Real
> A/B cueing arrives when the Edifiers join the chain.

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

# 5. Open a Set
open "/Users/arya/Music/Dance/Sets/<some track>.als"
# → Live launches with 5 stem tracks + master tempo set
# → Hit space, play scene 1, start mixing on APC40
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
