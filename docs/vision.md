# Vision

What this app is for, what creative ambition it serves, and how that shapes every design decision downstream. Read this before `dj_ux_flow.md` and the proposals — those documents elaborate on what this one declares.

This is a **personal tool for one user**. It is not a general-purpose DJ app and is not designed for someone else's flow. Optimize for *this* style.

## Style: live remixing

The defining creative ambition: **play stems as independent loops, not as pieces of songs.**

A set is a continuously evolving composition where the drums come from track A, the bass from B, the vocals from C, the synth from D — and any of them can be swapped, layered, or stripped out at any moment without stopping the rest. Closer to live electronic performance (ODESZA, Daedelus, RJD2's live set, modern producer-DJs) than to traditional song-to-song DJing.

**The "song" stops being the unit of play. The stem instance is.**

Three modes co-exist in this paradigm:

- **Anchor** — fire a whole row to play the original song combo. Safety net, recovery, "this song is just hitting".
- **Swap** — keep most stems running, swap one out for a stem from a different song. The most common move.
- **Build** — start with one stem (just drums), layer in bass, then vocals, then a synth. Build a track from scratch, live.

## Choose-your-adventure recommendations

The app is **not a search box**. It is a **continuously re-scoring set of suggestions** that respond to what's currently playing.

At every moment, **each column** of the 8×5 grid (drums, bass, vocals, other, mix) has its own recommendation stream. Each stream answers a different question:

| Column | Question its recs answer |
|---|---|
| Drums | "Given the current bass + vocals + other, what drums would land?" |
| Bass | "Given the current drums + vocals + other, what bass would land?" |
| Vocals | "Given the current drums + bass + other, what vocals would land?" |
| Other | "What melodic / synth / pad element fits the current combo?" |
| Mix | "What full song would I anchor to right now?" |

As you swap one stem, every other column's recs re-score around the new combination. Decisions cascade. Pull a new vocal in → the drums/bass/other columns update because the context changed. You move through a tree of "what's possible right now", and the system shows you the live edges.

Filters and free-text vibe search live per-column. "More aggressive drums" filters just the drums stream. ⌘K vibe search auto-scopes to whichever column has focus.

## What this app is NOT

Naming the anti-vision sharpens the vision:

- **Not a library browser.** Spotify already exists. Pre-set curation is part of the flow but it's not the focus.
- **Not a song-to-song DJ deck.** Rekordbox, Traktor, Serato exist. This repo explicitly removed Traktor. If you want crossfaders and song decks, use those.
- **Not a producer's DAW.** Ableton exists. We don't compete with the engine; we sit on top of it.
- **Not multi-user.** No collab, no cloud sync, no streaming.
- **Not mobile-first.** Laptop screen during dev; potentially iPad as secondary surface someday, but not the design target.
- **Not a "DJ AI"** that decides for you. It surfaces possibilities; the user decides. Choose-your-own-adventure, not autopilot.

If you wanted to play in song-mode, you would not need this app.

## What success looks like

A 30-minute set where:

- The user did not look at Ableton's UI once.
- The set evolved in ways that **surprised the user** — combinations they would not have curated in advance.
- The set arc has a coherent shape (warmup → peak → resolve) without being pre-planned.
- The user felt like they were **playing an instrument**, not selecting from a playlist.
- A passive listener could not identify the source tracks.

## How this shapes design

These decisions fall out of the vision and are locked:

1. **Frontend is the primary surface.** Ableton is the engine; the APC40 is the hands; the React app is the eyes and the brain. Live's UI is hidden during a set.

2. **Columns, not rows, are the primary visual unit** of the 8×5 grid. Each column is a stem role (drums/bass/vocals/other/mix). Rows exist as a loading convention and as a "fire whole song" anchor.

3. **Stems loop by default.** A drum stem keeps grooving until you swap it. Per-clip override available for stems that should play through (e.g. a vocal verse → chorus → verse arc).

4. **Recommendations are per-column and real-time.** Not a single ranked list. Four-to-five parallel streams that re-score on every active-combo change.

5. **Curation lives in the recs, not in tagging.** No "this track is drums-only" metadata. The system surfaces what works given current context; the user doesn't have to pre-classify.

6. **The 8×5 grid mirrors the APC40.** Same shape, same orientation (bottom row = scene 1), same row-is-scene semantics. Muscle memory wins.

7. **Compat is per-stem, not per-track.** Key/BPM/energy compatibility is computed against the active combo of stems, not against a "now playing song". When you only have drums playing, only the drums are part of the compat math for other columns.

## Provenance

This vision was articulated through conversation on 2026-05-17 after the user pushed past the original "stems are first-class" framing into a more specific creative ambition. Earlier iterations of the companion app assumed song-mode performance; the live-remixing pivot is a directional refinement, not a reset — the pipeline, the DB schema, and the .als writer all still apply. What changes is the **UI's primary unit of attention** (cell, not row) and the **shape of the recommendation system** (per-column streams, not a single list).
