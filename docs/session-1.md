# Session 1 — make sound, then mix two tracks

One sitting, about 45 minutes. No new features, no planning, no recs. The goal
is to leave this session having **heard yourself mix**, and having ticked
Phase 0 and Phase 1 in [`../LEARNING.md`](../LEARNING.md).

Read the whole thing once before you start. Then work down it with your hands.

> **The one thing to know before you start.** Live auto-warps every stem it's
> handed, independently, and it gets isolated stems wrong a lot — measured on
> your own library, 4 of the first 4 tracks had at least one stem on the wrong
> tempo (one bass read 113 BPM as 73). It doesn't error; the stem just slides
> out of time. **If something sounds out of time in this session, suspect the
> tool before you suspect yourself.** That's why Part 3 works on the mix cells,
> where Live is reliable, and stems only come in at Part 4 with the audit on.
> Details: [`proposals/warp-guard.md`](proposals/warp-guard.md).

---

## Part 0 — Bring the rig up (5 min)

You said the gear is wired and known good, so this is just confirmation.

- [ ] Scarlett 4i4 on, Edifiers on outs 1/2, Bose on outs 3/4, APC40 plugged in.
- [ ] Open Ableton. Preferences → Audio → In/Out both = **Scarlett 4i4 USB**.
- [ ] Preferences → Audio → Output Config → **both 1/2 and 3/4 enabled**.
- [ ] Mixer: Master → Audio To **1/2**. Cue Out **3/4**. Click the Solo button
      so it reads **Cue** (PFL), not Solo-in-Place.
- [ ] Preferences → Link/Tempo/MIDI → Control Surface = **Akai APC40 mkII**.

Two terminals:

```bash
cd ~/git/dance && source .venv/bin/activate && uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000
```

```bash
cd ~/git/dance/companion-app && npm run dev
```

- [ ] `http://localhost:5173` loads and the top strip shows a green **Live** dot.

---

## Part 1 — Phase 0: hear something (5 min)

Do not skip this because it seems trivial. Everything else stands on it, and
it's been unticked since May.

- [ ] Open any exported set:
      `open "$HOME/Music/Dance/Sets/Alive (feat. The Moth _ The Flame) - Kx5_ deadmau5_ Kaskade_ The Moth _ The Flame.als"`
- [ ] Live opens with 10 deck columns + master tempo set. Hit **play** on any clip.
- [ ] **Sound comes out of the Edifiers.** ✅ Tick Phase 0 in `LEARNING.md`.
- [ ] Solo that track. Sound moves to the **Bose only**, Edifiers keep playing
      the master. That's your cue bus working. ✅

If either fails, stop and fix it — nothing below works without this.

---

## Part 2 — Load two decks (10 min)

Back in the companion app, Booth view.

- [ ] ⌘K → type an artist you know → **Enter** on a result. That loads all 4
      stems + the mix onto a free scene.
- [ ] Do it again with a second track. Two scenes now have clips.
- [ ] In the **Song** column of the plan grid, use **⤒A** and **⤒B** instead if
      you want to choose the deck explicitly. A on the left, B on the right —
      same as the crossfader.

**Pick two tracks in the 120–129 range.** 276 of your 353 tracks live there, so
this costs you nothing and means the two decks are already near-beatmatched. If
you want them to sound good together as well, pick two in adjacent Camelot keys
— you have 141 tracks in 5A–8A alone.

- [ ] Wait ~20 seconds after each load. If an amber banner appears under the top
      strip naming a stem ("bass_a: warped to 646 beats but the other stems came
      out at 710"), that's the warp audit. Note it; you'll want it in Part 4.

---

## Part 3 — Your first mix, on the mix cells (15 min)

This is ordinary two-deck DJing. No stems yet. Live warps a full mix reliably,
so this part is unaffected by the warp problem.

In **Live's** session view, for each of your two scenes:

- [ ] Mute the 4 stem tracks (Drums/Bass/Vocals/Other) on that deck side.
- [ ] Unmute the **Mix** track for that side.

You now have a normal two-deck setup: Deck A's mix, Deck B's mix, crossfader.

- [ ] Fire Deck A's mix clip. Let it play.
- [ ] Crossfader hard left. Bring the master fader up. Listen for 30 seconds.
- [ ] PFL Deck B (the **PFL** button on its panel, or **S** on its column) and
      find the spot you want to come in on — in your headphones, master
      untouched.
- [ ] Fire Deck B's mix. Clips are quantized to 1 bar, so it lands on the
      downbeat for you.
- [ ] Move the crossfader across over ~16 bars. That's the mix.
- [ ] Do it three more times. Same two tracks. It gets better fast.

**That's a DJ set.** Everything above this line you could have done in
Rekordbox. Everything below is why you built this.

---

## Part 4 — The same transition, with stems (10 min)

Now unmute the 4 stem tracks and mute the Mix tracks — the reverse of Part 3.

- [ ] Check the warp banner first. If a stem was flagged, fix it now: click that
      clip in Live, and in Clip view hit **`*2`** or **`:2`** as the banner says.
      If it's flagged with neither (a generic "will drift"), just don't use that
      stem this session — mute it and carry on with three.
- [ ] Fire Deck A (▶ on its panel). All four stems play. Sounds like the song.
- [ ] **Ride the faders.** APC40 faders 1–4 are Deck A's stems. Pull the vocal
      out. Bring it back. Pull the bass. This is the whole thing — you're
      playing the song's parts, live.
- [ ] Now the transition again: instead of crossfading whole tracks, bring in
      **Deck B's drums only** under Deck A, then B's bass, then pull A's drums.
      That's Move 2, the bass swap.
- [ ] One rule: **never two basslines at once.** The app will nudge you.

---

## Part 5 — Record and close (5 min)

- [ ] Live's transport **Record** button → play 5 more minutes → stop.
- [ ] Listen back to at least 2 minutes tomorrow. This is the only honest
      feedback you have in a bedroom.
- [ ] Add a dated entry to the Session log in `LEARNING.md`. Use the template.
      Fill in **Broke:** honestly — that entry is what makes next session sharper.
- [ ] Tick Phase 0, Phase 1, and Phase 2 if Part 4 went well.

Save your performance set somewhere **outside** `~/Music/Dance/Sets/` — anything
in there gets overwritten by the next `export-als --all`.

---

## If it goes wrong

| Symptom | Look at |
|---|---|
| A stem is out of time | The warp banner. [`proposals/warp-guard.md`](proposals/warp-guard.md). Not your fault. |
| Grid says "Waiting for Ableton deck columns" | Backend running? Green Live dot? Try **↻ resync** in the top strip. |
| Cue leaks into the speakers | Solo button isn't in **Cue** mode. Part 0. |
| Live won't open the `.als` | [`troubleshooting.md`](troubleshooting.md) → "Live rejects the .als entirely". |
| Nothing plays after firing | Clip is quantized to 1 bar — wait for the downbeat. If still nothing, the master transport isn't running. |

## What's deliberately not in this session

Recs. The plan grid. ⌘K vibe search. The five role columns. FX. Journey
scoring. All of it works, none of it helps you learn to keep a beat going.
It'll still be there next week — and it'll make more sense once your hands
know what they're asking for.
