# Hardware

The physical rig that runs Dance. As of **2026-05-11**, $660 spent at Sweetwater (Kyle Sorensen, x3957 — 90-day price-match window through ~2026-08-11).

> **2026-05-16 — pending swap:** Messaged Kyle to upgrade the **Scarlett 2i2** → **Scarlett 4i4** for true independent cue/main outputs (well within the 45-day return window). The 2i2 section below is preserved for context; once Kyle confirms the swap I'll move the 2i2 to a "replaced" section and add the 4i4 properly.

Two independent USB chains. **Audio interface and controller never touch each other** — they both plug into the Mac.

```
AUDIO (out of Mac):
  Mac ──USB-C──▶ Scarlett 2i2 ──TRS-to-RCA──▶ Edifier R1700BT (main speakers)
                              └──1/4" jack──▶ GPM-103 adapter ──▶ Bose (cue headphones)

CONTROL (into Mac):
  APC40 mk2 ──USB-C-to-B──▶ Mac ──▶ Ableton Live (Session View)
```

## Bill of materials

| Item | Price | Role |
|---|---|---|
| **Akai APC40 mk2** (new) | $329 | Performance controller — only MIDI device, drives Ableton Session View |
| **Focusrite Scarlett 2i2** (4th gen, new) | $215 | USB-C audio interface — main outs + headphone cue |
| StarTech USB-C to USB-C, 3 ft | $25 | Scarlett → Mac (USB-C port) |
| StarTech USB-C to USB-B, 10 ft | $23 | APC40 → Mac (USB-C port) |
| Hosa CPR-203 TRS-to-RCA pair, 3 m | $20 | Scarlett TRS outs → Edifier RCA inputs |
| Hosa GPM-103 1/4"-to-3.5mm adapter | FREE | Bose 3.5mm → Scarlett 1/4" headphone jack |
| Subtotal | $612 | |
| CA tax + free shipping | $47 | |
| **Total** | **$660** | |

Sales engineer: **Kyle Sorensen, Sweetwater, x3957**. Order detail in [`ORDER.md`](ORDER.md).

## Why each piece

### APC40 mk2 — the performance controller

**Role:** The only MIDI device. Drives Ableton's Session View directly.

**Why this one:**
- Faders (for stem volumes) + 16 knobs (for EQ) + crossfader (for deck A/B stem-set blending) + 5×8 RGB pad grid (for clip launching) — all on one device.
- Native Ableton mapping out of the box — no manual MIDI mapping pain.
- Real-instrument build quality, not plastic toy.
- Future-proof for the first 1–2 years of gigging.

**Why not alternatives:**
- APC Mini + nanoKontrol2 ($170): two cheap devices, "training wheels" feel, re-buy guaranteed in 6 months.
- Push 2 used ($450+): no faders — wrong for "fade in this stem" gestures.
- Push 3 ($1000+): out of budget, overkill.
- Pioneer DDJ / Traktor S2: wrong tool — designed for jog-wheel 2-deck mixing, not stem-in-Ableton.

**Caveat:** Discontinued by Akai in 2022. Bought new from Sweetwater while stock lasts. Still natively supported in Ableton 12.4.

### Scarlett 2i2 — the audio interface

**Role:** Gets audio out of the Mac with two independent volume controls (speakers + headphones) and balanced outputs for eventual gigs.

**Why not optional:**
1. Two independent outputs — laptop headphone jack physically can't do main + cue at separate volumes.
2. Balanced TRS outputs — required for clean signal to club mixers (no hum/interference).
3. Real DAC + low noise floor — laptop output isn't accurate enough for mixing decisions.
4. Low latency — designed for real-time audio.

**Why this specific interface:**
- Most widely-owned interface in the world, battle-tested.
- 4th gen (2023), USB-C native.
- $215 is the sweet spot — cheaper compromises on preamps/converters, more expensive adds features we don't need yet.

**Known limitation:** Only one stereo output pair. The headphone jack mirrors it. "Full" independent cue (separate hardware outputs for main vs cue) needs a 4-output interface (Scarlett 4i4, Audient iD14) — that's the **9–12 month upgrade trigger**.

### USB cables — both USB-C, both bought separately

**Why:** MacBook Pro M2 Pro 14" has 3× USB-C + 1× USB-A. Included cables would've used the USB-A port twice. USB-C cables let us:
- Use Mac's USB-C ports directly.
- Keep USB-A free for future devices.
- Avoid hub-in-the-middle complexity.

10 ft on the APC40 cable chosen for desk-rearrangement slack. **No USB hub needed** (originally planned, dropped).

### Hosa CPR-203 — TRS-to-RCA cables

Scarlett outputs are 1/4" TRS (pro standard); Edifier R1700BT wants RCA (consumer standard). This cable bridges them. No cable, no audio.

### Hosa GPM-103 — 3.5 mm-to-1/4" adapter

Scarlett's front headphone jack is 1/4" (pro standard); Bose headphones output 3.5 mm. Kyle threw it in free as bundle negotiation.

## What we **did not** buy (and why)

### DJ headphones — deferred 4–8 weeks

Originally considered: Audio-Technica M50x ($159) or Sennheiser HD25 (~$170).

**Why deferred:**
- Existing wired Bose work fine for the learning phase (ANC off, via 3.5 mm + GPM-103 adapter).
- First 4–8 weeks are learning Ableton's interface, not real cueing.
- Real DJ-headphone limitations only become obvious once cueing-while-main-plays is the workflow.
- By then we'll know with real context whether HD25 / M50x / Pioneer HDJ-X7 is right.

**Trigger to buy:** Cueing limitations become obvious. Bass-boosted Bose tuning makes mixes sound wrong on other systems; Bose cups don't swivel for "one cup off" technique.

### USB hub — dropped from plan

Originally going to buy an Anker USB-A hub ($12). USB-C cable approach removed the need.

## Existing gear (already owned)

| Item | Role |
|---|---|
| **MacBook Pro M2 Pro 14"** (2023, 16 GB RAM) | Primary DJ machine. The brain. |
| **Edifier R1700BT** | Powered bookshelf speakers, RCA inputs, 66W RMS. |
| **Bose** wired headphones | 3.5 mm jack, ANC kept off, used wired. |
| Standing desk | Shared with work setup (Mac mini + monitor stay). |
| **Ableton Live Standard 12.4** | Locked at this version for `.als` export compatibility. See `src/dance/als/templates/blank_live12.xml`. |

## Locked decisions (don't re-litigate)

1. **Ableton Live as the audio engine.** Not Traktor, not rekordbox, not custom audio code.
2. **Wired everything.** No Bluetooth, no Sonos, no AirPlay in the audio path.
3. **Single high-quality controller** (APC40 mk2) over multi-cheap-controller setup.
4. **New from Sweetwater**, not used market. Worth the premium for warranty + condition certainty.
5. **USB-C native** for all music gear. USB-A port stays free.
6. **Sweetwater + Kyle Sorensen** as primary retailer/relationship.
7. **Bedroom-first, gig-capable.** All gear works for desk now AND gigs later.

## Upgrade triggers

| When this becomes annoying | Buy this |
|---|---|
| Cueing while main plays is awkward on Bose | DJ headphones (HD25 / M50x / HDJ-X7) — $160–250 |
| Want truly independent cue/main outputs | Scarlett 4i4 / Audient iD14 — $300–350 |
| Outgrow APC40 grid or want haptic | Push 3 or grandMA3 — $1000+ |
| Need to take the rig out of the bedroom | Gig bag, power conditioner, XLR cables, IEC kettle leads |
