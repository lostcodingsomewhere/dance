# Hardware

The physical rig that runs Dance. Initial buy **2026-05-11** ($660 at Sweetwater, Kyle Sorensen x3957). **2026-05-18:** swapped the Scarlett 2i2 → **Scarlett 4i4** (4th gen) inside the 45-day return window to unlock true independent cue/main outputs.

Two independent USB chains. **Audio interface and controller never touch each other** — they both plug into the Mac.

```
AUDIO (out of Mac):
  Mac ──USB-C──▶ Scarlett 4i4 ──Outs 1/2 (TRS)──▶ Hosa CPR-203 ──▶ Edifier R1700BT (main speakers)
                              └──Outs 3/4 (TRS or front headphone jack)──▶ GPM-103 adapter ──▶ Bose (cue)

CONTROL (into Mac):
  APC40 mk2 ──USB-C-to-B──▶ Mac ──▶ Ableton Live (Session View)
```

The 4i4 routes Outputs 1/2 to the main speakers and Outputs 3/4 to a separate stereo cue bus. In Ableton: **Preferences → Audio → Output Config:** enable both `1/2` and `3/4`; **Master → Audio To: 1/2**; **Cue Out: 3/4**. The Scarlett's front headphone amp can be assigned to mirror 3/4, which is the standard DJ workflow (preview a clip in headphones without it leaking to the master).

## Bill of materials

| Item | Price | Role |
|---|---|---|
| **Akai APC40 mk2** (new) | $329 | Performance controller — only MIDI device, drives Ableton Session View |
| **Focusrite Scarlett 4i4** (4th gen, new) | ~$299 | USB-C audio interface — main outs 1/2 + cue outs 3/4 |
| StarTech USB-C to USB-C, 3 ft | $25 | Scarlett → Mac (USB-C port) |
| StarTech USB-C to USB-B, 10 ft | $23 | APC40 → Mac (USB-C port) |
| Hosa CPR-203 TRS-to-RCA pair, 3 m | $20 | Scarlett outs 1/2 → Edifier RCA inputs |
| Hosa GPM-103 1/4"-to-3.5mm adapter | FREE | Bose 3.5mm → Scarlett 1/4" headphone jack |
| Subtotal | ~$696 | |
| CA tax + free shipping | ~$54 | |
| **Total** | **~$750** | exact figure confirmed when Sweetwater issues the swap RMA |

Sales engineer: **Kyle Sorensen, Sweetwater, x3957**. Order detail (initial buy + 2i2→4i4 swap) in [`ORDER.md`](ORDER.md).

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

### Scarlett 4i4 — the audio interface

**Role:** Two independent stereo output pairs so the master mix and the cue/preview bus can leave the box on different channels. Plus the usual: real DAC, low noise floor, balanced TRS outs for eventual gigs.

**Why not optional:**
1. **Two independent stereo outs** — DJ cueing requires sending the candidate clip to headphones *without* it leaking to the master. A 2-output interface (e.g. the 2i2 this rig started with) can't do that; the headphone jack mirrors outs 1/2.
2. Balanced TRS outputs — clean signal to club mixers (no hum/interference).
3. Low latency, real-time audio path.

**Why this specific interface:**
- 4th-gen 4i4 (2023+), USB-C native, well-supported on macOS via CoreAudio (no driver install).
- Independent monitor mix on the front panel — useful for "more click in headphones" without re-routing in software.
- Standalone mode (works as a clean DAC even with the Mac off) — gig insurance.

**Routing for cue/preview:**
- **Outs 1/2** → Edifier R1700BT speakers (Ableton master).
- **Outs 3/4** → cue bus (Ableton's Cue Out). Front headphone jack can be assigned to mirror 3/4 via the Scarlett's Direct Monitor / Focusrite Control mix.
- Net effect: when the user clicks "preview" on a rec card or solo's a deck in Live, audio goes to the Bose only — the speakers keep playing the master untouched.

**History:** Rig launched 2026-05-11 with a Scarlett 2i2 4th gen ($215, single stereo pair). The 2i2 was returned and swapped for the 4i4 on 2026-05-18 inside Sweetwater's 45-day window once it was clear independent cue was a workflow requirement, not a nice-to-have.

### USB cables — both USB-C, both bought separately

**Why:** MacBook Pro M2 Pro 14" has 3× USB-C + 1× USB-A. Included cables would've used the USB-A port twice. USB-C cables let us:
- Use Mac's USB-C ports directly.
- Keep USB-A free for future devices.
- Avoid hub-in-the-middle complexity.

10 ft on the APC40 cable chosen for desk-rearrangement slack. **No USB hub needed** (originally planned, dropped).

### Hosa CPR-203 — TRS-to-RCA cables

Scarlett outputs are 1/4" TRS (pro standard); Edifier R1700BT wants RCA (consumer standard). This cable bridges the **master pair (outs 1/2 → Edifier)**. Outs 3/4 don't need a TRS-to-RCA cable since the cue path terminates at the front headphone jack.

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
| Outgrow APC40 grid or want haptic | Push 3 or grandMA3 — $1000+ |
| Need more inputs (recording a guest, multi-channel feed) | Already covered — 4i4 has 4 ins |
| Need to take the rig out of the bedroom | Gig bag, power conditioner, XLR cables, IEC kettle leads |
