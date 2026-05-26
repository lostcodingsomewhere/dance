# Hardware-and-Live-Controls Discussion

Status: **discussion / proposal — not implemented**
Audience: you (the DJ) + future-Claude
Goal: figure out (1) how to fix the waveform-click-past-halfway issue in a way that fits your section-snap intent, (2) what to do with the APC40 mk2 surfaces we currently waste, and (3) how "speedup / risers / drops" should actually work given the stem-in-Live architecture. Bias toward **simple, few moving parts**, no Bluetooth, no extra devices.

---

## 1. Waveform click — what's actually wrong

### Code today

[`companion-app/src/components/ComboStrip.tsx:182`](../../companion-app/src/components/ComboStrip.tsx):

```ts
function handleSeek(ratio: number) {
  if (!duration || !clipBpm || clipBpm <= 0) return;
  const sectionStarts = (regions.data ?? [])
    .filter((r) => r.region_type === "section")
    .map((r) => r.position_ms / 1000 / duration)
    .filter((s) => s <= ratio + 0.001) // small epsilon
    .sort((a, b) => b - a);
  const snapped = sectionStarts.length > 0 ? sectionStarts[0] : 0;
  const beats = snapped * duration * (clipBpm / 60);
  onSeek(beats);
}
```

The math is fine. The behavior is **intentional snap-to-section** — exactly what you said you want: "click and have it go to that set section point starting from beginning of it."

### Why it feels broken past halfway

The snap takes the largest section start `≤ click ratio`. If `detect_regions.py` doesn't emit any section past ~50% for the track you're testing, *every* click in the second half snaps back to the last first-half section, and the playhead never moves forward. Exact match for the symptom.

Three possible causes (need to verify on a real track, not guess):

1. **Section detector under-segments long tracks.** `detect_regions.py` uses phrase-bar boundaries ≥ 8 bars. Some genres (long-form house) only produce 2-3 phrases. Outro is one giant section starting at 50%.
2. **Region positions are in ms but `duration_seconds` is the *clip* duration**, not the source-track duration. If the .als writer trims or our `cell.duration_seconds` reports something different from what regions were detected against, the ratio normalization (`position_ms / 1000 / duration`) maps late sections off the right edge and they get filtered out by the `≤ ratio` clamp.
3. **Stem waveform vs. track regions.** Regions are detected on the mix; the visualizer might show a stem waveform of a different length than the analyzed mix.

### Proposed fix (cheap, fits your intent)

- **Visualize the section ticks** clearly on the waveform so you can see what the snap targets are. `Waveform.tsx` already renders region ticks (`b.x/100`) — confirm they appear across the *whole* waveform on the offending track. If they're all bunched in the first half, the snap is doing what you asked, the data is just sparse.
- **If snap-target is sparse: fall back to "snap to nearest section-or-cue, including ones to the *right*."** Right now it only ever snaps backward. Change the rule: snap to the closest section (left or right) within some tolerance (say 4 bars in beat-distance); past that, do a precise seek to the click position. This gives you "click roughly here → land on the nearest drop," not "click anywhere right of halfway → land at 50%."
- **Hover preview**: show the snap target as a ghost playhead while hovering, so you see *where* the click will land before committing. Tiny UX win, ~10 lines.
- **Stretch**: a modifier key (shift-click) for precise seek that bypasses snap entirely. Useful for cueing into a specific drum hit.

Open question: do you also want cue points (`region_type === 'cue'`) as snap targets, or sections only? Cues are denser — could clutter, or could be exactly what you want for jumping into specific drum fills.

---

## 2. APC40 mk2 — what's mapped today vs. what's possible

**Currently mapped: nothing of ours.** The APC40 runs in Live's stock control-surface mode. That covers a lot:

| APC40 surface | Live's native behavior (free, works today) |
|---|---|
| 8×5 clip grid | Launches clips in the selected scene/tracks |
| Track strip faders | Channel volume per track (= per stem column in our `.als`) |
| Solo / Mute / Rec arm | Per-track solo / mute |
| Track Select | Selects a track |
| Clip Stop | Stops a clip in that track |
| Master fader, Cue Level | Master volume, headphone cue level |
| Crossfader | A/B crossfader (not used in stem-DJ — leave centered) |
| Tempo encoder | Tempo nudge |
| Transport (play / stop / record) | Transport |
| Scene Launch buttons (right column) | Launch a whole scene = "switch to the next song's combo" |

That's the bottom 80% of stem-DJing already covered for free. Don't touch it.

### What's wasted today

The APC40 mk2 has these surfaces that Live maps to *device parameters* by default, which is useless to us because we don't have a device chain we care about per track:

| Surface | Default Live behavior | What it could do for us |
|---|---|---|
| **8 Device Control knobs** (right side, "Device" section) | Macros of the selected device | Master-bus FX rack macros: HPF, LPF, reverb wet, delay wet, riser wet, sidechain depth, drive, "drop call" macro |
| **8 channel knobs (top), Pan mode** | Per-channel pan | Mostly leave to Pan. Stems benefit from occasional pan widening — useful for transitions |
| Same 8 knobs, **Sends A/B/C/D** mode | Per-channel send to return tracks A-D | These become our FX returns: **A = Riser bus**, **B = Reverb tail**, **C = Delay throw**, **D = Filter sweep**. Send a stem to a return to ramp it into the FX |
| Same 8 knobs, **User 1/2/3/4** mode | Custom MIDI we define | Our companion-app actions: User 1 = "jump to next section" per column, User 2 = "fire the prepped riser scene," etc. |
| **Pan / Send / User mode buttons** | Switches what the top 8 knobs do | Same as above |
| **8 Scene Launch buttons** | Launch entire scene row | A whole scene = one song's stem combo. Already what we want |
| Shift, Bank, Nudge +/- | Modifier + bank-switch | Useful for accessing more than 8 tracks worth of stuff — we only have 5 columns so Bank isn't urgent |

### Proposed mapping (keep simple)

Three layers, no custom firmware:

1. **Layer 1: stock Live APC mode.** Clip grid, track faders, solo/mute, transport, tempo, scene launch, master, cue. *No setup needed.*
2. **Layer 2: an FX-return rack on the master, populated in our `.als` template.**
   - Return A: **Riser** (white-noise sample with low→high filter sweep + reverse cymbal; LFO-modulated cutoff)
   - Return B: **Reverb tail** (big plate, ~6 s decay, sidechained to master kick)
   - Return C: **Delay throw** (1/4-note ping-pong, feedback ~50%)
   - Return D: **Filter sweep** (Auto Filter, manual cutoff)
   - Channel Sends knobs (in Send A/B/C/D mode) become "ramp this stem into the riser / reverb tail / delay throw / filter."
   - Master Device-Control knobs become master HPF/LPF/drive/sidechain.
3. **Layer 3: User mode on the top 8 knobs for our *companion-app actions* (over MIDI → OSC bridge → our API).**
   - User-1 knob per column: scroll the column's next-song recommendation list (turn knob = highlight different rec; pad above = load it)
   - User-2 knob per column: nudge stem pitch / rate independently (for harmonic-mix correction without retiming the whole track)
   - User-3: arm "drop-call" — next bar boundary triggers a pre-defined scene swap
   - User-4: rec / preview to cue bus (already wired via Solo, but a one-button "preview the rec" is nicer)

   We'd need a tiny **MIDI listener** alongside our OSC listener to consume User-mode CC messages and route them into companion-app commands. Could live in `src/dance/osc/midi_bridge.py`.

### Decision points for you

- Do you want User mode to even exist, or is "Send mode for FX returns + native everything else" enough? (Simpler = better unless you actually want app integration from the hardware.)
- How many FX returns are you OK adding to the `.als` template? More returns = more CPU and more visual noise in Live.

---

## 3. Speedups (x2), risers, drops — how to actually do them

You have **three primitives** for this, in increasing complexity:

### a) Per-clip warp rate change ("x2 this stem")

Each stem clip in our `.als` is warped. Changing the clip's *warp markers* doubles playback speed. In Live: select clip, double the warp ratio. Via OSC: we already have `set_tempo` but not per-clip warp manipulation. AbletonOSC exposes `/live/clip/set/looping`, `/live/clip/set/loop_end`, but not warp-mode directly — would need to investigate.

**Cheap version:** add a second copy of each stem clip pre-rendered at 2x rate (offline) into an empty scene row, labeled "drop scene." Launch that scene = 2x for one bar then back. Zero runtime CPU, zero new OSC.

### b) Tempo automation (whole-mix speedup)

Ableton can ramp the master tempo via an automation envelope on the master track. We could pre-bake a "tempo-up 4 bars 1x→1.5x→1x" automation in a dummy MIDI clip in our `.als` template. Triggering the clip ramps tempo, then snaps back.

This is **the simplest 'drop' effect** — one scene-launch button = a built-in tempo build.

### c) Riser / FX scene (recommended primary)

A scene row dedicated to FX one-shots:
- Slot in Return A → white-noise riser sample, 4 bars long
- Slot in Return B → reverse cymbal hit
- Slot in Return C → reverb throw (silence clip that triggers a long reverb tail on the master)

Launching the scene fires the riser; it auto-stops when done. APC40's Scene Launch button (right column) = "fire the next FX combo."

**Recommendation: do (c) first, then (a) if you want surgical drops on a specific stem.** Skip (b) until you actually want it — tempo automation is fiddly to undo live.

### Decision points for you

- Do risers live in dedicated FX scenes, or as send-knob ramps (Send A = riser wetness)? Scenes = one-shot, sends = continuous control. Both are doable. Both might be too much.
- Do we want "drops" as pre-baked 2x clips (zero risk), or runtime warp-rate changes via OSC (more dynamic, more breakage risk)?

---

## 4. The "keep it simple" recommendation

If we commit to all of this it's a lot. A small first step that gets you 80% of the value:

1. **Fix the waveform snap so it can jump forward** (proposal §1 — small frontend change).
2. **Add FX returns A-D to the `.als` template** (Riser, Reverb, Delay, Filter). Stock APC40 Send mode now controls FX ramps. Zero new code.
3. **Add 1-2 "FX scene" rows** with pre-rendered riser / reverse-cymbal / reverb-throw samples. Scene launch via APC40 right-column button.
4. **Defer User-mode MIDI bridge** until you've lived with steps 1-3 for a few sets and know what's missing.

This needs:
- ~30 lines of TypeScript (snap-tolerance + hover preview)
- A one-time edit to `src/dance/als/templates/blank_live12.xml` (add 4 return tracks + a riser clip slot — needs Live-side authoring + re-export per the "don't change `.als` writer without re-verifying" rule)
- A handful of audio samples committed to a new `assets/fx/` (riser.wav, reverse_cymbal.wav, reverb_throw.wav)

No new OSC, no new MIDI bridge, no API changes.

---

## Things I don't know yet — questions back at you

1. On a track where you can reproduce "can't click past halfway," can you check what section regions exist via `curl 'http://127.0.0.1:8000/regions/<track_id>'` (or whatever the route is) so we know if cause #1 vs #2 vs #3 above is real?
2. Do you actually use the APC40 crossfader, or is it always centered? (Affects whether we wire it to anything.)
3. How many FX-return tracks is too many before Live feels crowded in the session view at your laptop's resolution?
4. Do you want stem-pitch independence (different pitch per stem) at all, or is that always "bad idea" territory for you?

If you have answers, I'll fold them in and we can pick a slice to actually implement.
