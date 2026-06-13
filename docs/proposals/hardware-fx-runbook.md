# FX Returns + APC40 Send-Mode — Live Author Runbook

> ⚠️ **Status: SUPERSEDED by [`fx-phase-1-runbook.md`](fx-phase-1-runbook.md).**
> **Do not follow this runbook.** The FX return-track names it specifies (`A-Reverb`, `B-Delay`, `C-Filter`, `D-Riser`) do **not** match the OSC bridge's exact-name requirement — the bridge looks up return tracks named **`Filter`**, **`Reverb`**, **`Delay`**, and **`Riser`** (bare names, no `A-`/`C-` prefix). Authoring returns with the names below yields **dead FX**: the bridge can't find them, so the throws/sweeps silently do nothing. Use the phase-1 runbook for the current, verified return-track naming and wiring. This doc is kept for historical context only.

Status: **runbook** — to be executed by you (the DJ) with Ableton Live 12.4 open. Not auto-executable from Claude because the `.als` template is extracted from a real Live save and must round-trip through Live to stay valid.

Companion to [hardware-and-live-controls.md](hardware-and-live-controls.md) §2 (FX returns) and §3 (risers).

Goal: turn the wasted APC40 surfaces (top 8 channel knobs in Send mode + Scene Launch column) into useful per-stem FX ramps + riser one-shots, with **zero code changes** to our pipeline.

---

## Current template state

```
$ grep EffectiveName src/dance/als/templates/blank_live12.xml
1-MIDI       — empty MIDI for APC40 native use
2-MIDI       — empty MIDI
3-Audio      — Drums stem column
4-Audio      — Bass stem column
A-Reverb     — already present (empty shell — needs device loaded)
B-Delay      — already present (empty shell — needs device loaded)
Main         — master
0-Main       — Cue track (Scarlett outs 3/4)
```

So Returns A and B already exist as named shells but probably have no actual device inside. We need to (1) populate A and B with real devices, (2) add Returns C and D, (3) re-export the template.

> ⚠️ **One rule.** After every save, regenerate a fresh `.als` for a real track via `dance export-als <id>`, open it in Live, and confirm: 5 stem tracks load, audio plays, no "Missing Media" dialog, no device-load errors. This catches the most common breakage (template path drift, device version mismatch, plugin missing).

---

## Step 1 — open the template in Live

The template is the *raw XML* of an Ableton project — Live can't open it directly. Two paths:

**Option A (recommended):** open the last `.als` you generated (e.g. the most recent file in `$HOME/dance-sets/`). That `.als` is built from the template + a track, so any change you save back out can be re-extracted into the template.

**Option B:** rename `blank_live12.xml` → `blank_live12.als.gz`, gzip it (`gzip -k blank_live12.xml && mv blank_live12.xml.gz blank_live12.als`), open in Live.

After your edits you'll *reverse* this in Step 5.

---

## Step 2 — populate Returns A & B

| Return | Device chain | Macro 1 default |
|---|---|---|
| **A-Reverb** | Reverb (built-in) → "Hall" preset; Decay 6 s, Dry/Wet 100% (the *send* controls how much hits it) | — |
| **B-Delay** | Ping Pong Delay → 1/4 sync, Feedback 50%, Dry/Wet 100% | — |

Per Return: drop the device, dial it in, save. That's it — no automation needed; the APC40's Send-mode knobs will modulate how much of each stem feeds in.

---

## Step 3 — add Returns C & D

Right-click any track header → **Create Return Track**, twice.

| Return | Rename to | Device chain |
|---|---|---|
| **C** | `C-Filter` | Auto Filter → LP 24dB, Frequency mapped to Macro 1, Resonance 0.3 |
| **D** | `D-Riser` | Utility (gain) → Auto Filter (HP, swept up by macro) → Saturator (drive) → optional: Reverb tail. Macro 1 sweeps the HP cutoff |

The **D-Riser** chain is the secret sauce. When you ramp a stem's Send D up over 4 bars, the stem gets piped through a rising HP filter + drive — instant "lift before the drop" without baking a sample. (For a more dramatic riser, also reserve the FX scene row in Step 4.)

---

## Step 4 — add an "FX one-shots" scene row

Add a fresh scene at the bottom of the session (Create → Insert Scene). In that scene's slot **on Return A** (or a dedicated audio track if you prefer), drop a pre-rendered riser sample:

- `assets/fx/riser_4bar.wav` (you'll provide — any pitched-up white-noise riser, 4 bars at 120 BPM)
- Set the clip to **Trigger** launch mode, **Warp on**, **Master** sync — so it always starts on the next bar regardless of when you hit Scene Launch.
- Optional: add a reverse cymbal in another slot, a "reverb throw" silence-trigger in another.

Now hitting the APC40 Scene Launch button for that row = fire the riser on the next downbeat.

---

## Step 5 — re-export the template

If you used Option A (real `.als`):
```bash
gunzip -kc <your-set>.als > /tmp/exported.xml
# Strip just the Project header + Tracks block back into blank_live12.xml.
# Easiest: open both in a diff tool, copy the new returns/scene over.
```

If you used Option B (renamed template):
```bash
gunzip -kc src/dance/als/templates/blank_live12.als > src/dance/als/templates/blank_live12.xml
rm src/dance/als/templates/blank_live12.als
```

Then **regenerate a test set** and open it in Live:
```bash
dance export-als <track-id>
open ~/dance-sets/<track-title>.als
```

If Live throws *anything*, see [docs/troubleshooting.md](../troubleshooting.md) "Live rejects the .als entirely" — usually a stray device version or absolute path that doesn't survive the round-trip.

---

## Step 6 — APC40 Send-mode cheatsheet (taped to the back of the controller)

Once Returns A-D exist, **no software changes needed** — Live's native APC40 driver automatically maps the top 8 channel knobs in Send mode:

| APC40 mode | What the top 8 knobs do |
|---|---|
| **Pan** | Per-column stereo pan (mostly leave alone — stems are stereo-aware) |
| **Send A** | How much each stem feeds the **Reverb** return |
| **Send B** | How much each stem feeds the **Delay** return |
| **Send C** | How much each stem feeds the **Filter** return (HP/LP sweep) |
| **Send D** | How much each stem feeds the **Riser** return (HP+drive lift) |
| **User 1-4** | Unused — reserved for future MIDI→OSC bridge (see [hardware-and-live-controls.md](hardware-and-live-controls.md) §2) |

| Live moment | What to grab |
|---|---|
| "Need a build-up" | Send D up on drums + bass over 4 bars; Scene Launch the riser sample on the last bar |
| "Drop's flat" | Send A on the vocal mini-burst for 1 bar = tail throw |
| "Transition out" | Send C on full mix, sweep filter shut over 8 bars |
| "Big break" | Mute drums + bass; Send B up on vocals for delay throws |

---

## What the companion app needs (minimal UI surface)

Per your request: *"whatever we need is simply reflected in our UI as simple as possible."*

For Phase 1 (this runbook), **the UI needs nothing new.** The APC40 controls Live directly via MIDI; our app shows the resulting stem volumes / play state through the existing OSC subscription. No new buttons, no new endpoints.

Phase 2 (post-runbook, if you want it) — three tiny additions worth ~50 LOC of UI:

1. **Send-level meters** in the Booth header — 4 small bars (A/B/C/D) showing current send-knob position per active stem column. Read-only mirror of what your hand is doing on the APC. ~20 LOC, one new OSC subscription.
2. **"Riser armed" indicator** — when the FX scene row's next slot is triggered (clip queued, not yet firing), a faint pulse on the Booth header. Tells you "I pressed it, the bar boundary's coming." ~10 LOC.
3. **A `FX` chip in `BoothColumnHeaders`** — appears red when any send on that column is > 0.3. Glanceable "this stem is being processed right now." ~20 LOC.

All three are deferable until you've actually used Phase 1 in a set and know which would help.

---

## Validation checklist before declaring this done

Per CLAUDE.md "Real-data verification" rule:

- [ ] Open the new `.als` in Live — all 5 stem tracks load, no Missing Media dialog
- [ ] APC40 Send A knob raises a stem's reverb wetness audibly
- [ ] APC40 Send D knob + Scene Launch fires the riser sample on the next bar
- [ ] Tempo / play / stop / scene launch all still work natively
- [ ] No CPU spike > 50% on the M2 Pro under normal session
- [ ] Re-running `dance export-als <id>` produces a fresh `.als` that *also* opens cleanly (template wasn't corrupted by the round-trip)

Report back what worked, what didn't, and we'll triage from there.
