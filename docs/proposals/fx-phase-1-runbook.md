# FX Phase 1 — Filter Return + Riser One-Shot (Live Author Runbook)

Status: **runbook** — execute in Ableton Live, 5–10 minutes, one-time per `.als` template refresh.

What this gives you:
- **HPF / Filter toggle** per deck — one click in the app, all four stems on that side get a parallel filter send. Snap on for the transition, snap off after.
- **Riser one-shot** — fires a pre-rendered build-up sample, quantized to the next bar so it lands on a downbeat.

The companion app's wiring is already shipped (commit `5af21d6+`). The HPF and ▲ Riser buttons render but stay disabled until Live has the Filter return + Riser clip authored. As soon as those exist and the bridge re-syncs, the buttons enable themselves automatically — no app restart needed.

---

## Prerequisites

- Ableton Live 12.4 open with your most recent `dance export-als` set loaded.
- A pre-rendered riser sample. Either:
  - Buy/download one (Splice has hundreds — search "white noise riser 4 bar 120 BPM").
  - Or use a Live preset: drag in any **Noise** generator + an **Auto Filter** with a long sweep, render the result, drop the rendered audio into the FX clip slot in Step 3.

---

## Step 1 — Create the Filter return track

1. In Ableton's Session View, **right-click any track header → Create Return Track**.
2. **Rename** the new return to exactly `Filter` (case-sensitive — the bridge matches on this).
3. **Drag an Auto Filter** device onto the Filter return's chain.
4. Configure Auto Filter:
   - **Filter Type**: Hi-Pass 12 dB
   - **Frequency**: ~400 Hz (this is the "filter on" cutoff — adjust to taste)
   - **Resonance**: 0.30 (slight peak, not screamy)
5. **Map Macro 1** (if you want runtime macro control later, optional for Phase 1):
   - Right-click Macro 1 → MIDI Map mode → click Filter Frequency → save.
   - Skip if you just want fixed-cutoff filtering.
6. **Crossfade group**: leave at None. The Filter return is always audible regardless of the master crossfader; the per-deck send level is what makes it deck-aware.

> ⚠️ The return's name must be exactly `Filter`. The bridge scans Live's track names at startup and on resync — typos = no discovery.

---

## Step 2 — Verify the Filter wiring

1. In the companion app, click **↻ RESYNC** in the header (top right of MasterStrip).
2. The **HPF** button in each deck header should turn from "disabled greyed-out" to "active outline".
3. Click HPF on Deck A:
   - Live's Session View → Deck A stem tracks → Send-A knob should jump to 100%.
   - Audio: Deck A audibly thins (lows cut).
4. Click HPF again to toggle off. Sends snap back to 0%. Full range returns.

If the button stays disabled: open Live's track list and confirm the return name is exactly `Filter`. Check the backend log (`/tmp/dance-backend.log`) for the `"No 'Filter' return track in Live"` warning.

---

## Step 3 — Create the Riser one-shot

The riser lives as an audio clip in a dedicated row, fired one-shot. Two layout options:

### Option A (recommended) — Add a dedicated `FX` track

1. **Create Audio Track** anywhere right of the existing decks. Name it exactly `FX`.
2. **Pick a scene** beyond your normal-set scenes (say scene 7).
3. **Drag your riser sample** into scene 7's slot on the `FX` track.
4. **Rename the clip** to exactly `Riser` (this is what the bridge looks up).
5. Configure the clip:
   - **Warp**: ON, mode "Beats", set to your set's master BPM.
   - **Loop**: OFF (we want one-shot, not loop).
   - **Launch Mode**: Trigger (so re-firing restarts it cleanly).
   - **Launch Quantization**: 1 Bar (matches the rest of the rig — fires on the next downbeat).
   - **Output Routing**: Master (it should bleed through both decks, not follow the crossfader).
6. **Crossfade group**: None.

### Option B — Put the riser on the Filter return

Same setup, but put the audio clip on scene 7 of the `Filter` return instead of a new track. Saves one track but couples Filter and Riser routing.

---

## Step 4 — Verify Riser

1. Click **↻ RESYNC** in the companion app.
2. The **▲ Riser** button (centered above the two decks, amber when ready) should enable.
3. Click it mid-set. The riser fires on the **next bar boundary** (not instantly — that's the 1-Bar quantization).
4. After the sample plays out, it auto-stops (no loop). Re-click to fire again.

If the button stays disabled: open Live, confirm the clip name is exactly `Riser` and the track name is exactly `FX` (Option A) or `Filter` (Option B).

---

## Step 5 — Re-export the template (lock the changes in)

So future `dance export-als` calls include the Filter return + Riser clip:

1. **Save your `.als`** in Live (Cmd-S).
2. Unpack it:
   ```bash
   gunzip -kc "$(ls -t ~/dance-sets/*.als | head -1)" > /tmp/exported.xml
   ```
3. **Open both** `/tmp/exported.xml` and `src/dance/als/templates/blank_live12.xml` in a diff tool.
4. **Copy the new `<ReturnTrack>` blocks** (Filter, and if Option A then also the new `FX` `<AudioTrack>`) into the template, after the existing returns and before the Main track.
5. **Save** the template.
6. Test by generating a fresh set for an unrelated track:
   ```bash
   dance export-als <some-other-track-id>
   open ~/dance-sets/<title>.als
   ```
7. Confirm in Live that the new set has the Filter return + Riser clip baked in.

> ⚠️ Per CLAUDE.md "Real-data verification" — after editing the template, generate one fresh `.als` and open it. Confirm 9 audio tracks + 2 mix tracks + Filter return + FX track all load, audio plays, no Missing Media dialog. If anything breaks see [docs/troubleshooting.md](../troubleshooting.md) "Live rejects the .als entirely".

---

## What to do if you mess up

The `.als` template's a real Live save. Worst case:
```bash
cd src/dance/als/templates/
git checkout -- blank_live12.xml
```
Go back to Live and try again. Nothing irreversible.

---

## Phase 2 — Reverb throw + Delay throw

Same pattern as Filter. Two more return tracks:

### Reverb return

1. Right-click → Create Return Track. Rename to **`Reverb`** (exact, case-sensitive).
2. Drag a **Reverb** device onto its chain.
3. Preset: **Hall** or **Plate**. Decay: **6 s**. Predelay: **40 ms**. Dry/Wet: **100%** (the per-deck send controls how much hits it).
4. Crossfade group: None.

### Delay return

1. Create Return Track. Rename to **`Delay`** (exact).
2. Drag a **Ping Pong Delay** onto its chain (or Filter Delay — your taste).
3. Sync: **on**. Time: **1/4 note**. Feedback: **45%**. Dry/Wet: **100%**.
4. Crossfade group: None.

**Verify**:
1. ↻ RESYNC in the app.
2. **REV** and **DLY** buttons in each deck header should enable (the rose-cyan and fuchsia squares).
3. Click REV on Deck A → that deck audibly gains a big reverb tail. Click again → dry returns.
4. Same for DLY → tempo-synced echoes layer in.

Throw workflow: REV on, let the next phrase ring out, slam crossfader to B, REV off. The throw covers the cut.

## Phase 3 — Continuous filter sweep (no extra Live setup)

Already shipped. **Shift-click the HPF button** to sweep the filter 0→1 over 4 bars instead of snap-toggling. Same gesture works either direction — shift-click again to sweep back out.

The bridge uses a background timer that fires ~25 interpolated send-level updates over the duration. Ease-out cubic curve (accelerates then slows) — feels musical, matches what analog filter sweeps do.

You can tune the sweep duration via the OSC API directly if you want longer/shorter ramps:
```
curl -X POST 'http://localhost:8000/api/v1/ableton/fx/filter/a/sweep?bars=8&direction=in'
```

Default: 4 bars, direction auto (toggles based on current state).

---

## Validation checklist before declaring this done

- [ ] Filter return track named `Filter` with Auto Filter HP @ 400 Hz
- [ ] HPF buttons in app enable after RESYNC
- [ ] Clicking HPF audibly thins the deck (only that deck, not both)
- [ ] Toggling HPF off restores full range
- [ ] Riser clip named `Riser` on `FX` (or `Filter`) track
- [ ] ▲ Riser button enables after RESYNC
- [ ] Clicking ▲ Riser fires the sample on the next bar
- [ ] Template re-exported; a fresh `dance export-als` keeps the new structure
- [ ] No regressions: existing 10-deck stem loading still works, anchors still detect, crossfader still routes

Report back what worked, what didn't, and we'll move to Phase 2.
