# AbletonOSC setup

The `dance` companion app talks to Ableton Live via [AbletonOSC](https://github.com/ideoforms/AbletonOSC), an open-source MIT-licensed Live remote-script that exposes the Live API over OSC.

You install it once into your Ableton Live `Remote Scripts` folder, restart Live, and enable it as a Control Surface. The dance backend then sends/receives OSC on default ports 11000/11001.

## Install steps (macOS, Live 11/12)

1. Quit Ableton Live.
2. Clone AbletonOSC into Live's MIDI Remote Scripts directory:

   ```bash
   git clone https://github.com/ideoforms/AbletonOSC.git \
     ~/Music/Ableton/User\ Library/Remote\ Scripts/AbletonOSC
   ```

   (If `Remote Scripts` doesn't exist, create it.)

3. Start Ableton Live.
4. Open **Live → Preferences → Link, Tempo & MIDI → Control Surface**.
5. In the **Control Surface** dropdown choose **AbletonOSC**.
6. Leave **Input** and **Output** at their default (no MIDI device needed).

Live will load AbletonOSC. The default ports are:
- **11000** — Live listens here (the dance backend sends commands to this port).
- **11001** — Live sends to here (the dance backend receives state updates on this port).

## Verifying

With AbletonOSC loaded, run the dance backend (Phase 2.2) and check the logs for an `OSC listener bound on 127.0.0.1:11001` line. From the Python REPL:

```python
from dance.osc import AbletonOSCClient
client = AbletonOSCClient()
client.set_tempo(128.0)
```

Live's tempo display should jump to 128. If nothing happens:
- Confirm AbletonOSC is selected as a Control Surface (it'll show a green status indicator).
- Check Live's status bar for OSC log lines.
- macOS firewall may block UDP — allow Python to receive incoming connections.

## What dance uses

- **Reads:** tempo, beat position, currently playing clip per track, track volume, **per-deck Solo** (→ `soloed_kinds`), master crossfader, song num_tracks.
- **Sends:** transport (play/stop), clip launch/stop, tempo set, track volume/pan/mute/solo, **track crossfade-assign** (see patch below), track name/color, create/delete audio track, **create audio clip from a sample file** (Live 12.0.5+, see patch below), create/delete clip, set clip warp/loop/color/name, status-bar message.

See `src/dance/osc/client.py` for the full method list and `src/dance/osc/bridge.py` for how state pushes flow into the FastAPI WebSocket layer.

## One-time AbletonOSC patch for auto-loading stems

Live 12.0.5 quietly added `ClipSlot.create_audio_clip(path)` to the Live Object Model (confirmed on [Cycling74 forums](https://cycling74.com/forums/function-createaudioclip-class-clipslot-and-track-missing-lom)) — you can load an audio sample from disk straight into a session-view clip slot. Upstream AbletonOSC hasn't merged a handler for it yet ([PR #196](https://github.com/ideoforms/AbletonOSC/pull/196), [PR #168](https://github.com/ideoforms/AbletonOSC/pull/168) — both target arrangement view, neither merged), so we patch our local install.

Add one line to `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC/abletonosc/clip_slot.py`:

```python
methods = [
    "fire",
    "stop",
    "create_clip",
    "delete_clip",
    "create_audio_clip",   # ← Live 12.0.5+
]
```

Then **fully quit and reopen Ableton** (toggling Control Surface off/on isn't enough — Live's Python keeps modules in `sys.modules`). The handler is now bound to `/live/clip_slot/create_audio_clip (track_idx, slot_idx, absolute_path)`.

Known quirk: AbletonOSC's wrapper logs each call via `logger.info(track, slot, rv)` which mangles into a Python `logging` format-string error after a successful call. This shows up as `RemoteScriptError: Message: <track>` in Live's log but **the audio clip is created successfully** — the exception fires *after* the LOM side effect. Harmless.

## One-time AbletonOSC patch for crossfader assignment

Stock AbletonOSC exposes a track's mixer `volume`, `panning`, and `send`, but **not** `mixer_device.crossfade_assign` — the property that decides which side of Live's master crossfader a track follows. Without it, the crossfader has no effect on app-loaded stems (every track defaults to "None"), so the static `.als` export gets crossfader blending but a live-loaded deck doesn't. We add a small handler to our local fork to close that gap.

`crossfade_assign` is a plain integer attribute on `track.mixer_device` (not a `DeviceParameter` with a `.value`, the way `volume`/`panning` are), so it needs dedicated get/set handlers rather than the generic mixer-property path. Add these inside `init_api` in `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC/abletonosc/track.py`, right above the `mixer_properties_rw = ["volume", "panning"]` line:

```python
def track_get_crossfade_assign(track, params: Tuple[Any] = ()):
    return track.mixer_device.crossfade_assign,

def track_set_crossfade_assign(track, params: Tuple[Any] = ()):
    value, = params
    track.mixer_device.crossfade_assign = int(value)

self.osc_server.add_handler("/live/track/get/crossfade_assign",
                            create_track_callback(track_get_crossfade_assign))
self.osc_server.add_handler("/live/track/set/crossfade_assign",
                            create_track_callback(track_set_crossfade_assign))
```

Live LOM enum for `MixerDevice.crossfade_assign`:

| value | meaning |
|-------|---------|
| `0`   | **A** — track follows the crossfader's A side |
| `1`   | **None** — always audible, ignores the crossfader (Live's default) |
| `2`   | **B** — track follows the crossfader's B side |

These match `dance/als/writer.py`'s `CrossFadeState/Manual` values (`_CROSSFADE_A="0"`, `_CROSSFADE_NONE="1"`, `_CROSSFADE_B="2"`) exactly, so a live-loaded deck routes identically to a static `.als` export. The bridge assigns A-side decks → A, B-side decks → B, and the mix references → None when it creates the deck columns (`AbletonBridge._crossfade_assign_for`, called in `_create_deck_columns`).

As with the `create_audio_clip` patch, **fully quit and reopen Ableton** after editing — toggling the Control Surface isn't enough. The handlers then bind to `/live/track/set/crossfade_assign (track_idx, value)` and `/live/track/get/crossfade_assign (track_idx)`.

> Solo state needs **no** fork patch: `solo` is already in the upstream `properties_rw` list, so `/live/track/start_listen/solo`, `/live/track/get/solo`, and `/live/track/stop_listen/solo` work out of the box. The bridge subscribes to each deck column's solo to surface `AbletonState.soloed_kinds` over the WebSocket. See `docs/proposals/live-contract-additions.md`.

## "Load to Live" — how it works now

`POST /api/v1/ableton/load-track` is the one-click flow:

1. Reads `num_tracks` to know where new tracks will land.
2. **Lazily creates 5 reusable "Deck" tracks** (Mix / Drums / Bass / Vocals / Other) once per process lifetime — `/live/song/create_audio_track` + `set_track_name` + `set_track_color`. Subsequent loads reuse those columns instead of appending new ones, so 10 songs = 5 columns + 10 scenes, not 50 tracks.
3. **Auto-loads each stem** via the patched `/live/clip_slot/create_audio_clip` (track-idx, scene-idx, absolute path). Mix is intentionally left empty — the 4 stems sum to the full mix, so loading Mix too would double the audio.
4. Names each clip slot via `/live/clip/set/name`.
5. Pops a status-bar message ("Loaded {title} → scene N").
6. Records `(scene_index → dance_track_id)` in the bridge so `GET /api/v1/ableton/decks` can render the companion's Scene Map widget.

`stems_loaded` in the response (0–4) tells the companion whether to fall back to Finder-reveal (incomplete load → user drags the missing stems manually) or just show "fire the scene to play" (full load).

## Out of scope

- Routing audio between Live and the dance backend — Live IS the audio engine; the backend never touches audio playback.
- MIDI controller mapping — your Launchpad / knob controller goes directly into Live, not through this pipeline.
