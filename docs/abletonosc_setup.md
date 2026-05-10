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

- **Reads:** tempo, beat position, currently playing clip per track, track volume, song num_tracks.
- **Sends:** transport (play/stop), clip launch/stop, tempo set, track volume/pan/mute/solo, track name/color, create/delete audio track, create/delete clip, set clip warp/loop/color/name, status-bar message.

See `src/dance/osc/client.py` for the full method list and `src/dance/osc/bridge.py` for how state pushes flow into the FastAPI WebSocket layer.

## "Push to Live" — what works and what doesn't

`POST /api/v1/ableton/load-track` is the one-click "send this track + stems to Live" endpoint. It uses every OSC primitive AbletonOSC exposes, but it stops short of fully loading the audio because **AbletonOSC has no command for loading a sample file into a clip slot**.

Specifically:
- `/live/clip_slot/create_clip` only creates an **empty MIDI** clip — it has a `length` parameter but no `file_path`.
- `/live/clip/get/file_path` is read-only; there is no matching `/set/file_path`.
- There is no `/live/clip_slot/load_file`, `/live/song/load_sample`, or browser-drop command.
- Live's underlying Python Remote Scripts API does not expose `ClipSlot.create_audio_clip(path)` either — this is a Live limitation, not just an AbletonOSC gap.

So `push_track_to_live` does the next best thing:

1. Reads `num_tracks` from Live to learn where new tracks will land.
2. Calls `/live/song/create_audio_track` once for the full mix and once per stem (drums/bass/vocals/other).
3. Renames each track via `/live/track/set/name` (e.g. `"My Song — Drums"`).
4. Color-codes each track via `/live/track/set/color` so the four stems are visually distinct.
5. Pops a status-bar message in Live ("drag the stems onto scene 1…").
6. Returns the assigned track indices to the caller. The React UI then reveals the stems folder in Finder so the user can drag the files onto the prepared slots.

If/when AbletonOSC adds a sample-loading command, the bridge can be upgraded to a true one-click flow without changing the API contract.

## Out of scope

- Routing audio between Live and the dance backend — Live IS the audio engine; the backend never touches audio playback.
- MIDI controller mapping — your Launchpad / knob controller goes directly into Live, not through this pipeline.
