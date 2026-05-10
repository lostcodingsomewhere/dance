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

- **Reads:** tempo, beat position, currently playing clip per track, track volume.
- **Sends:** transport (play/stop), clip launch/stop, tempo set, track volume/pan/mute/solo.

See `src/dance/osc/client.py` for the full method list and `src/dance/osc/bridge.py` for how state pushes flow into the FastAPI WebSocket layer.

## Out of scope

- Routing audio between Live and the dance backend — Live IS the audio engine; the backend never touches audio playback.
- MIDI controller mapping — your Launchpad / knob controller goes directly into Live, not through this pipeline.
