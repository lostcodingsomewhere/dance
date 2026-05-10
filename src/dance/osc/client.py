"""OSC client — sends commands to AbletonOSC.

AbletonOSC routes Live API calls through OSC addresses like
``/live/song/start_playing`` and ``/live/clip_slot/fire``. The full address
map: https://github.com/ideoforms/AbletonOSC

This wrapper exposes typed methods for the operations we actually use, so
callers don't pass raw OSC strings.
"""

from __future__ import annotations

import logging
from typing import Any

from pythonosc import udp_client

logger = logging.getLogger(__name__)


# Default AbletonOSC ports
ABLETON_RECEIVE_PORT = 11000  # Live listens here (our outgoing)
ABLETON_SEND_PORT = 11001     # Live sends here (our incoming)


class AbletonOSCClient:
    """Send-only OSC client for AbletonOSC.

    All methods are fire-and-forget over UDP. AbletonOSC will respond on a
    separate port — see :class:`AbletonOSCListener` for the receive side.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = ABLETON_RECEIVE_PORT) -> None:
        self.host = host
        self.port = port
        self._client = udp_client.SimpleUDPClient(host, port)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def play(self) -> None:
        self._send("/live/song/start_playing")

    def stop(self) -> None:
        self._send("/live/song/stop_playing")

    def continue_playing(self) -> None:
        self._send("/live/song/continue_playing")

    def set_tempo(self, bpm: float) -> None:
        self._send("/live/song/set/tempo", bpm)

    # ------------------------------------------------------------------
    # Clip slots (track_index, scene_index are 0-based)
    # ------------------------------------------------------------------

    def fire_clip(self, track: int, scene: int) -> None:
        """Trigger the clip in (track, scene)."""
        self._send("/live/clip_slot/fire", track, scene)

    def stop_clip(self, track: int, scene: int) -> None:
        self._send("/live/clip_slot/stop", track, scene)

    def stop_track(self, track: int) -> None:
        self._send("/live/track/stop_all_clips", track)

    # ------------------------------------------------------------------
    # Mixer
    # ------------------------------------------------------------------

    def set_track_volume(self, track: int, volume: float) -> None:
        """Set track volume. 0.0 = -inf dB, 0.85 = 0 dB, 1.0 = +6 dB."""
        self._send("/live/track/set/volume", track, volume)

    def set_track_panning(self, track: int, panning: float) -> None:
        """Pan: -1.0 (left) to +1.0 (right)."""
        self._send("/live/track/set/panning", track, panning)

    def set_track_send(self, track: int, send_index: int, level: float) -> None:
        self._send("/live/track/set/send", track, send_index, level)

    def set_track_mute(self, track: int, muted: bool) -> None:
        self._send("/live/track/set/mute", track, 1 if muted else 0)

    def set_track_solo(self, track: int, soloed: bool) -> None:
        self._send("/live/track/set/solo", track, 1 if soloed else 0)

    # ------------------------------------------------------------------
    # Subscriptions — ask AbletonOSC to push state changes
    # ------------------------------------------------------------------

    def start_listen_tempo(self) -> None:
        self._send("/live/song/start_listen/tempo")

    def start_listen_beat(self) -> None:
        self._send("/live/song/start_listen/beat")

    def start_listen_playing_clip(self, track: int) -> None:
        self._send("/live/track/start_listen/playing_slot_index", track)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send(self, address: str, *args: Any) -> None:
        logger.debug("OSC → %s %s", address, args)
        self._client.send_message(address, list(args) if args else [])
