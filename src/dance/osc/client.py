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

    def set_track_name(self, track: int, name: str) -> None:
        """Rename a track in Live's session view."""
        self._send("/live/track/set/name", track, name)

    def set_track_color(self, track: int, color: int) -> None:
        """Set a track's color via Live's palette (32-bit RGB int)."""
        self._send("/live/track/set/color", track, color)

    # ------------------------------------------------------------------
    # Song-level track/scene management
    #
    # AbletonOSC exposes ``/live/song/create_audio_track`` and friends. Pass
    # ``index = -1`` to append (the AbletonOSC default). Note: AbletonOSC
    # does *not* expose a programmatic "load sample into clip slot" command —
    # Live's Python API doesn't support it. The best we can do over OSC is
    # prepare empty named/colored audio tracks; the user still drags samples
    # from Finder. See ``docs/abletonosc_setup.md`` for details.
    # ------------------------------------------------------------------

    def create_audio_track(self, index: int = -1) -> None:
        """Insert a new audio track at ``index`` (default: append)."""
        self._send("/live/song/create_audio_track", index)

    def delete_track(self, index: int) -> None:
        """Delete the track at ``index``."""
        self._send("/live/song/delete_track", index)

    def create_scene(self, index: int = -1) -> None:
        """Insert a new scene at ``index`` (default: append)."""
        self._send("/live/song/create_scene", index)

    # ------------------------------------------------------------------
    # Clip slot / clip — what AbletonOSC *does* support.
    #
    # ``create_clip`` creates an EMPTY MIDI clip — there is no
    # ``load_sample`` equivalent for audio clips in the OSC API. The
    # setters below operate on a clip that already exists in the slot
    # (e.g. one the user dragged in).
    # ------------------------------------------------------------------

    def create_clip(self, track: int, slot: int, length: float) -> None:
        """Create an empty (MIDI) clip of ``length`` beats."""
        self._send("/live/clip_slot/create_clip", track, slot, length)

    def delete_clip(self, track: int, slot: int) -> None:
        self._send("/live/clip_slot/delete_clip", track, slot)

    def set_clip_warp(self, track: int, slot: int, warp: bool) -> None:
        self._send("/live/clip/set/warping", track, slot, 1 if warp else 0)

    def set_clip_loop(
        self, track: int, slot: int, start_beats: float, end_beats: float
    ) -> None:
        """Set loop start + end (in beats). Two messages — Live needs both."""
        self._send("/live/clip/set/loop_start", track, slot, start_beats)
        self._send("/live/clip/set/loop_end", track, slot, end_beats)

    def set_clip_color(self, track: int, slot: int, color: int) -> None:
        """Set clip color via Live's palette index (0-69)."""
        self._send("/live/clip/set/color_index", track, slot, color)

    def set_clip_name(self, track: int, slot: int, name: str) -> None:
        self._send("/live/clip/set/name", track, slot, name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_num_tracks(self) -> None:
        """Ask Live to push the current track count to the listener port."""
        self._send("/live/song/get/num_tracks")

    def show_message(self, message: str) -> None:
        """Pop a status-bar message in Live (handy for user feedback)."""
        self._send("/live/api/show_message", message)

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
