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

    def fire_scene(self, scene: int) -> None:
        """Trigger every clip on scene ``scene`` simultaneously (anchor mode).

        Plays the full original combination for a row of staged stems. Used
        when the user wants to play a track as-recorded instead of remixing.
        """
        self._send("/live/scene/fire", scene)

    def stop_all_clips(self) -> None:
        """Stop every playing clip without halting the transport.

        Clears the combo; the master clock keeps running so the next fired
        clip syncs cleanly. ``stop()`` (transport off) is the harder kill.
        """
        self._send("/live/song/stop_all_clips")

    def set_record_mode(self, on: bool) -> None:
        """Toggle Live's session record. When on, fired clips that have
        an armed track record into the next empty slot. We use it as a
        plain "capture this take" affordance — UI guards live in the API
        layer.
        """
        self._send("/live/song/set/record_mode", 1 if on else 0)

    # Master crossfader: -1 (full A) ... 0 (center) ... +1 (full B). Lives
    # on Song.master_track.mixer_device.crossfader. Driven primarily by the
    # APC40's hardware crossfader (when tracks are assigned to A/B groups
    # via their CrossFadeAssignment); we expose the read+write so the FE
    # can mirror the value on-screen between Deck A and Deck B.
    def get_crossfader(self) -> None:
        self._send("/live/song/get/crossfader")

    def set_crossfader(self, value: float) -> None:
        self._send("/live/song/set/crossfader", float(value))

    def start_listen_crossfader(self) -> None:
        self._send("/live/song/start_listen/crossfader")

    def stop_listen_crossfader(self) -> None:
        self._send("/live/song/stop_listen/crossfader")

    # Solo/Cue mode: Live has a global toggle (master strip "Solo/Cue"
    # button) that switches every track's S button between two behaviors:
    #   0 = Solo (master mutes everything except the soloed track)
    #   1 = Cue  (soloed track routes to the Cue output / outs 3/4 — PFL)
    # Stem-DJing needs Cue. We set it on bridge init so the per-deck PFL
    # buttons in the UI Just Work without the user toggling it themselves.
    def set_solo_cue_mode(self, cue: bool) -> None:
        self._send("/live/song/set/solo_cue_mode", 1 if cue else 0)

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

    def set_track_output_routing_type(self, track: int, type_str: str) -> None:
        """Set a track's output routing type (e.g. ``"Ext. Out"``, ``"Master"``).

        Used to route the Cue deck track to outs 3/4 instead of Master so
        previews play through headphones without leaking to the speakers.
        Available values come from Live's Preferences → Audio → Output Config.
        """
        self._send("/live/track/set/output_routing_type", track, type_str)

    def set_track_output_routing_channel(self, track: int, channel: str) -> None:
        """Set a track's output routing channel (e.g. ``"3/4"``, ``"1/2"``).

        Only meaningful when output routing type is ``"Ext. Out"``. With the
        Scarlett 4i4, ``"3/4"`` is the dedicated cue bus.
        """
        self._send("/live/track/set/output_routing_channel", track, channel)

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

    def create_audio_clip(self, track: int, slot: int, path: str) -> None:
        """Load an audio sample from disk into the given clip slot.

        Live 12.0.5+. Requires our local AbletonOSC fork — see
        ``docs/abletonosc_setup.md`` (we patched ``clip_slot.py`` to register
        ``/live/clip_slot/create_audio_clip``). Errors silently in Live's log
        if the path doesn't exist, the track is frozen, the track is not an
        audio track, or the slot is recording.
        """
        self._send("/live/clip_slot/create_audio_clip", track, slot, path)

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

    def get_clip_name(self, track: int, slot: int) -> None:
        """Ask Live to push the clip's name back via /live/clip/get/name.
        Reply args: (track, slot, name) or no reply if the slot is empty."""
        self._send("/live/clip/get/name", track, slot)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_num_tracks(self) -> None:
        """Ask Live to push the current track count to the listener port."""
        self._send("/live/song/get/num_tracks")

    def get_track_names(self) -> None:
        """Ask Live to push every track's name in one reply.

        Reply addressed to ``/live/song/get/track_names``; payload is a tuple
        of strings in track-index order. AbletonOSC also supports a range
        ``(min, max)`` but we always want all of them.
        """
        self._send("/live/song/get/track_names")

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

    def start_listen_is_playing(self) -> None:
        self._send("/live/song/start_listen/is_playing")

    def start_listen_playing_clip(self, track: int) -> None:
        self._send("/live/track/start_listen/playing_slot_index", track)

    def start_listen_track_meter(self, track: int) -> None:
        """Subscribe to a track's output meter level. AbletonOSC pushes
        ``/live/track/get/output_meter_level`` replies at Live's meter rate
        (~30 Hz); we throttle FE re-renders separately."""
        self._send("/live/track/start_listen/output_meter_level", track)

    def stop_listen_track_meter(self, track: int) -> None:
        self._send("/live/track/stop_listen/output_meter_level", track)

    def start_listen_clip_position(self, track: int, slot: int) -> None:
        """Subscribe to a clip's playing_position (beat-accurate playhead).
        Reply: ``/live/clip/get/playing_position`` (track, slot, beats)."""
        self._send("/live/clip/start_listen/playing_position", track, slot)

    def stop_listen_clip_position(self, track: int, slot: int) -> None:
        self._send("/live/clip/stop_listen/playing_position", track, slot)

    def set_clip_loop_start(self, track: int, slot: int, beats: float) -> None:
        """Move just the loop_start of a clip (in beats). Used by scrub —
        we set start_marker AND loop_start together so the loop wraps to
        the clicked position rather than resetting to 0."""
        self._send("/live/clip/set/loop_start", track, slot, beats)

    def set_clip_start_marker(self, track: int, slot: int, beats: float) -> None:
        """Set where the clip's playback begins on next fire. Takes effect
        only after the clip is re-fired; doesn't seek a currently-playing
        clip. ``beats`` is floating-point beats from the clip's origin."""
        self._send("/live/clip/set/start_marker", track, slot, beats)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send(self, address: str, *args: Any) -> None:
        logger.debug("OSC → %s %s", address, args)
        self._client.send_message(address, list(args) if args else [])
