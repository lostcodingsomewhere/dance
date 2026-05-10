"""High-level Ableton bridge: combines OSC client + listener and maintains
the latest observed state. This is what the FastAPI backend talks to.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dance.osc.client import (
    ABLETON_RECEIVE_PORT,
    ABLETON_SEND_PORT,
    AbletonOSCClient,
)
from dance.osc.listener import AbletonOSCListener

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from dance.core.database import StemFile, Track

logger = logging.getLogger(__name__)


@dataclass
class AbletonState:
    """Snapshot of the most recent observed Live state."""

    tempo: float | None = None
    is_playing: bool | None = None
    beat: float | None = None
    # track_index -> playing scene_index (or -1 if no clip playing)
    playing_clips: dict[int, int] = field(default_factory=dict)
    # track_index -> volume 0-1
    track_volumes: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo": self.tempo,
            "is_playing": self.is_playing,
            "beat": self.beat,
            "playing_clips": dict(self.playing_clips),
            "track_volumes": dict(self.track_volumes),
        }


# Subscriber for state-change events.
StateListener = Callable[[AbletonState], None]


class AbletonBridge:
    """One-stop wrapper: sends commands AND tracks state pushed by AbletonOSC.

    Usage::

        bridge = AbletonBridge()
        bridge.start()              # spins up listener thread
        bridge.client.play()
        snapshot = bridge.state.to_dict()
        bridge.subscribe(lambda s: print(s.tempo))
        bridge.stop()

    Designed for one instance per FastAPI process. Not thread-safe across
    multiple writers to ``state``; the listener thread updates it and readers
    just snapshot via ``to_dict()``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        send_port: int = ABLETON_RECEIVE_PORT,
        listen_port: int = ABLETON_SEND_PORT,
    ) -> None:
        self.client = AbletonOSCClient(host=host, port=send_port)
        self.listener = AbletonOSCListener(host=host, port=listen_port)
        self.state = AbletonState()
        self._subscribers: list[StateListener] = []
        self._lock = threading.Lock()

        # Request/reply scratchpad: handlers stash results here, callers
        # ``threading.Event``-wait for them. Keyed by OSC reply address.
        self._reply_events: dict[str, threading.Event] = {}
        self._reply_values: dict[str, Any] = {}

        # Wire incoming OSC → state updates.
        self.listener.on("/live/song/get/tempo", self._on_tempo)
        self.listener.on("/live/song/get/beat", self._on_beat)
        self.listener.on("/live/song/get/is_playing", self._on_is_playing)
        self.listener.on(
            "/live/track/get/playing_slot_index", self._on_playing_clip
        )
        self.listener.on("/live/track/get/volume", self._on_track_volume)
        self.listener.on("/live/song/get/num_tracks", self._on_num_tracks)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.listener.start()
        # Ask AbletonOSC to start pushing the things we care about.
        try:
            self.client.start_listen_tempo()
            self.client.start_listen_beat()
        except OSError as exc:
            # Live isn't listening; that's fine in dev/test.
            logger.info("Could not subscribe to Live (%s) — continuing without push state", exc)

    def stop(self) -> None:
        self.listener.stop()

    # ------------------------------------------------------------------
    # Subscriptions for downstream consumers (e.g., the WebSocket layer)
    # ------------------------------------------------------------------

    def subscribe(self, listener: StateListener) -> None:
        with self._lock:
            self._subscribers.append(listener)

    def _broadcast(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub(self.state)
            except Exception:  # noqa: BLE001
                logger.exception("State subscriber crashed")

    # ------------------------------------------------------------------
    # OSC → state handlers (run on listener thread)
    # ------------------------------------------------------------------

    def _on_tempo(self, _address: str, args: tuple[Any, ...]) -> None:
        if args:
            self.state.tempo = float(args[0])
            self._broadcast()

    def _on_beat(self, _address: str, args: tuple[Any, ...]) -> None:
        if args:
            self.state.beat = float(args[0])
            self._broadcast()

    def _on_is_playing(self, _address: str, args: tuple[Any, ...]) -> None:
        if args:
            self.state.is_playing = bool(args[0])
            self._broadcast()

    def _on_playing_clip(self, _address: str, args: tuple[Any, ...]) -> None:
        # AbletonOSC sends (track_index, scene_index)
        if len(args) >= 2:
            track, scene = int(args[0]), int(args[1])
            self.state.playing_clips[track] = scene
            self._broadcast()

    def _on_track_volume(self, _address: str, args: tuple[Any, ...]) -> None:
        if len(args) >= 2:
            track, vol = int(args[0]), float(args[1])
            self.state.track_volumes[track] = vol
            self._broadcast()

    def _on_num_tracks(self, address: str, args: tuple[Any, ...]) -> None:
        if args:
            self._reply_values[address] = int(args[0])
            evt = self._reply_events.get(address)
            if evt is not None:
                evt.set()

    # ------------------------------------------------------------------
    # Request/reply helpers
    # ------------------------------------------------------------------

    def _await_reply(
        self, address: str, send: Callable[[], None], timeout: float = 0.5
    ) -> Any | None:
        """Send a query, wait for the matching reply, return its value.

        Returns ``None`` on timeout (Live not running, etc.). Designed for
        single-value replies like ``/live/song/get/num_tracks``.
        """
        evt = threading.Event()
        self._reply_events[address] = evt
        self._reply_values.pop(address, None)
        try:
            send()
            if not evt.wait(timeout):
                return None
            return self._reply_values.get(address)
        finally:
            self._reply_events.pop(address, None)
            self._reply_values.pop(address, None)

    def get_num_tracks(self, timeout: float = 0.5) -> int | None:
        """Ask Live for its current track count and wait for the reply."""
        return self._await_reply(
            "/live/song/get/num_tracks", self.client.get_num_tracks, timeout
        )

    # ------------------------------------------------------------------
    # High-level: push a track + its stems into Live
    # ------------------------------------------------------------------

    # Live's track-color palette indexes — picked to keep stems visually
    # distinct. These are RGB ints, not the 0-69 clip-color-index range.
    _STEM_TRACK_COLORS: dict[str, int] = {
        "mix":    0xFFA500,  # orange — the full mix
        "drums":  0xFF3030,  # red
        "bass":   0x9050FF,  # purple
        "vocals": 0x30B0FF,  # blue
        "other":  0x60D060,  # green
    }
    _STEM_ORDER: tuple[str, ...] = ("drums", "bass", "vocals", "other")

    def push_track_to_live(
        self,
        track: "Track",
        stems: list["StemFile"],
        *,
        include_stems: bool = True,
        scene_index: int = 0,
        num_tracks_timeout: float = 0.5,
    ) -> dict[str, Any]:
        """Insert a full-mix track (and optionally one track per stem) into Live.

        AbletonOSC has no command to *load* a sample from disk into a clip
        slot (Live's Python API doesn't expose it). So this method only does
        what OSC can: create empty, named, color-coded audio tracks at the
        end of the song, ready for the user to drag stems onto.

        Returns ``{"scene_index": int, "track_indices": {"mix": int, ...},
        "warnings": [str, ...]}``. The track indices are *expected* positions
        — appended at the end of the song — because OSC create calls are
        fire-and-forget.
        """
        warnings: list[str] = []

        # Find current track count so we know what indices our new tracks land
        # at. If Live isn't running we still proceed (fire the OSC anyway)
        # and assume 0 — callers tolerate this.
        base = self.get_num_tracks(timeout=num_tracks_timeout)
        if base is None:
            warnings.append(
                "Could not read song num_tracks from Live (timeout); "
                "indices below assume the new tracks were appended at index 0."
            )
            base = 0

        # Validate sources up front (don't crash if a path is missing).
        title = (track.title or track.file_name or f"Track {track.id}").strip()
        full_mix_path = Path(track.file_path) if track.file_path else None
        if full_mix_path is None or not full_mix_path.exists():
            warnings.append(
                f"Full-mix file missing on disk: {track.file_path!r}"
            )

        # Map stem kind -> StemFile for quick lookup.
        stems_by_kind: dict[str, StemFile] = {}
        for s in stems:
            stems_by_kind[str(s.kind).lower()] = s

        track_indices: dict[str, int] = {}
        next_index = base

        def _add_track(label: str, color: int, name: str) -> int:
            nonlocal next_index
            idx = next_index
            self.client.create_audio_track(-1)  # append
            self.client.set_track_name(idx, name)
            self.client.set_track_color(idx, color)
            next_index += 1
            track_indices[label] = idx
            return idx

        # 1. Full mix track.
        _add_track("mix", self._STEM_TRACK_COLORS["mix"], f"{title} — Mix")

        # 2. Per-stem tracks.
        if include_stems:
            for kind in self._STEM_ORDER:
                stem = stems_by_kind.get(kind)
                if stem is None:
                    warnings.append(f"No {kind} stem available for track {track.id}")
                    continue
                stem_path = Path(stem.path) if stem.path else None
                if stem_path is None or not stem_path.exists():
                    warnings.append(f"{kind} stem file missing on disk: {stem.path!r}")
                color = self._STEM_TRACK_COLORS.get(kind, 0x808080)
                _add_track(kind, color, f"{title} — {kind.capitalize()}")

        # Friendly status nudge in Live's status bar.
        try:
            self.client.show_message(
                f"Dance: {len(track_indices)} track(s) ready — drag {title} stems onto scene {scene_index + 1}"
            )
        except OSError:  # pragma: no cover - best-effort UI
            pass

        return {
            "scene_index": scene_index,
            "track_indices": track_indices,
            "warnings": warnings,
        }
