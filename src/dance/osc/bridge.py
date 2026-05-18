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

        # Indices of the 5 reusable "Deck" tracks in Live (mix, drums, bass,
        # vocals, other). ``None`` means we haven't created them yet; populated
        # on the first ``push_track_to_live`` call, then reused for every
        # subsequent load. Survives in-memory for the bridge's lifetime but
        # not across process restarts — that's fine; on restart we just
        # re-create the deck tracks.
        self._deck_columns: dict[str, int] | None = None

        # scene_index -> dance Track id. Populated by push_track_to_live for
        # every (scene, track) we've staged in Live's session view. The API
        # exposes this map so the React companion can render a "scene map"
        # without having to query Live directly.
        self._deck_scenes: dict[int, int] = {}

        # Index of the dedicated "Cue" track in Live — output routed to the
        # Scarlett 4i4's outs 3/4 so previews play in headphones without
        # leaking to the master speakers. Lazy-created on first preview call.
        # See preview_audio() / stop_preview().
        self._cue_track_idx: int | None = None

        # Wire incoming OSC → state updates.
        self.listener.on("/live/song/get/tempo", self._on_tempo)
        self.listener.on("/live/song/get/beat", self._on_beat)
        self.listener.on("/live/song/get/is_playing", self._on_is_playing)
        self.listener.on(
            "/live/track/get/playing_slot_index", self._on_playing_clip
        )
        self.listener.on("/live/track/get/volume", self._on_track_volume)
        self.listener.on("/live/song/get/num_tracks", self._on_num_tracks)
        self.listener.on("/live/song/get/track_names", self._on_track_names)

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
        # Best-effort adopt existing Deck columns so a backend restart
        # doesn't create duplicates in Live. Silent on timeout — Live may
        # not be running yet.
        try:
            recovered = self.recover_deck_columns(timeout=1.0)
            if recovered is not None:
                logger.info("Adopted existing deck columns: %s", recovered)
        except Exception:  # noqa: BLE001 — never let recovery crash boot
            logger.exception("Deck-column recovery failed")

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

    def _on_track_names(self, address: str, args: tuple[Any, ...]) -> None:
        # AbletonOSC sends one variadic-string reply with N names.
        self._reply_values[address] = [str(a) for a in args]
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

    def get_track_names(self, timeout: float = 1.0) -> list[str] | None:
        """Ask Live for every track's name in one OSC roundtrip. Returns the
        names in track-index order, or ``None`` on timeout (Live not
        running)."""
        return self._await_reply(
            "/live/song/get/track_names", self.client.get_track_names, timeout
        )

    def recover_deck_columns(self, timeout: float = 1.0) -> dict[str, int] | None:
        """Scan Live's current track names for our deck columns
        (``Deck Mix`` / ``Deck Drums`` / …) and populate the bridge cache
        from them. This is what lets a backend restart pick up where it
        left off without creating duplicate deck columns in Live.

        Returns the recovered ``_deck_columns`` map or ``None`` if Live
        isn't reachable. Also recovers ``_cue_track_idx`` if a track named
        "Cue" is present.
        """
        names = self.get_track_names(timeout=timeout)
        if names is None:
            return None
        recovered: dict[str, int] = {}
        for kind, expected in self._DECK_DISPLAY_NAMES.items():
            for idx, name in enumerate(names):
                if name == expected and kind not in recovered:
                    recovered[kind] = idx
                    break
        # Adopt the Cue track too — partial recovery is fine here since the
        # Cue track is independent of the deck columns.
        for idx, name in enumerate(names):
            if name == self._CUE_DISPLAY_NAME:
                self._cue_track_idx = idx
                break
        # Only adopt deck columns if we found ALL 5; partial recoveries (user
        # renamed one) are confusing — better to create a fresh set.
        if len(recovered) == len(self._DECK_DISPLAY_NAMES):
            self._deck_columns = recovered
            return recovered
        return None

    def clean_live_decks(self, timeout: float = 1.0) -> dict[str, Any]:
        """Delete every ``Deck *`` track AND the Cue track in Live (covers
        stragglers from previous backend runs) and reset the bridge cache.
        Returns a summary of what was removed.

        Deletes in reverse-index order so the upstream indices don't shift
        out from under us mid-iteration.
        """
        names = self.get_track_names(timeout=timeout)
        if names is None:
            return {"deleted": 0, "warning": "Live unreachable; nothing deleted."}
        expected = set(self._DECK_DISPLAY_NAMES.values()) | {self._CUE_DISPLAY_NAME}
        deck_indices = [i for i, n in enumerate(names) if n in expected]
        for idx in sorted(deck_indices, reverse=True):
            try:
                self.client.delete_track(idx)
            except OSError as exc:  # pragma: no cover - best-effort
                logger.warning("delete_track(%d) failed: %s", idx, exc)
        self._deck_columns = None
        self._deck_scenes = {}
        self._cue_track_idx = None
        return {"deleted": len(deck_indices), "indices": sorted(deck_indices)}

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
    # Order matters — defines column layout in Live.
    _DECK_KINDS: tuple[str, ...] = ("mix", "drums", "bass", "vocals", "other")
    _DECK_DISPLAY_NAMES: dict[str, str] = {
        "mix":    "Deck Mix",
        "drums":  "Deck Drums",
        "bass":   "Deck Bass",
        "vocals": "Deck Vocals",
        "other":  "Deck Other",
    }

    # Dedicated Cue track — output routed to outs 3/4 (the Scarlett 4i4's
    # cue bus). Created next to the deck columns; always in this exact
    # configuration so previews never leak to master.
    _CUE_DISPLAY_NAME: str = "Cue"
    _CUE_COLOR: int = 0xFFE066  # warm yellow — visually distinct from decks
    _CUE_OUTPUT_TYPE: str = "Ext. Out"
    _CUE_OUTPUT_CHANNEL: str = "3/4"
    # Slot inside the Cue track used for all preview clips. We always
    # delete + recreate so anchor-mode fire_scene calls never accidentally
    # re-trigger a stale preview.
    _CUE_SLOT: int = 0

    def reset_deck_columns(self) -> None:
        """Forget the cached deck-column indices AND the scene→track map.

        The *next* push_track_to_live will (re)create the Deck tracks. Does
        **not** delete anything in Live — the user is in charge of their
        session view.
        """
        self._deck_columns = None
        self._deck_scenes = {}

    def get_deck_state(self) -> dict[str, Any]:
        """Snapshot of which Ableton tracks are our deck columns and which
        scenes are staged with our songs. The API surfaces this so the FE
        can render a "scene map" widget that stays in sync with the bridge's
        view of Live."""
        return {
            "columns": dict(self._deck_columns) if self._deck_columns else None,
            "scenes": dict(self._deck_scenes),
        }

    def push_track_to_live(
        self,
        track: "Track",
        stems: list["StemFile"],
        *,
        scene_index: int = 0,
        num_tracks_timeout: float = 0.5,
        # `include_stems` is accepted for backward-compat with the old API but
        # ignored — the deck-column layout always reserves all 5 channels.
        include_stems: bool = True,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Stage a track on a specific scene in Live's session view.

        Live's Python API doesn't expose "load sample into clip slot", so the
        actual drop is done by the user (we open Finder to the stems folder).
        What OSC *can* do is prepare the destination:

          - Maintain 5 reusable audio tracks in Live, named ``Deck Mix /
            Drums / Bass / Vocals / Other`` and colored per stem.
          - Each "load" call claims **one scene** of those 5 columns. Songs
            stack vertically: song A → scene 0, song B → scene 1, etc. The
            APC40's scene-launch buttons map directly to switching between
            them.

        The 5 columns are created lazily on the first call and reused after
        that, so loading 10 songs uses 5 tracks and 10 scenes, not 50 tracks.

        Returns ``{"scene_index": int, "track_indices": {"mix": idx, ...},
        "warnings": [str, ...]}``. ``track_indices`` is the deck-column map
        (same for every call once the columns exist).
        """
        warnings: list[str] = []

        base = self.get_num_tracks(timeout=num_tracks_timeout)
        live_reachable = base is not None
        if not live_reachable:
            warnings.append(
                "Could not read song num_tracks from Live (timeout); "
                "deck-column indices below are best-effort."
            )

        # (Re)create the 5 deck columns if we don't have them, or — only when
        # Live is reachable — if the user has deleted tracks in Live so our
        # cached indices no longer fit. We don't invalidate on unreachable
        # Live, otherwise repeated load-track calls during a dev cycle (or
        # while Live is closed) would loop forever creating phantom columns
        # at index 0.
        if self._deck_columns is None:
            self._deck_columns = self._create_deck_columns(
                start_index=base if live_reachable else 0
            )
        elif live_reachable:
            cached_max = max(self._deck_columns.values())
            assert base is not None  # narrowed by live_reachable
            if cached_max >= base:
                self._deck_columns = self._create_deck_columns(start_index=base)

        # We just guaranteed self._deck_columns is non-None above.
        deck_columns: dict[str, int] = self._deck_columns

        # Validate sources for this load.
        title = (track.title or track.file_name or f"Track {track.id}").strip()
        full_mix_path = Path(track.file_path) if track.file_path else None
        if full_mix_path is None or not full_mix_path.exists():
            warnings.append(
                f"Full-mix file missing on disk: {track.file_path!r}"
            )
        stems_by_kind = {str(s.kind).lower(): s for s in stems}
        for kind in ("drums", "bass", "vocals", "other"):
            stem = stems_by_kind.get(kind)
            if stem is None:
                warnings.append(f"No {kind} stem available for track {track.id}")
                continue
            stem_path = Path(stem.path) if stem.path else None
            if stem_path is None or not stem_path.exists():
                warnings.append(f"{kind} stem file missing on disk: {stem.path!r}")

        # Auto-load each stem into the matching deck column on this scene.
        # Live 12.0.5+ exposes ClipSlot.create_audio_clip(path) and our
        # patched AbletonOSC binds it to /live/clip_slot/create_audio_clip.
        # The "mix" column is intentionally left empty — summing stems
        # already produces the full mix, so loading it too would double the
        # audio. Users who want the full-mix file can drop it manually.
        stems_loaded = 0
        for kind in ("drums", "bass", "vocals", "other"):
            stem = stems_by_kind.get(kind)
            if stem is None or not stem.path:
                continue
            stem_path = Path(stem.path)
            if not stem_path.exists():
                continue
            t_idx = deck_columns[kind]
            try:
                self.client.create_audio_clip(t_idx, scene_index, str(stem_path))
                self.client.set_clip_name(t_idx, scene_index, f"{title} ({kind})")
                stems_loaded += 1
            except OSError as exc:  # pragma: no cover - best-effort
                warnings.append(f"OSC send for {kind} failed: {exc}")

        try:
            self.client.show_message(
                f"Dance: {title} → scene {scene_index + 1} "
                f"({stems_loaded}/4 stems loaded)"
            )
        except OSError:  # pragma: no cover - best-effort UI
            pass

        # Remember the placement so the API can expose a scene map. If the
        # caller staged a different song on the same scene we overwrite —
        # the user explicitly replaced it.
        self._deck_scenes[scene_index] = track.id

        return {
            "scene_index": scene_index,
            "track_indices": deck_columns,
            "stems_loaded": stems_loaded,
            "warnings": warnings,
        }

    def _create_deck_columns(self, *, start_index: int) -> dict[str, int]:
        """Append 5 named, colored deck tracks to Live and return their indices.

        Indices are *predicted* (created via fire-and-forget OSC). The caller
        is responsible for using them in subsequent OSC calls in the same
        order, which is safe because AbletonOSC processes commands serially.
        """
        columns: dict[str, int] = {}
        idx = start_index
        for kind in self._DECK_KINDS:
            self.client.create_audio_track(-1)
            self.client.set_track_name(idx, self._DECK_DISPLAY_NAMES[kind])
            self.client.set_track_color(idx, self._STEM_TRACK_COLORS[kind])
            columns[kind] = idx
            idx += 1
        return columns

    # ------------------------------------------------------------------
    # Cue / preview — audition a candidate clip in headphones without
    # leaking to master. Requires the Scarlett 4i4 (or similar 4-out
    # interface) with outs 3/4 enabled in Live's Output Config.
    # ------------------------------------------------------------------

    def _ensure_cue_track(self, *, num_tracks_timeout: float = 0.5) -> int:
        """Return the Cue track's index, creating + routing it if needed.

        The Cue track lives next to the deck columns. Its output is set to
        ``Ext. Out → 3/4`` (the 4i4's cue bus). Once created, it's reused
        for the life of the bridge.
        """
        if self._cue_track_idx is not None:
            return self._cue_track_idx

        # Try to adopt an existing "Cue" track first.
        recovered = self.recover_deck_columns(timeout=num_tracks_timeout)
        if self._cue_track_idx is not None:
            return self._cue_track_idx

        # Otherwise create it. Predict its index from the current track
        # count (same trick _create_deck_columns uses).
        base = self.get_num_tracks(timeout=num_tracks_timeout)
        if base is None:
            # Live unreachable — best-effort prediction so we don't loop on
            # a closed Live. Caller will see preview fail.
            base = 0
        if recovered is not None:
            # Deck columns adopted; Cue appends after them.
            base = max(max(self._deck_columns.values()) + 1, base) if self._deck_columns else base
        idx = base
        self.client.create_audio_track(-1)
        self.client.set_track_name(idx, self._CUE_DISPLAY_NAME)
        self.client.set_track_color(idx, self._CUE_COLOR)
        self.client.set_track_output_routing_type(idx, self._CUE_OUTPUT_TYPE)
        self.client.set_track_output_routing_channel(idx, self._CUE_OUTPUT_CHANNEL)
        self._cue_track_idx = idx
        return idx

    def preview_audio(
        self,
        audio_path: str,
        *,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Audition ``audio_path`` through the Cue track (headphones only).

        Stops any in-progress preview first, then drops the new clip into
        the Cue track's slot and fires it. The clip is replaced on every
        call (rather than reused) so the user can rapidly cycle through
        previews without accumulated state.

        Returns ``{"ok": bool, "cue_track_idx": int, "slot": int, "audio_path": str,
        "label": str | None, "warnings": [str, ...]}``.
        """
        warnings: list[str] = []
        path = Path(audio_path)
        if not path.exists():
            return {
                "ok": False,
                "cue_track_idx": self._cue_track_idx,
                "slot": self._CUE_SLOT,
                "audio_path": str(path),
                "label": label,
                "warnings": [f"Audio file not found: {audio_path!r}"],
            }

        cue_idx = self._ensure_cue_track()

        # Stop + clear any prior preview so the new one starts cleanly.
        try:
            self.client.stop_clip(cue_idx, self._CUE_SLOT)
        except OSError:  # pragma: no cover - best-effort
            pass
        try:
            self.client.delete_clip(cue_idx, self._CUE_SLOT)
        except OSError:  # pragma: no cover - best-effort
            pass

        try:
            self.client.create_audio_clip(cue_idx, self._CUE_SLOT, str(path))
            if label:
                self.client.set_clip_name(cue_idx, self._CUE_SLOT, label)
            self.client.fire_clip(cue_idx, self._CUE_SLOT)
        except OSError as exc:
            warnings.append(f"OSC send failed: {exc}")
            return {
                "ok": False,
                "cue_track_idx": cue_idx,
                "slot": self._CUE_SLOT,
                "audio_path": str(path),
                "label": label,
                "warnings": warnings,
            }

        return {
            "ok": True,
            "cue_track_idx": cue_idx,
            "slot": self._CUE_SLOT,
            "audio_path": str(path),
            "label": label,
            "warnings": warnings,
        }

    def stop_preview(self) -> dict[str, Any]:
        """Stop the current preview and clear the Cue track's slot.

        Idempotent — safe to call when nothing is previewing. Deleting the
        clip (not just stopping) prevents an accidental ``fire_scene(0)``
        from re-triggering a stale preview during anchor mode.
        """
        if self._cue_track_idx is None:
            return {"ok": True, "cleared": False}
        try:
            self.client.stop_clip(self._cue_track_idx, self._CUE_SLOT)
        except OSError:  # pragma: no cover - best-effort
            pass
        try:
            self.client.delete_clip(self._cue_track_idx, self._CUE_SLOT)
        except OSError:  # pragma: no cover - best-effort
            pass
        return {"ok": True, "cleared": True}
