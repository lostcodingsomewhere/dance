"""High-level Ableton bridge: combines OSC client + listener and maintains
the latest observed state. This is what the FastAPI backend talks to.
"""

from __future__ import annotations

import json
import logging
import os
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
    # track_index -> volume 0-1 (fader position, not audio level)
    track_volumes: dict[int, float] = field(default_factory=dict)
    # track_index -> output meter level 0-1 (actual audio amplitude).
    # Subscribed on deck columns so the FE can render a VU meter that
    # represents what's coming out of our live-remixing combo.
    track_meters: dict[int, float] = field(default_factory=dict)
    # track_index -> currently-playing clip's playing_position in beats.
    # Subscribed per clip when the bridge sees that clip start playing
    # (in _on_playing_clip); unsubscribed when it stops. Drives the
    # accurate playhead in the FE's MasterVisualizer + CueStrip.
    playing_positions: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo": self.tempo,
            "is_playing": self.is_playing,
            "beat": self.beat,
            "playing_clips": dict(self.playing_clips),
            "track_volumes": dict(self.track_volumes),
            "track_meters": dict(self.track_meters),
            "playing_positions": dict(self.playing_positions),
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
        state_file: Path | None = None,
    ) -> None:
        self.client = AbletonOSCClient(host=host, port=send_port)
        self.listener = AbletonOSCListener(host=host, port=listen_port)
        self.state = AbletonState()
        self._subscribers: list[StateListener] = []
        self._lock = threading.Lock()

        # Persistence path for deck columns / cells / cue-track index. Pass
        # an explicit path to enable atomic save/restore across backend
        # restarts; pass ``None`` (the test default) to disable persistence
        # entirely. Production wires this via api/app.py from
        # ``settings.data_dir / "deck_state.json"``.
        self._state_file = state_file

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

        # (scene_index, kind) -> dance Track id. One entry per loaded cell in
        # Live's session view. Cells in a single row can come from different
        # tracks (the live-remixing model); anchor mode is just the case
        # where all 4 stem cells in a row point at the same track. The API
        # exposes this map as the cell-level deck view.
        self._deck_cells: dict[tuple[int, str], int] = {}

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
        self.listener.on(
            "/live/track/get/output_meter_level", self._on_track_meter
        )
        self.listener.on(
            "/live/clip/get/playing_position", self._on_playing_position
        )
        self.listener.on("/live/song/get/num_tracks", self._on_num_tracks)
        self.listener.on("/live/song/get/track_names", self._on_track_names)
        self.listener.on("/live/clip/get/name", self._on_clip_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.listener.start()
        # First, restore from disk: deck columns + cells + cue-track from
        # the last session. Lets the FE render the correct grid immediately
        # on backend restart, before Live has a chance to confirm.
        self._load_state()
        # Ask AbletonOSC to start pushing the things we care about.
        try:
            self.client.start_listen_tempo()
            self.client.start_listen_beat()
            self.client.start_listen_is_playing()
        except OSError as exc:
            # Live isn't listening; that's fine in dev/test.
            logger.info("Could not subscribe to Live (%s) — continuing without push state", exc)
        # Best-effort adopt existing Deck columns so a backend restart
        # doesn't create duplicates in Live. If the persisted columns and
        # Live's actual track-name layout disagree, Live wins. Silent on
        # timeout — Live may not be running yet.
        try:
            recovered = self.recover_deck_columns(timeout=1.0)
            if recovered is not None:
                logger.info("Adopted existing deck columns: %s", recovered)
            # Also subscribe meters + playing-clip on whatever columns we
            # have (recovered OR persisted-from-disk) so the FE state
            # populates immediately without a fresh load.
            if self._deck_columns:
                self._subscribe_deck_columns(self._deck_columns)
        except Exception:  # noqa: BLE001 — never let recovery crash boot
            logger.exception("Deck-column recovery failed")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Atomically write deck columns/cells/cue-track to disk. Called on
        every mutation so a backend restart can reconstruct the grid.
        Best-effort: write failures are logged but don't crash.
        No-op when state_file is None (test default)."""
        if self._state_file is None:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "deck_columns": self._deck_columns,
                # Tuples aren't JSON; serialize as list-of-triples.
                "deck_cells": [
                    [s, k, tid] for (s, k), tid in self._deck_cells.items()
                ],
                "cue_track_idx": self._cue_track_idx,
            }
            tmp = self._state_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, self._state_file)
        except OSError as exc:  # pragma: no cover - best-effort
            logger.warning("Could not persist bridge state: %s", exc)

    def _load_state(self) -> None:
        """Restore deck columns/cells/cue-track from disk. Best-effort: on
        any error (missing file, malformed JSON, schema drift) we just keep
        the defaults and move on — the bridge will rebuild as the user
        loads tracks. No-op when state_file is None (test default)."""
        if self._state_file is None:
            return
        try:
            raw = self._state_file.read_text()
        except OSError:
            return
        try:
            data = json.loads(raw)
            cols = data.get("deck_columns")
            if isinstance(cols, dict):
                self._deck_columns = {str(k): int(v) for k, v in cols.items()}
            cells = data.get("deck_cells", [])
            if isinstance(cells, list):
                self._deck_cells = {
                    (int(s), str(k)): int(tid) for s, k, tid in cells
                }
            cue = data.get("cue_track_idx")
            if isinstance(cue, int):
                self._cue_track_idx = cue
            logger.info(
                "Restored bridge state: %d columns, %d cells",
                len(self._deck_columns or {}),
                len(self._deck_cells),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Could not parse persisted bridge state: %s", exc)

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
        # AbletonOSC sends (track_index, scene_index). scene_index == -1 is
        # the "no clip playing on this track" signal — clear it from state
        # so the FE's ``playing_clips[trackIdx] != null`` checks read as
        # "stopped" rather than "playing scene -1".
        #
        # Also manages playing_position subscriptions: subscribe when a clip
        # starts, unsubscribe when it stops or is replaced. Keeps OSC
        # traffic scoped to what's actually firing.
        if len(args) >= 2:
            track, scene = int(args[0]), int(args[1])
            prev_scene = self.state.playing_clips.get(track)
            if scene < 0:
                self.state.playing_clips.pop(track, None)
                self.state.playing_positions.pop(track, None)
                if prev_scene is not None and prev_scene >= 0:
                    try:
                        self.client.stop_listen_clip_position(track, prev_scene)
                    except OSError:  # pragma: no cover - best-effort
                        pass
            else:
                self.state.playing_clips[track] = scene
                # Replaced the clip on this track? Drop the old subscription.
                if prev_scene is not None and prev_scene != scene:
                    try:
                        self.client.stop_listen_clip_position(track, prev_scene)
                    except OSError:  # pragma: no cover
                        pass
                try:
                    self.client.start_listen_clip_position(track, scene)
                except OSError:  # pragma: no cover - best-effort
                    pass
            self._broadcast()

    def _on_playing_position(self, _address: str, args: tuple[Any, ...]) -> None:
        # AbletonOSC sends (track, slot, position_beats). Live throttles to
        # the project's transport rate (~30 Hz). We just store per-track and
        # broadcast.
        if len(args) >= 3:
            track = int(args[0])
            pos = float(args[2])
            self.state.playing_positions[track] = pos
            self._broadcast()

    def _on_track_volume(self, _address: str, args: tuple[Any, ...]) -> None:
        if len(args) >= 2:
            track, vol = int(args[0]), float(args[1])
            self.state.track_volumes[track] = vol
            self._broadcast()

    def _on_track_meter(self, _address: str, args: tuple[Any, ...]) -> None:
        if len(args) >= 2:
            track, level = int(args[0]), float(args[1])
            self.state.track_meters[track] = level
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

    def _on_clip_name(self, address: str, args: tuple[Any, ...]) -> None:
        # AbletonOSC sends (track, slot, name) for /live/clip/get/name. Used
        # by scan_live_for_cells() to adopt clips that were already in Live
        # before the bridge knew about them.
        if len(args) >= 3:
            self._reply_values[address] = (
                int(args[0]),
                int(args[1]),
                str(args[2]) if args[2] is not None else "",
            )
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
            self._subscribe_deck_columns(recovered)
            self._persist_state()
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
        self._deck_cells = {}
        self._cue_track_idx = None
        self._persist_state()
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
        """Forget the cached deck-column indices AND the per-cell load map.

        The *next* push_track_to_live will (re)create the Deck tracks. Does
        **not** delete anything in Live — the user is in charge of their
        session view.
        """
        self._deck_columns = None
        self._deck_cells = {}
        self._persist_state()

    def get_deck_state(self) -> dict[str, Any]:
        """Snapshot of which Ableton tracks are our deck columns and which
        cells (scene × kind) are loaded with which dance-track ids. The API
        surfaces this so the FE can render the SceneGrid + ComboStrip in
        sync with the bridge's view of Live.

        ``cells`` is a list of ``{"scene_index": int, "kind": str, "track_id": int}``
        rows — one per loaded cell.
        """
        cells = [
            {"scene_index": s, "kind": k, "track_id": tid}
            for (s, k), tid in sorted(self._deck_cells.items())
        ]
        return {
            "columns": dict(self._deck_columns) if self._deck_columns else None,
            "cells": cells,
        }

    def scan_live_for_cells(
        self,
        *,
        num_slots: int = 16,
        timeout_per_slot: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Walk each deck-column track's first ``num_slots`` clip slots and
        return what's there. Sequential one-shot OSC queries to
        ``/live/clip/get/name``; empty slots return no reply (we time out
        quickly and move on).

        Returns a list of ``{"scene_index": int, "kind": str, "clip_name":
        str}`` for every populated cell. Caller is responsible for matching
        the clip names back to dance Track ids (we set names to
        ``"{title} ({kind})"`` when loading).

        Used by the resync endpoint to adopt clips that were placed in
        Live before the bridge knew about them — e.g. after a backend
        restart with persistence missing.
        """
        if self._deck_columns is None:
            return []
        found: list[dict[str, Any]] = []
        for kind, track_idx in self._deck_columns.items():
            for slot in range(num_slots):
                name = self._get_clip_name(
                    track_idx, slot, timeout=timeout_per_slot
                )
                if name:
                    found.append(
                        {"scene_index": slot, "kind": kind, "clip_name": name}
                    )
        return found

    def _get_clip_name(
        self, track: int, slot: int, *, timeout: float = 0.15
    ) -> str | None:
        """One-shot clip-name query. Returns None on timeout / empty slot.

        Verifies the reply matches the requested (track, slot) so a delayed
        reply for an earlier slot doesn't bleed into the next query.
        """
        addr = "/live/clip/get/name"
        # Fresh event so we don't accidentally observe a stale reply.
        evt = threading.Event()
        self._reply_events[addr] = evt
        self._reply_values.pop(addr, None)
        try:
            self.client.get_clip_name(track, slot)
        except OSError:
            return None
        if not evt.wait(timeout):
            return None
        result = self._reply_values.get(addr)
        if not isinstance(result, tuple) or len(result) < 3:
            return None
        reply_track, reply_slot, reply_name = result
        if reply_track != track or reply_slot != slot:
            return None
        return reply_name or None

    def adopt_cells(self, cells: dict[tuple[int, str], int]) -> None:
        """Replace ``_deck_cells`` with the given (scene, kind) → track_id map
        and persist. Caller pre-resolves track_ids via DB lookup; we just
        store + persist."""
        self._deck_cells = dict(cells)
        self._persist_state()

    def next_free_slot(self, kind: str) -> int:
        """Lowest scene index where the given kind's cell is empty."""
        used = {s for (s, k) in self._deck_cells if k == kind}
        i = 0
        while i in used:
            i += 1
        return i

    def delete_cell(self, track_index: int, slot_index: int) -> dict[str, Any]:
        """Stop + delete a single clip slot in one of the deck columns.

        - Looks up ``kind`` from ``_deck_columns`` so we can drop the matching
          entry from ``_deck_cells`` (the bridge's in-memory deck map).
        - Stops the clip first (no-op if already stopped), then deletes the
          clip from the slot via the OSC primitive — leaving the *slot* in
          place (an empty slot, ready for the next ``Load to Live``).
        - Persists the updated ``_deck_cells`` so a bridge restart doesn't
          resurrect the dead cell.

        Safe on slots that don't belong to a deck column (returns ``ok:
        False`` with a warning rather than blowing up) — keeps the API
        idempotent for retries.
        """
        if self._deck_columns is None:
            return {"ok": False, "warning": "no deck columns staged yet"}
        # Reverse-lookup the kind from the deck columns map.
        kind: str | None = None
        for k, idx in self._deck_columns.items():
            if idx == track_index:
                kind = k
                break
        if kind is None:
            return {
                "ok": False,
                "warning": f"track {track_index} is not a deck column",
            }
        try:
            self.client.stop_clip(track_index, slot_index)
        except OSError:  # pragma: no cover - best-effort
            pass
        try:
            self.client.delete_clip(track_index, slot_index)
        except OSError as exc:  # pragma: no cover - best-effort
            logger.warning("delete_clip(%d, %d) failed: %s", track_index, slot_index, exc)
            return {"ok": False, "warning": str(exc)}
        # Drop from our deck-cells cache + persist. Use pop with default so
        # we don't KeyError when the cache was already empty for that slot
        # (e.g. user clicked X on a stale UI).
        removed = self._deck_cells.pop((slot_index, kind), None)
        self._persist_state()
        return {
            "ok": True,
            "track_index": track_index,
            "slot_index": slot_index,
            "kind": kind,
            "removed_track_id": removed,
        }

    def stop_scene(self, scene_index: int) -> dict[str, Any]:
        """Stop every deck-column cell on a scene. Live has no 'stop scene'
        primitive, so we iterate the 5 deck-column tracks and stop_clip on
        each at the given scene index. Idempotent on cells that aren't
        playing (no-op at the OSC layer)."""
        if self._deck_columns is None:
            return {"ok": False, "warning": "no deck columns staged yet"}
        for track_idx in self._deck_columns.values():
            try:
                self.client.stop_clip(track_idx, scene_index)
            except OSError:  # pragma: no cover - best-effort
                pass
        return {"ok": True, "scene_index": scene_index}

    def next_free_row(self) -> int:
        """Lowest scene index where ALL stem kinds are empty.

        Used as the default load slot for whole-song (anchor) loads so that a
        fresh row is reserved for the 4-stem combo. Excludes the ``mix`` kind
        from the emptiness check: full-song loads populate mix alongside the
        stems (so checking stems alone is equivalent), and single-stem loads
        leave mix empty by design (so including it would conflate "no anchor
        yet" with "this row is free").
        """
        kinds = ("drums", "bass", "vocals", "other")
        i = 0
        while True:
            if all((i, k) not in self._deck_cells for k in kinds):
                return i
            i += 1

    def push_track_to_live(
        self,
        track: "Track",
        stems: list["StemFile"],
        *,
        scene_index: int | None = None,
        kinds: list[str] | None = None,
        num_tracks_timeout: float = 0.5,
        # `include_stems` accepted for backward-compat with the old API; when
        # False we still default ``kinds`` to ``[]`` so nothing loads.
        include_stems: bool = True,
    ) -> dict[str, Any]:
        """Stage some or all of a track's stems on a scene in Live's session view.

        ``kinds`` controls which stems are loaded:
        - ``None`` → all 4 stems (drums/bass/vocals/other) into one row
          (anchor / whole-song mode).
        - A list (e.g. ``["drums"]``) → only those stems load (live-remixing
          single-cell mode).

        ``scene_index`` controls where they land:
        - ``None`` → the bridge picks: ``next_free_row()`` for full-song
          loads, ``next_free_slot(kind)`` for single-stem loads.
        - An int → use that exact scene (caller's choice; overwrites any
          existing cell at that intersection).

        Returns ``{"scene_index": int, "track_indices": {kind: idx, ...},
        "stems_loaded": int, "warnings": [str, ...]}``. ``track_indices``
        always carries the full deck-column map; ``stems_loaded`` is how
        many cells this call actually populated.
        """
        warnings: list[str] = []

        # Honor the backward-compat include_stems=False shape.
        if kinds is None and not include_stems:
            kinds = []
        # Default: whole-song load.
        full_song = kinds is None
        if kinds is None:
            kinds = ["drums", "bass", "vocals", "other"]
        # Filter to known stem kinds. Mix isn't loaded (it's the original
        # full-track audio and we never duplicate it into the deck row).
        valid_kinds = [k for k in kinds if k in ("drums", "bass", "vocals", "other")]

        base = self.get_num_tracks(timeout=num_tracks_timeout)
        live_reachable = base is not None
        if not live_reachable:
            warnings.append(
                "Could not read song num_tracks from Live (timeout); "
                "deck-column indices below are best-effort."
            )

        # (Re)create the 5 deck columns if we don't have them, or — only when
        # Live is reachable — if the user has deleted tracks in Live so our
        # cached indices no longer fit.
        if self._deck_columns is None:
            self._deck_columns = self._create_deck_columns(
                start_index=base if live_reachable else 0
            )
        elif live_reachable:
            cached_max = max(self._deck_columns.values())
            assert base is not None  # narrowed by live_reachable
            if cached_max >= base:
                self._deck_columns = self._create_deck_columns(start_index=base)

        deck_columns: dict[str, int] = self._deck_columns

        # Resolve the scene_index now that we know which kinds we're loading.
        if scene_index is None:
            if full_song:
                scene_index = self.next_free_row()
            elif valid_kinds:
                # Single-stem load — drop into the lowest slot in this kind's
                # column. When multiple kinds were requested we use the
                # first kind's column to anchor the slot choice.
                scene_index = self.next_free_slot(valid_kinds[0])
            else:
                scene_index = 0

        # Validate sources for this load.
        title = (track.title or track.file_name or f"Track {track.id}").strip()
        stems_by_kind = {str(s.kind).lower(): s for s in stems}
        for kind in valid_kinds:
            stem = stems_by_kind.get(kind)
            if stem is None:
                warnings.append(f"No {kind} stem available for track {track.id}")
                continue
            stem_path = Path(stem.path) if stem.path else None
            if stem_path is None or not stem_path.exists():
                warnings.append(f"{kind} stem file missing on disk: {stem.path!r}")

        # Auto-load each requested stem into the matching deck column on the
        # chosen scene. The mix cell is also populated on full-song loads —
        # the file goes in muted (the mix *track* is muted at creation, see
        # _create_deck_columns) so it doesn't double the summed stems, but
        # the DJ can unmute it to A/B against the original or fall back to
        # it if a stem is glitchy/missing.
        stems_loaded = 0
        for kind in valid_kinds:
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
                self._deck_cells[(scene_index, kind)] = track.id
                stems_loaded += 1
            except OSError as exc:  # pragma: no cover - best-effort
                warnings.append(f"OSC send for {kind} failed: {exc}")

        # Whole-song loads also drop the original mix file into the SONG
        # cell. Without this, the SceneGrid renders the SONG cell as empty
        # after a load — looks broken, doesn't match the offline .als writer
        # which has done the same thing for offline-generated sets all along
        # (see als/writer.py — `muted=(entry.kind == "mix")`).
        if full_song and track.file_path:
            mix_path = Path(track.file_path)
            if mix_path.exists():
                mix_idx = deck_columns["mix"]
                try:
                    self.client.create_audio_clip(mix_idx, scene_index, str(mix_path))
                    self.client.set_clip_name(mix_idx, scene_index, f"{title} (mix)")
                    self._deck_cells[(scene_index, "mix")] = track.id
                except OSError as exc:  # pragma: no cover - best-effort
                    warnings.append(f"OSC send for mix failed: {exc}")
            else:
                warnings.append(f"mix file missing on disk: {track.file_path!r}")

        try:
            self.client.show_message(
                f"Dance: {title} → scene {scene_index + 1} "
                f"({stems_loaded} cell{'s' if stems_loaded != 1 else ''} loaded)"
            )
        except OSError:  # pragma: no cover - best-effort UI
            pass

        # Persist after every load so a backend restart can reconstruct
        # the grid without losing what's in Live.
        self._persist_state()

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

        The mix track is muted at creation: it holds the original full-track
        recording as a reference / parachute, and would double the audio if
        unmuted alongside the summed stems. The DJ unmutes it explicitly
        when they want to A/B against the original or fall back to it.
        """
        columns: dict[str, int] = {}
        idx = start_index
        for kind in self._DECK_KINDS:
            self.client.create_audio_track(-1)
            self.client.set_track_name(idx, self._DECK_DISPLAY_NAMES[kind])
            self.client.set_track_color(idx, self._STEM_TRACK_COLORS[kind])
            if kind == "mix":
                try:
                    self.client.set_track_mute(idx, True)
                except OSError:  # pragma: no cover - best-effort
                    pass
            columns[kind] = idx
            idx += 1
        self._subscribe_deck_columns(columns)
        return columns

    def _subscribe_deck_columns(self, columns: dict[str, int]) -> None:
        """Ask AbletonOSC to push meter + playing-clip updates for each deck
        column. Without the playing-clip subscription AbletonState.playing_clips
        never populates, so the FE thinks nothing is firing even when audio
        is flowing — that's why a fresh backend boot would render "silent"
        across the MasterVisualizer + SceneGrid despite the VU bouncing.

        Idempotent on Live's side — re-subscribing is a no-op. Best-effort:
        if OSC isn't reachable we just won't get push updates; FE renders
        the empty state until subscriptions land.
        """
        for track_idx in columns.values():
            try:
                self.client.start_listen_track_meter(track_idx)
            except OSError:  # pragma: no cover - best-effort
                pass
            try:
                self.client.start_listen_playing_clip(track_idx)
            except OSError:  # pragma: no cover - best-effort
                pass

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
        self._persist_state()
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
