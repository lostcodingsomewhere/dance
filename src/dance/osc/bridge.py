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
    # Master crossfader position. Live's range is -1 (full A) to +1 (full B),
    # with 0 = center. We subscribe on bridge init and push updates to the FE
    # so the on-screen crossfader bar follows the APC40 hardware fader live.
    crossfader: float | None = None
    # Deck-kinds whose Live track currently has Solo engaged. With Solo/Cue
    # mode = Cue (set on bridge init), a soloed track is routed to the
    # headphone PFL bus (outs 3/4). Derived from per-deck solo subscriptions;
    # lets the FE light up the real PFL state instead of guessing from the
    # last /pfl/{side} call. Sorted in _DECK_KINDS order for stable output.
    soloed_kinds: list[str] = field(default_factory=list)
    # Monotonic counter that increments on EVERY mutation of the loaded
    # deck-cell map (load / clear / move / anchor-fill / resync). The FE
    # watches this and refetches GET /ableton/decks the instant it changes,
    # instead of polling on a timer.
    deck_map_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo": self.tempo,
            "is_playing": self.is_playing,
            "beat": self.beat,
            "playing_clips": dict(self.playing_clips),
            "track_volumes": dict(self.track_volumes),
            "track_meters": dict(self.track_meters),
            "playing_positions": dict(self.playing_positions),
            "crossfader": self.crossfader,
            "soloed_kinds": list(self.soloed_kinds),
            "deck_map_revision": self.deck_map_revision,
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

        # FX state — populated by ``_discover_fx_tracks`` on first push and
        # whenever the user re-syncs. Lazy because the FX tracks live in
        # the .als template, not in code, and only exist after the user
        # has authored them per docs/proposals/fx-phase-1-runbook.md.
        #
        # ``_fx_return_idx[name]`` → Live track index of an FX return
        # (e.g. "Filter" → 12). Used by per-deck filter toggle to pick the
        # right Send slot when ramping send-A levels.
        # ``_fx_scene_idx[name]`` → scene index hosting a one-shot FX clip
        # (e.g. "Riser" → 7). Fired by name via /transport/fx/{name}.
        # ``_fx_track_for_scene[name]`` → which track index hosts that
        # one-shot clip (the riser sample lives on a "FX" track or on the
        # Riser return itself, configurable per the runbook).
        self._fx_return_idx: dict[str, int] = {}
        self._fx_scene_idx: dict[str, int] = {}
        self._fx_track_for_scene: dict[str, int] = {}
        # Per-deck FX on/off state. Cheap UI mirror so toggle buttons can
        # render their active state without round-tripping every render.
        # Authoritative state is Live's actual send levels, but those are
        # buffered by AbletonOSC and can lag.
        self._filter_active: dict[str, bool] = {"a": False, "b": False}
        self._reverb_active: dict[str, bool] = {"a": False, "b": False}
        self._delay_active: dict[str, bool] = {"a": False, "b": False}

        # In-flight ramp timers per (side, fx_name) — populated by
        # ``ramp_fx_send`` (Phase 3 continuous sweep). Keyed by tuple so
        # the same side can have a filter ramp and a reverb ramp running
        # in parallel. Cancelled when a new ramp on the same key starts
        # or when a toggle preempts it.
        self._fx_ramp_timers: dict[tuple[str, str], list[threading.Timer]] = {}

        # (scene_index, kind) -> dance Track id. One entry per loaded cell in
        # Live's session view. Cells in a single row can come from different
        # tracks (the live-remixing model); anchor mode is just the case
        # where all 4 stem cells in a row point at the same track. The API
        # exposes this map as the cell-level deck view.
        self._deck_cells: dict[tuple[int, str], int] = {}

        # track_index -> bool, the raw Solo state Live pushed for each deck
        # column. Populated by per-deck solo subscriptions (see
        # _subscribe_deck_columns). The bridge maps these track indices back
        # to deck-kinds via _deck_columns to derive AbletonState.soloed_kinds.
        # We keep the raw per-track map (not just the kind list) so an
        # un-recovered track index doesn't get silently dropped.
        self._track_solo: dict[int, bool] = {}

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
        self.listener.on("/live/song/get/crossfader", self._on_crossfader)
        self.listener.on("/live/track/get/solo", self._on_track_solo)

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
            self.client.start_listen_crossfader()
            # Set Live's master Solo/Cue mode to "Cue" so per-track Solo
            # buttons act as PFL (route to outs 3/4) instead of muting
            # master. Stem-DJing needs PFL semantics. Idempotent — Live
            # already in Cue mode is a no-op.
            self.client.set_solo_cue_mode(True)
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
            is_pre_pair_shape = False
            if isinstance(cols, dict):
                # Discard any persisted columns dict that doesn't match
                # the current 10-track shape — either pre-pair (5 cols,
                # no _a/_b) or the 9-track intermediate (single ``mix``).
                # In either case, in-place migration would leave a partial
                # layout in Live; safer to drop the cache and let the next
                # push_track_to_live rebuild from scratch.
                key_set = set(cols.keys())
                missing_pair = "drums_a" not in key_set and "drums" in key_set
                missing_split_mix = "mix" in key_set and "mix_a" not in key_set
                if missing_pair or missing_split_mix:
                    is_pre_pair_shape = True
                    self._deck_columns = None
                else:
                    self._deck_columns = {str(k): int(v) for k, v in cols.items()}
            cells = data.get("deck_cells", [])
            if isinstance(cells, list):
                self._deck_cells = {
                    (int(s), self._migrate_deck_kind(str(k))): int(tid)
                    for s, k, tid in cells
                }
            cue = data.get("cue_track_idx")
            if isinstance(cue, int):
                self._cue_track_idx = cue
            logger.info(
                "Restored bridge state: %d columns, %d cells%s",
                len(self._deck_columns or {}),
                len(self._deck_cells),
                " (pre-pair columns discarded — will rebuild)"
                    if is_pre_pair_shape else "",
            )
            # Forward-migrate any pre-current-shape state so the next
            # persist writes in the new shape and we never re-process this.
            # Triggered either by old cells (renamed by _migrate_deck_kind)
            # or by the columns-discard branch above.
            if is_pre_pair_shape or any(
                k in {"drums", "bass", "vocals", "other", "mix"}
                for _, k in self._deck_cells
            ):
                self._persist_state()  # pragma: no cover - migration edge
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Could not parse persisted bridge state: %s", exc)

    @staticmethod
    def _migrate_deck_kind(kind: str) -> str:
        """Forward-migrate pre-deck-pair deck-kind names.

        Two historical shapes get folded into the current 10-track layout:

        - Pre-pair (single-deck) ``drums`` / ``bass`` / ``vocals`` / ``other``
          → A-side (the conventional "primary" deck).
        - 9-track intermediate shape ``mix`` → ``mix_a``. (Mix split into
          per-deck references in the 10-track shape.)
        """
        if kind in {"drums", "bass", "vocals", "other"}:
            return f"{kind}_a"
        if kind == "mix":
            return "mix_a"
        return kind

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

    def _on_crossfader(self, _address: str, args: tuple[Any, ...]) -> None:
        if args:
            # Live reports -1..+1; clamp defensively.
            v = float(args[0])
            self.state.crossfader = max(-1.0, min(1.0, v))
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

    def _on_track_solo(self, _address: str, args: tuple[Any, ...]) -> None:
        # AbletonOSC sends (track_index, 0|1) for /live/track/get/solo (both
        # on the immediate snapshot at subscribe time and on every change).
        # Record the raw per-track state, then recompute the deck-kind list.
        if len(args) >= 2:
            track = int(args[0])
            soloed = bool(args[1])
            self._track_solo[track] = soloed
            self._recompute_soloed_kinds()
            self._broadcast()

    def _recompute_soloed_kinds(
        self, columns: dict[str, int] | None = None
    ) -> None:
        """Derive ``AbletonState.soloed_kinds`` from the raw per-track solo
        map + the current deck-column layout. Emits deck-kinds (e.g.
        ``"drums_a"``) in canonical ``_DECK_KINDS`` order so the output is
        stable across re-derivations.

        ``columns`` defaults to ``self._deck_columns``; callers mid-creation
        (when ``self._deck_columns`` hasn't been assigned the freshly-built
        map yet) pass the new layout explicitly.

        A track index with no deck-column mapping (e.g. the Cue track, or a
        solo push that arrived before recovery) is ignored — only deck
        columns contribute. No-op-safe when the layout is None."""
        if columns is None:
            columns = self._deck_columns or {}
        # Invert: track_index -> deck_kind.
        idx_to_kind = {idx: kind for kind, idx in columns.items()}
        soloed = {
            idx_to_kind[idx]
            for idx, on in self._track_solo.items()
            if on and idx in idx_to_kind
        }
        self.state.soloed_kinds = [k for k in self._DECK_KINDS if k in soloed]

    def _bump_deck_revision(self) -> None:
        """Increment the deck-map revision counter and broadcast.

        Called from every code path that mutates the loaded deck-cell map
        (load, clear/delete, anchor-fill, adopt/resync, reset). The FE
        watches ``AbletonState.deck_map_revision`` and refetches the deck map
        the instant it changes — no 2 s polling timer. Pairs with the
        ``_persist_state`` calls that already bracket every cell mutation."""
        self.state.deck_map_revision += 1
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
        for kind, accepted in self._DECK_RECOVERY_NAMES.items():
            for idx, name in enumerate(names):
                if name in accepted and kind not in recovered:
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
            # Columns are part of the deck map the FE renders — a recovery
            # that adopts a different layout should trigger a refetch.
            self._bump_deck_revision()
            return recovered
        return None

    # Display names from prior layouts. Kept here so ``clean_live_decks``
    # can sweep stragglers after each reshape. Covers:
    #
    #   • Original 5-deck shape (single "Deck Mix" + 4 unsplit stems).
    #   • 9-deck intermediate (A/B per stem + single "Deck Mix").
    #
    # The current 10-deck layout uses "Deck Mix A" / "Deck Mix B" in
    # ``_DECK_DISPLAY_NAMES``; those are merged via union in
    # ``clean_live_decks`` so both old and current names get nuked.
    _LEGACY_DECK_DISPLAY_NAMES: frozenset[str] = frozenset({
        # Pre-pair (5-track) layout
        "Deck Mix",
        "Deck Drums",
        "Deck Bass",
        "Deck Vocals",
        "Deck Other",
        # 9-track intermediate also had "Deck Mix" (single); included above.
    })

    def clean_live_decks(self, timeout: float = 1.0) -> dict[str, Any]:
        """Delete every ``Deck *`` track AND the Cue track in Live (covers
        stragglers from previous backend runs and the pre-deck-pair
        layout) and reset the bridge cache. Returns a summary of what was
        removed.

        Deletes in reverse-index order so the upstream indices don't shift
        out from under us mid-iteration.
        """
        names = self.get_track_names(timeout=timeout)
        if names is None:
            return {"deleted": 0, "warning": "Live unreachable; nothing deleted."}
        expected = (
            set(self._DECK_DISPLAY_NAMES.values())
            | self._LEGACY_DECK_DISPLAY_NAMES
            | {self._CUE_DISPLAY_NAME}
        )
        deck_indices = [i for i, n in enumerate(names) if n in expected]
        for idx in sorted(deck_indices, reverse=True):
            try:
                self.client.delete_track(idx)
            except OSError as exc:  # pragma: no cover - best-effort
                logger.warning("delete_track(%d) failed: %s", idx, exc)
        self._deck_columns = None
        self._deck_cells = {}
        self._cue_track_idx = None
        self._track_solo = {}
        self._recompute_soloed_kinds()
        self._persist_state()
        self._bump_deck_revision()
        return {"deleted": len(deck_indices), "indices": sorted(deck_indices)}

    # ------------------------------------------------------------------
    # High-level: push a track + its stems into Live
    # ------------------------------------------------------------------

    # Live's track-color palette indexes — picked to keep stems visually
    # distinct. A-side is the bright color, B-side a darker variant of the
    # same hue so the eye groups them as "the same role." MIX is orange,
    # darker for B-side to match the convention.
    # These are RGB ints, not the 0-69 clip-color-index range.
    _STEM_TRACK_COLORS: dict[str, int] = {
        "drums_a":  0xFF3030,  # red
        "drums_b":  0xA01818,  # dark red
        "bass_a":   0x9050FF,  # purple
        "bass_b":   0x583090,  # dark purple
        "vocals_a": 0x30B0FF,  # blue
        "vocals_b": 0x186080,  # dark blue
        "other_a":  0x60D060,  # green
        "other_b":  0x357835,  # dark green
        "mix_a":    0xFFA500,  # orange — A-deck mix reference
        "mix_b":    0xA06800,  # dark orange — B-deck mix reference
    }
    # Order matters — defines column layout in Live. The 8 stem decks come
    # FIRST so APC40's default 8-strip view maps to them one-to-one (one
    # hardware fader per stem deck). Mix tracks live at indices 8-9,
    # reachable via APC40 bank-shift; the Cue track (created later) sits
    # at index 10. See docs/proposals/two-deck-ui-rethink.md.
    _DECK_KINDS: tuple[str, ...] = (
        "drums_a", "drums_b",
        "bass_a", "bass_b",
        "vocals_a", "vocals_b",
        "other_a", "other_b",
        "mix_a", "mix_b",
    )
    # Source stem kinds — what Demucs emits, what the pipeline operates on,
    # what callers pass to push_track_to_live. The bridge maps each to its
    # A or B deck side at load time based on row availability.
    _SOURCE_STEM_KINDS: tuple[str, ...] = ("drums", "bass", "vocals", "other")
    _DECK_DISPLAY_NAMES: dict[str, str] = {
        "drums_a":  "Deck Drums A",
        "drums_b":  "Deck Drums B",
        "bass_a":   "Deck Bass A",
        "bass_b":   "Deck Bass B",
        "vocals_a": "Deck Vocals A",
        "vocals_b": "Deck Vocals B",
        "other_a":  "Deck Other A",
        "other_b":  "Deck Other B",
        "mix_a":    "Deck Mix A",
        "mix_b":    "Deck Mix B",
    }

    # Names accepted when ADOPTING existing Live tracks during recovery.
    # The canonical name we CREATE (via push_track_to_live → set_track_name)
    # is the "Deck …"-prefixed form above. But the static .als writer
    # (dance/als/writer.py:_display_for) emits the bare "Drums A" form, so
    # opening an exported Set yields tracks recovery must also adopt —
    # otherwise the deck grid never populates after `dance export-als`.
    # We only WIDEN what we accept; everything we create stays prefixed.
    # (Explicit literal, not a class-body comprehension — those can't see
    # other class vars like _DECK_DISPLAY_NAMES.)
    _DECK_RECOVERY_NAMES: dict[str, frozenset[str]] = {
        "drums_a": frozenset({"Deck Drums A", "Drums A"}),
        "drums_b": frozenset({"Deck Drums B", "Drums B"}),
        "bass_a": frozenset({"Deck Bass A", "Bass A"}),
        "bass_b": frozenset({"Deck Bass B", "Bass B"}),
        "vocals_a": frozenset({"Deck Vocals A", "Vocals A"}),
        "vocals_b": frozenset({"Deck Vocals B", "Vocals B"}),
        "other_a": frozenset({"Deck Other A", "Other A"}),
        "other_b": frozenset({"Deck Other B", "Other B"}),
        "mix_a": frozenset({"Deck Mix A", "Mix A"}),
        "mix_b": frozenset({"Deck Mix B", "Mix B"}),
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
        self._recompute_soloed_kinds()
        self._persist_state()
        self._bump_deck_revision()

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
        self._bump_deck_revision()

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
        self._bump_deck_revision()
        return {
            "ok": True,
            "track_index": track_index,
            "slot_index": slot_index,
            "kind": kind,
            "removed_track_id": removed,
        }

    def stop_scene(self, scene_index: int) -> dict[str, Any]:
        """Stop every deck-column cell on a scene. Live has no 'stop scene'
        primitive, so we iterate all deck-column tracks and stop_clip on
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
        """Lowest scene index where ALL 8 stem deck channels are empty.

        Used as the default load slot for whole-song (anchor) loads so that
        a fresh row is reserved. Excludes ``mix`` from the emptiness check:
        full-song loads populate mix alongside the stems (so checking stems
        is equivalent), and single-stem loads leave mix empty by design.
        Checks both A and B sides — a row counts as occupied if EITHER side
        has any stem in it.
        """
        deck_stem_kinds = tuple(
            f"{src}_{side}" for src in self._SOURCE_STEM_KINDS for side in ("a", "b")
        )
        i = 0
        while True:
            if all((i, k) not in self._deck_cells for k in deck_stem_kinds):
                return i
            i += 1

    def fire_deck(self, side: str, scene_index: int) -> dict[str, Any]:
        """Fire every cell on a deck side at one scene — the per-deck
        "play" button. Fires all 5 of the side's tracks (4 stems + mix)
        at the given scene; quiet on empty slots (Live ignores them).

        ``side`` must be ``"a"`` or ``"b"``. Returns
        ``{"ok": bool, "side": str, "scene_index": int, "fired": int}``.
        """
        if side not in ("a", "b"):
            return {"ok": False, "warning": f"side must be 'a' or 'b', got {side!r}"}
        if self._deck_columns is None:
            return {"ok": False, "warning": "no deck columns staged yet"}
        fired = 0
        for kind, idx in self._deck_columns.items():
            if not kind.endswith(f"_{side}"):
                continue
            try:
                self.client.fire_clip(idx, scene_index)
                fired += 1
            except OSError:  # pragma: no cover - best-effort
                pass
        return {"ok": True, "side": side, "scene_index": scene_index, "fired": fired}

    def stop_deck(self, side: str) -> dict[str, Any]:
        """Stop every clip on a deck side — the per-deck "stop" button.

        Iterates the side's 5 tracks and stops_all_clips on each, so any
        firing OR queued clip is halted. Master transport keeps running.
        """
        if side not in ("a", "b"):
            return {"ok": False, "warning": f"side must be 'a' or 'b', got {side!r}"}
        if self._deck_columns is None:
            return {"ok": False, "warning": "no deck columns staged yet"}
        stopped = 0
        for kind, idx in self._deck_columns.items():
            if not kind.endswith(f"_{side}"):
                continue
            try:
                self.client.stop_track(idx)
                stopped += 1
            except OSError:  # pragma: no cover - best-effort
                pass
        return {"ok": True, "side": side, "stopped": stopped}

    def set_pfl_side(self, side: str | None) -> dict[str, Any]:
        """Route an entire deck side (A or B) to the headphone cue bus.

        Uses Live's per-track Solo with Solo/Cue mode set to ``Cue`` (we
        flip the mode on bridge init, see ``start``). When a track is
        soloed in Cue mode, its audio routes to the Cue output (outs
        3/4 → headphones) AND continues to play through master — exactly
        the PFL behavior a DJ wants.

        ``side`` ∈ {``"a"``, ``"b"``, ``None``}. ``None`` clears PFL on
        both sides (solo-off all stem decks + both mix decks).

        Affects only the 8 stem decks and the 2 mix decks — leaves the
        Cue track + any non-deck tracks untouched.

        Returns ``{"ok": True, "side": side, "tracks_affected": int}``.
        """
        if side is not None and side not in ("a", "b"):
            return {
                "ok": False,
                "warning": f"side must be 'a' or 'b' or None, got {side!r}",
            }
        if self._deck_columns is None:
            return {"ok": False, "warning": "no deck columns staged yet"}
        affected = 0
        for kind, idx in self._deck_columns.items():
            # Soloed iff the kind is on the requested side. Mixes go
            # along with their side's stems — full PFL of the deck.
            should_solo = side is not None and kind.endswith(f"_{side}")
            try:
                self.client.set_track_solo(idx, should_solo)
                affected += 1
            except OSError:  # pragma: no cover - best-effort
                pass
        return {"ok": True, "side": side, "tracks_affected": affected}

    # ------------------------------------------------------------------
    # FX returns — Filter (Phase 1), Riser one-shot (Phase 2).
    # Authoring lives in the .als template (the user adds the return
    # tracks + clip slots per docs/proposals/fx-phase-1-runbook.md);
    # the bridge discovers them by name at runtime. Missing FX = no-op
    # with a warning, so the rest of the rig keeps working.
    # ------------------------------------------------------------------

    # Track names the bridge looks for during _discover_fx_tracks.
    # Returns are matched as Live tracks whose name is EXACTLY one of:
    _FX_RETURN_NAMES: tuple[str, ...] = ("Filter", "Reverb", "Delay", "Riser")
    # One-shot FX clip locations: clip name → expected track-name +
    # scene-index hint. Bridge will scan track names to find the track,
    # then look up the clip slot by clip name.
    _FX_CLIP_NAMES: tuple[str, ...] = ("Riser",)

    def discover_fx_tracks(self, *, timeout: float = 1.0) -> dict[str, Any]:
        """Scan Live's track names + scene names to map our FX clips and
        return tracks. Idempotent — called on bridge start and whenever
        the user wires `/decks/resync`.

        Returns ``{"returns": {...}, "scenes": {...}}`` for telemetry.
        Tolerant of missing FX — the rig works without them; just the
        FX buttons no-op.
        """
        names = self.get_track_names(timeout=timeout)
        if names is None:
            return {"warning": "Live unreachable; FX not discovered"}
        # Returns: match by exact track name.
        for i, n in enumerate(names):
            if n in self._FX_RETURN_NAMES:
                self._fx_return_idx[n.lower()] = i
        # One-shot scenes: scan clip names in each FX return + a dedicated
        # "FX" track if present. We rely on naming the clips like
        # "Riser" / "Reverb Hit" / "Delay Throw" so the bridge knows
        # which FX a fired scene represents.
        for fx_name in self._FX_CLIP_NAMES:
            # Look for a clip on the matching return track first; fall
            # back to a generic "FX" track if the user prefers that
            # layout.
            for candidate_track_name in (fx_name, "FX"):
                for i, n in enumerate(names):
                    if n != candidate_track_name:
                        continue
                    slot = self._find_clip_slot(i, fx_name, timeout=timeout)
                    if slot is not None:
                        self._fx_scene_idx[fx_name.lower()] = slot
                        self._fx_track_for_scene[fx_name.lower()] = i
                        break
                if fx_name.lower() in self._fx_scene_idx:
                    break
        return {
            "returns": dict(self._fx_return_idx),
            "scenes": dict(self._fx_scene_idx),
        }

    def _find_clip_slot(
        self, track_idx: int, clip_name: str, *, timeout: float = 0.3
    ) -> int | None:
        """Walk a track's clip slots looking for a named clip. Returns
        the scene index, or None if not found within ``max_scenes``.
        Reuses the existing synchronous _get_clip_name helper used by
        the resync flow."""
        max_scenes = 16
        target = clip_name.strip().lower()
        for s in range(max_scenes):
            name = self._get_clip_name(track_idx, s, timeout=timeout)
            if name and name.strip().lower() == target:
                return s
        return None

    def _send_index_for_fx(self, return_name: str) -> int | None:
        """Resolve which Live Send slot index corresponds to an FX return
        track. Live's send indices are positional (Send 0 = first return,
        Send 1 = second, ...), so we sort the discovered FX returns by
        track index and use that order.
        """
        key = return_name.lower()
        if key not in self._fx_return_idx:
            return None
        return_indices = sorted(self._fx_return_idx.values())
        try:
            return return_indices.index(self._fx_return_idx[key])
        except ValueError:  # pragma: no cover
            return None

    def _set_deck_send(self, side: str, send_index: int, level: float) -> int:
        """Write a send level to every track on a deck side (4 stems + mix).
        Returns count of tracks affected. Best-effort on OSC failures."""
        if self._deck_columns is None:
            return 0
        affected = 0
        for kind, idx in self._deck_columns.items():
            if not kind.endswith(f"_{side}"):
                continue
            try:
                self.client.set_track_send(idx, send_index, level)
                affected += 1
            except OSError:  # pragma: no cover - best-effort
                pass
        return affected

    def _toggle_fx_send(
        self, side: str, return_name: str, state_key: str
    ) -> dict[str, Any]:
        """Generic per-deck FX send toggle, shared by Filter, Reverb, and
        Delay. ``state_key`` indexes into the dict that tracks ON/OFF UI
        state per FX type per side (e.g. ``_filter_active``).
        """
        if side not in ("a", "b"):
            return {"ok": False, "warning": f"side must be 'a' or 'b', got {side!r}"}
        if return_name.lower() not in self._fx_return_idx:
            self.discover_fx_tracks(timeout=0.3)
        send_index = self._send_index_for_fx(return_name)
        if send_index is None:
            return {
                "ok": False,
                "warning": f"No {return_name!r} return track in Live. "
                "Author one per docs/proposals/fx-phase-1-runbook.md.",
            }
        state_dict = getattr(self, state_key)
        now_active = not state_dict[side]
        level = 1.0 if now_active else 0.0
        # Cancel any in-flight ramp on this side — toggles win over sweeps.
        self._cancel_filter_sweep(side, return_name)
        affected = self._set_deck_send(side, send_index, level)
        state_dict[side] = now_active
        return {
            "ok": True,
            "side": side,
            "fx": return_name.lower(),
            "active": now_active,
            "tracks_affected": affected,
        }

    def toggle_filter(self, side: str) -> dict[str, Any]:
        """Toggle the parallel filter send for a deck side. ON = all 5
        deck tracks send 100% to the Filter return (HPF in the template).
        OFF = sends back to 0. See ``_toggle_fx_send`` for the shared
        implementation."""
        return self._toggle_fx_send(side, "Filter", "_filter_active")

    def toggle_reverb(self, side: str) -> dict[str, Any]:
        """Toggle the parallel Reverb send for a deck side. Use for
        vocal throws + tail effects during transitions. Requires a
        'Reverb' return track in the .als."""
        return self._toggle_fx_send(side, "Reverb", "_reverb_active")

    def toggle_delay(self, side: str) -> dict[str, Any]:
        """Toggle the parallel Delay send for a deck side. Use for
        echo-throws on the last beat before a transition. Requires a
        'Delay' return track in the .als."""
        return self._toggle_fx_send(side, "Delay", "_delay_active")

    def fire_fx(self, name: str) -> dict[str, Any]:
        """Fire a named one-shot FX clip (e.g. ``riser``). Quantized to
        the next bar by LaunchQuantisation = 1 Bar on each clip.

        Requires the named FX clip to exist in the .als template —
        usually on a return track or a dedicated "FX" track. Without
        it, no-op with a warning.
        """
        key = name.strip().lower()
        if key not in self._fx_scene_idx or key not in self._fx_track_for_scene:
            self.discover_fx_tracks(timeout=0.3)
        if key not in self._fx_scene_idx:
            return {
                "ok": False,
                "warning": f"No FX clip named {name!r}. "
                "Author one per docs/proposals/fx-phase-1-runbook.md.",
            }
        track_idx = self._fx_track_for_scene[key]
        scene_idx = self._fx_scene_idx[key]
        try:
            self.client.fire_clip(track_idx, scene_idx)
        except OSError as exc:  # pragma: no cover
            return {"ok": False, "warning": f"OSC fire failed: {exc}"}
        return {"ok": True, "name": key, "track": track_idx, "scene": scene_idx}

    def filter_state(self) -> dict[str, bool]:
        """Snapshot of which decks currently have filter sends engaged."""
        return dict(self._filter_active)

    def fx_state(self) -> dict[str, Any]:
        """Combined snapshot of all FX toggle states + discovered tracks.
        Used by the UI to render every FX button's active state and to
        know which FX are even available (return tracks present in the
        .als)."""
        return {
            "filter": dict(self._filter_active),
            "reverb": dict(self._reverb_active),
            "delay": dict(self._delay_active),
            "returns": dict(self._fx_return_idx),
            "scenes": dict(self._fx_scene_idx),
        }

    # ------------------------------------------------------------------
    # Phase 3 — continuous FX sweep (timer-driven send ramp)
    # ------------------------------------------------------------------

    def _cancel_filter_sweep(self, side: str, return_name: str) -> None:
        """Cancel any in-flight ramp for (side, return_name). Called when
        a binary toggle preempts a running sweep, or when a new sweep
        starts on the same side+fx combo."""
        key = (side, return_name.lower())
        timers = self._fx_ramp_timers.pop(key, [])
        for t in timers:
            try:
                t.cancel()
            except Exception:  # pragma: no cover - timer is best-effort
                pass

    def ramp_fx_send(
        self,
        side: str,
        return_name: str,
        target_level: float,
        duration_bars: float,
    ) -> dict[str, Any]:
        """Smoothly ramp a deck's send level to a target over duration_bars.

        Uses threading.Timer to fire interpolated set_track_send updates
        at ~25 fps. Cancellable: a new ramp on the same (side, fx)
        replaces the in-flight one; binary toggles via ``toggle_*``
        cancel any ramp on that combo.

        ``duration_bars`` is interpreted at the current master tempo
        (4 beats per bar at the current BPM). Defaults gracefully when
        tempo isn't known.

        Returns ``{"ok": bool, "side": str, "fx": str, "target": float,
        "duration_sec": float, "steps": int}``.
        """
        if side not in ("a", "b"):
            return {"ok": False, "warning": f"side must be 'a' or 'b', got {side!r}"}
        if not 0.0 <= target_level <= 1.0:
            return {"ok": False, "warning": "target_level must be in [0, 1]"}
        if return_name.lower() not in self._fx_return_idx:
            self.discover_fx_tracks(timeout=0.3)
        send_index = self._send_index_for_fx(return_name)
        if send_index is None:
            return {
                "ok": False,
                "warning": f"No {return_name!r} return track in Live.",
            }
        # Compute interpolation grid.
        bpm = self.state.tempo or 120.0
        seconds_per_bar = (60.0 / bpm) * 4.0
        duration_sec = max(0.0, duration_bars * seconds_per_bar)
        if duration_sec <= 0.0:
            # No-duration ramp = snap. Use the toggle path semantics.
            self._set_deck_send(side, send_index, target_level)
            self._sync_state_after_ramp(side, return_name, target_level)
            return {
                "ok": True,
                "side": side,
                "fx": return_name.lower(),
                "target": target_level,
                "duration_sec": 0.0,
                "steps": 1,
            }
        # ~25 fps update rate, capped to a reasonable step count.
        steps = max(4, min(120, int(round(duration_sec * 25))))
        interval = duration_sec / steps
        # Treat current "active" boolean as the starting level — we don't
        # query Live for the actual current send level (slow + lossy).
        # If reality diverges (user nudged a send knob mid-sweep) the
        # ramp lands at target anyway.
        state_dict = self._state_dict_for(return_name)
        start_level = 1.0 if state_dict and state_dict[side] else 0.0
        delta = target_level - start_level
        self._cancel_filter_sweep(side, return_name)
        timers: list[threading.Timer] = []
        for i in range(1, steps + 1):
            t = i / steps
            # Ease-out (cubic) for filter sweeps — feels musical, accel
            # then slows. For reverb / delay throws the ear doesn't
            # notice the curve so this works there too.
            eased = 1.0 - (1.0 - t) ** 3
            level = start_level + delta * eased
            is_final_step = i == steps
            timer = threading.Timer(
                i * interval,
                self._apply_ramp_step,
                args=(side, return_name, send_index, level, is_final_step, target_level),
            )
            timer.daemon = True
            timers.append(timer)
        self._fx_ramp_timers[(side, return_name.lower())] = timers
        for t in timers:
            t.start()
        return {
            "ok": True,
            "side": side,
            "fx": return_name.lower(),
            "target": target_level,
            "duration_sec": duration_sec,
            "steps": steps,
        }

    def _apply_ramp_step(
        self,
        side: str,
        return_name: str,
        send_index: int,
        level: float,
        is_final: bool,
        final_target: float,
    ) -> None:
        """Single tick of a ramp — set the send on every deck-side track."""
        self._set_deck_send(side, send_index, level)
        if is_final:
            # Snap to exact target on final step + update UI state mirror.
            self._set_deck_send(side, send_index, final_target)
            self._sync_state_after_ramp(side, return_name, final_target)
            # Clear our timer list for this combo so a fresh ramp starts clean.
            self._fx_ramp_timers.pop((side, return_name.lower()), None)

    def _state_dict_for(self, return_name: str) -> dict[str, bool] | None:
        """Return the per-side ON/OFF mirror dict for a given FX name."""
        return {
            "filter": self._filter_active,
            "reverb": self._reverb_active,
            "delay": self._delay_active,
        }.get(return_name.lower())

    def _sync_state_after_ramp(
        self, side: str, return_name: str, target_level: float
    ) -> None:
        """Update the UI state mirror to match where a ramp landed.
        Threshold at 0.5: above = "active", below = "inactive"."""
        state_dict = self._state_dict_for(return_name)
        if state_dict is not None:
            state_dict[side] = target_level > 0.5

    def _pick_side(self, *, scene_index: int) -> str:
        """For a row, return which side ('a' or 'b') to load into.

        Prefers the side with fewer stem cells already filled at this scene.
        Tie → 'a' (matches DJ convention — A-side is the "current" deck).
        Used by ``push_track_to_live`` to keep all stems of one load on the
        same side, so anchor-detection (4-of-4 same track in a row) stays
        meaningful per side.
        """
        a_count = sum(
            1 for src in self._SOURCE_STEM_KINDS
            if (scene_index, f"{src}_a") in self._deck_cells
        )
        b_count = sum(
            1 for src in self._SOURCE_STEM_KINDS
            if (scene_index, f"{src}_b") in self._deck_cells
        )
        return "a" if a_count <= b_count else "b"

    def _free_stem_slot(self, kinds: list[str], *, forced_side: str | None) -> tuple[int, str]:
        """Resolve ``(scene_index, side)`` for a single-stem load so the
        chosen side's cell is empty for every kind in ``kinds`` — never
        clobbering a cell that's already loaded (and possibly playing).

        - ``forced_side`` ('a'/'b'): lowest scene where THAT side is free
          for all kinds. A forced load advances past an occupied cell rather
          than overwriting it (the promote-a-rec enqueue contract).
        - ``forced_side`` None (auto-pick): lowest scene where EITHER side is
          free; when both are free defer to ``_pick_side`` (less-full,
          tie → 'a'), else take whichever side is free.
        """

        def side_free(scene: int, s: str) -> bool:
            return all((scene, f"{k}_{s}") not in self._deck_cells for k in kinds)

        i = 0
        while True:
            if forced_side is not None:
                if side_free(i, forced_side):
                    return i, forced_side
            else:
                a_free = side_free(i, "a")
                b_free = side_free(i, "b")
                if a_free and b_free:
                    return i, self._pick_side(scene_index=i)
                if a_free:
                    return i, "a"
                if b_free:
                    return i, "b"
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
        # Optional deck side override. ``None`` lets _pick_side auto-pick
        # the less-full side at the target row. ``"a"`` or ``"b"`` forces
        # that side, used by the UI's explicit A/B load buttons.
        side: str | None = None,
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

        # Resolve where this load lands. All stems of one push_track_to_live
        # call go to the SAME side so anchor-detection (4-of-4 same track in
        # a row) stays meaningful per side. A forced side (UI's A/B buttons)
        # must never overwrite an occupied — possibly *playing* — cell, so
        # the scene scan honors it instead of being applied after the fact.
        forced_side = side if side in ("a", "b") else None
        if scene_index is None:
            if full_song:
                # Whole-song loads reserve a fully-empty row; _pick_side
                # (or the forced side) then chooses which deck side fills it.
                scene_index = self.next_free_row()
                side = forced_side or self._pick_side(scene_index=scene_index)
            elif valid_kinds:
                # Single-stem load: pick a scene whose target side is free
                # for these kinds so we enqueue rather than clobber.
                scene_index, side = self._free_stem_slot(valid_kinds, forced_side=forced_side)
            else:
                scene_index = 0
                side = forced_side or "a"
        else:
            # Caller pinned an exact scene — honor it even if it overwrites.
            side = forced_side or self._pick_side(scene_index=scene_index)

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
        # chosen scene. Source kinds (drums/bass/vocals/other) map to deck
        # kinds with the side suffix (drums_a or drums_b) decided above.
        # The mix cell is also populated on full-song loads — the file goes
        # in muted (the mix *track* is muted at creation, see
        # _create_deck_columns) so it doesn't double the summed stems, but
        # the DJ can unmute it to A/B against the original.
        stems_loaded = 0
        for kind in valid_kinds:
            stem = stems_by_kind.get(kind)
            if stem is None or not stem.path:
                continue
            stem_path = Path(stem.path)
            if not stem_path.exists():
                continue
            deck_kind = f"{kind}_{side}"
            t_idx = deck_columns[deck_kind]
            try:
                self.client.create_audio_clip(t_idx, scene_index, str(stem_path))
                self.client.set_clip_name(
                    t_idx, scene_index, f"{title} ({kind} {side.upper()})"
                )
                self._deck_cells[(scene_index, deck_kind)] = track.id
                stems_loaded += 1
            except OSError as exc:  # pragma: no cover - best-effort
                warnings.append(f"OSC send for {deck_kind} failed: {exc}")

        # Whole-song loads also drop the original mix file into the
        # side-matching MIX cell (mix_a or mix_b). Without this the SceneGrid
        # renders the SONG cell as empty after a load — looks broken,
        # doesn't match the offline .als writer (see als/writer.py).
        if full_song and track.file_path:
            mix_path = Path(track.file_path)
            if mix_path.exists():
                mix_kind = f"mix_{side}"
                mix_idx = deck_columns[mix_kind]
                try:
                    self.client.create_audio_clip(mix_idx, scene_index, str(mix_path))
                    self.client.set_clip_name(
                        mix_idx, scene_index, f"{title} (mix {side.upper()})"
                    )
                    self._deck_cells[(scene_index, mix_kind)] = track.id
                except OSError as exc:  # pragma: no cover - best-effort
                    warnings.append(f"OSC send for {mix_kind} failed: {exc}")
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
        # Tell the FE the deck-cell map changed so it refetches immediately.
        self._bump_deck_revision()

        return {
            "scene_index": scene_index,
            "track_indices": deck_columns,
            "stems_loaded": stems_loaded,
            "warnings": warnings,
        }

    def _create_deck_columns(self, *, start_index: int) -> dict[str, int]:
        """Append 9 named, colored deck tracks to Live and return their indices.

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
            # Crossfader routing: A-side decks → group A, B-side decks →
            # group B, mix → None (always audible). Mirrors the static .als
            # writer (dance/als/writer.py:_crossfade_value_for) so a
            # live-loaded deck blends under the crossfader exactly like an
            # exported Set would. Requires the AbletonOSC fork patch — see
            # docs/abletonosc_setup.md. Best-effort: on stock AbletonOSC the
            # address is unhandled and the deck stays at Live's default
            # (None), so the crossfader just won't affect it.
            try:
                self.client.set_track_crossfade_assign(
                    idx, self._crossfade_assign_for(kind)
                )
            except OSError:  # pragma: no cover - best-effort
                pass
            if kind in ("mix_a", "mix_b"):
                try:
                    self.client.set_track_mute(idx, True)
                except OSError:  # pragma: no cover - best-effort
                    pass
            columns[kind] = idx
            idx += 1
        self._subscribe_deck_columns(columns)
        return columns

    @staticmethod
    def _side_of(deck_kind: str) -> str | None:
        """``"drums_a"`` → ``"a"``; ``"mix_b"`` → ``"b"``; bare → None.
        Mirrors dance.als.writer._side_of — kept in sync by hand (both walk
        the same _DECK_KINDS / DECK_ORDER shape)."""
        if deck_kind.endswith("_a"):
            return "a"
        if deck_kind.endswith("_b"):
            return "b"
        return None

    @classmethod
    def _crossfade_assign_for(cls, deck_kind: str) -> int:
        """Live MixerDevice.crossfade_assign enum for a deck-kind:
        A-side → 0 (A), B-side → 2 (B), anything else → 1 (None).

        Single semantic source of truth with the static .als writer's
        _crossfade_value_for (writer encodes the same 0/1/2 as strings into
        CrossFadeState/Manual). The integer constants live on
        AbletonOSCClient (CROSSFADE_A/NONE/B)."""
        side = cls._side_of(deck_kind)
        if side == "a":
            return AbletonOSCClient.CROSSFADE_A
        if side == "b":
            return AbletonOSCClient.CROSSFADE_B
        return AbletonOSCClient.CROSSFADE_NONE

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
            try:
                # PFL/Solo state — AbletonOSC replies once immediately and
                # on every change, populating AbletonState.soloed_kinds.
                self.client.start_listen_solo(track_idx)
            except OSError:  # pragma: no cover - best-effort
                pass
        # Column layout may have shifted (recovery / re-create), so the
        # track_index → kind mapping behind soloed_kinds can change even
        # before any new solo push arrives. Recompute from the layout we
        # were handed (self._deck_columns may not be assigned yet during
        # _create_deck_columns).
        self._recompute_soloed_kinds(columns)

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
