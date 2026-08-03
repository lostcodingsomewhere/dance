"""OSC bridge tests — use real UDP loopback so we exercise the network code,
not a mock. Each test gets its own free port via ``port=0``.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

from pythonosc import udp_client

from dance.osc.bridge import AbletonBridge, AbletonState
from dance.osc.client import AbletonOSCClient
from dance.osc.listener import AbletonOSCListener

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind to port 0, capture the OS-assigned port, release."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class _FakeLive:
    """A stateful fake Ableton that models the track-management subset the
    bridge's create/adopt paths exercise. Unlike a static num_tracks stub it
    actually grows/shrinks its track list on create/delete and reflects names
    on rename, so ``get_num_tracks`` / ``get_track_names`` replies stay
    consistent with what the bridge just did — which is what the robust
    create helper (count-delta confirmation + name verification) requires.

    It also models clip slots: ``create_audio_clip`` marks a (track, slot) as
    holding a clip, ``delete_clip`` clears it, and ``get_clip_slot_has_clip``
    replies accordingly. To reproduce the async create→fire race, the clip
    only becomes "present" after ``has_clip_delay`` has_clip queries have been
    answered (so a test can assert the bridge WAITED before firing).

    Knobs:
    - ``has_clip_delay``: number of has_clip queries returning False before
      the clip flips to present (models Live finishing the async create).
    - ``answer_has_clip``: when False, the fake never replies to has_clip at
      all (models a fork lacking the handler → bridge fires best-effort).
    - ``clip_never_appears``: when True, the clip is created but has_clip
      keeps returning False forever (models a failed/too-slow create).

    Run it as a context manager. ``received`` is the list of every OSC
    message the fake saw, for command-shape assertions.
    """

    def __init__(
        self,
        initial_names: list[str] | None = None,
        *,
        has_clip_delay: int = 0,
        answer_has_clip: bool = True,
        clip_never_appears: bool = False,
        is_playing_delay: int = 0,
        answer_is_playing: bool = True,
        clip_lengths: dict[tuple[int, int], float] | None = None,
        num_scenes: int = 8,
    ) -> None:
        self.names: list[str] = list(initial_names or [])
        self.received: list[tuple[str, tuple[Any, ...]]] = []
        self.send_port = _free_port()
        self.listen_port = _free_port()
        self._listener = AbletonOSCListener(port=self.send_port)
        self._reply = udp_client.SimpleUDPClient("127.0.0.1", self.listen_port)
        self._lock = __import__("threading").Lock()
        # (track, slot) -> number of has_clip queries still owed a False reply
        # before the clip is considered present. Created via create_audio_clip.
        self._pending_clip: dict[tuple[int, int], int] = {}
        self._present_clip: set[tuple[int, int]] = set()
        self._has_clip_delay = has_clip_delay
        self._answer_has_clip = answer_has_clip
        self._clip_never_appears = clip_never_appears
        # (track, slot) -> number of is_playing queries still owed a False
        # reply before the clip is considered playing (models a compressed
        # sample that keeps decoding after the clip object exists).
        self._is_playing_delay = is_playing_delay
        self._answer_is_playing = answer_is_playing
        self._playing_pending: dict[tuple[int, int], int] = {}
        # (track, slot) -> warped beat-length Live reports for that clip.
        # Models the outcome of Live's auto-warp: a stem it guessed at half
        # tempo comes back with half the beats. Absent key = no reply, which
        # is what a fork without the length handler (or a silent Live) does.
        self._clip_lengths: dict[tuple[int, int], float] = dict(clip_lengths or {})
        # Clip slots only exist inside scenes. An exported .als ships 8, and
        # creating a clip beyond the last one raises IndexError inside Live.
        self.num_scenes = num_scenes

    def _on_any(self, addr: str, args: tuple[Any, ...]) -> None:
        with self._lock:
            self.received.append((addr, args))
            if addr == "/live/song/create_audio_track":
                # AbletonOSC appends at the end for index -1 (the only form we
                # send). Give it a Live-default name like "5-Audio".
                self.names.append(f"{len(self.names) + 1}-Audio")
            elif addr == "/live/song/delete_track":
                idx = int(args[0])
                if 0 <= idx < len(self.names):
                    self.names.pop(idx)
            elif addr == "/live/track/set/name":
                idx, name = int(args[0]), str(args[1])
                if 0 <= idx < len(self.names):
                    self.names[idx] = name
            elif addr == "/live/song/create_scene":
                self.num_scenes += 1
            elif addr == "/live/song/get/num_scenes":
                self._reply.send_message("/live/song/get/num_scenes", [self.num_scenes])
            elif addr == "/live/song/get/num_tracks":
                self._reply.send_message(
                    "/live/song/get/num_tracks", [len(self.names)]
                )
            elif addr == "/live/song/get/track_names":
                self._reply.send_message(
                    "/live/song/get/track_names", list(self.names)
                )
            elif addr == "/live/clip_slot/create_audio_clip":
                key = (int(args[0]), int(args[1]))
                self._present_clip.discard(key)
                self._pending_clip[key] = self._has_clip_delay
                self._playing_pending[key] = self._is_playing_delay
            elif addr == "/live/clip_slot/delete_clip":
                key = (int(args[0]), int(args[1]))
                self._present_clip.discard(key)
                self._pending_clip.pop(key, None)
            elif addr == "/live/clip_slot/get/has_clip":
                if not self._answer_has_clip:
                    return  # model a fork without the has_clip handler
                key = (int(args[0]), int(args[1]))
                present = key in self._present_clip
                if not present and key in self._pending_clip and not self._clip_never_appears:
                    if self._pending_clip[key] <= 0:
                        self._present_clip.add(key)
                        self._pending_clip.pop(key, None)
                        present = True
                    else:
                        self._pending_clip[key] -= 1
                self._reply.send_message(
                    "/live/clip_slot/get/has_clip",
                    [key[0], key[1], 1 if present else 0],
                )
            elif addr == "/live/clip/get/length":
                key = (int(args[0]), int(args[1]))
                if key not in self._clip_lengths:
                    return  # no clip / no handler → bridge sees a timeout
                self._reply.send_message(
                    "/live/clip/get/length",
                    [key[0], key[1], self._clip_lengths[key]],
                )
            elif addr == "/live/clip/get/is_playing":
                if not self._answer_is_playing:
                    return  # model a fork without the is_playing handler
                key = (int(args[0]), int(args[1]))
                playing = key in self._present_clip
                if playing and self._playing_pending.get(key, 0) > 0:
                    # Still "decoding" — owe a few False replies first.
                    self._playing_pending[key] -= 1
                    playing = False
                self._reply.send_message(
                    "/live/clip/get/is_playing",
                    [key[0], key[1], 1 if playing else 0],
                )

    def __enter__(self) -> _FakeLive:
        self._listener.on_any(self._on_any)
        self._listener.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._listener.stop()

    def make_bridge(self) -> AbletonBridge:
        return AbletonBridge(send_port=self.send_port, listen_port=self.listen_port)

    def count(self, addr: str) -> int:
        with self._lock:
            return sum(1 for a, _ in self.received if a == addr)

    def args_for(self, addr: str) -> list[tuple[Any, ...]]:
        with self._lock:
            return [args for a, args in self.received if a == addr]


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


def test_listener_receives_messages():
    port = _free_port()
    listener = AbletonOSCListener(port=port)
    received: list[tuple[str, tuple[Any, ...]]] = []

    listener.on("/foo/bar", lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = udp_client.SimpleUDPClient("127.0.0.1", port)
        client.send_message("/foo/bar", [42, "hello"])

        assert _wait_for(lambda: len(received) > 0)
        addr, args = received[0]
        assert addr == "/foo/bar"
        assert args == (42, "hello")
    finally:
        listener.stop()


def test_listener_unmatched_address_is_silent():
    """Messages with no registered handler don't crash the listener."""
    port = _free_port()
    listener = AbletonOSCListener(port=port)
    received: list[Any] = []
    listener.on("/handled", lambda a, args: received.append(args))
    listener.start()
    try:
        client = udp_client.SimpleUDPClient("127.0.0.1", port)
        client.send_message("/ignored", [1])
        client.send_message("/handled", [2])
        assert _wait_for(lambda: received == [(2,)])
    finally:
        listener.stop()


def test_listener_on_any_catches_everything():
    port = _free_port()
    listener = AbletonOSCListener(port=port)
    seen: list[str] = []
    listener.on_any(lambda addr, args: seen.append(addr))
    listener.start()
    try:
        client = udp_client.SimpleUDPClient("127.0.0.1", port)
        client.send_message("/a", [1])
        client.send_message("/b", [2])
        assert _wait_for(lambda: set(seen) == {"/a", "/b"})
    finally:
        listener.stop()


def test_listener_handler_exception_does_not_kill_thread():
    port = _free_port()
    listener = AbletonOSCListener(port=port)
    good: list[Any] = []
    listener.on("/boom", lambda a, args: (_ for _ in ()).throw(RuntimeError("nope")))
    listener.on("/boom", lambda a, args: good.append(args))
    listener.on("/after", lambda a, args: good.append(args))
    listener.start()
    try:
        client = udp_client.SimpleUDPClient("127.0.0.1", port)
        client.send_message("/boom", [1])
        client.send_message("/after", [2])
        assert _wait_for(lambda: (1,) in good and (2,) in good)
    finally:
        listener.stop()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def test_client_sends_to_correct_address():
    """Spin up a listener as a fake Ableton, verify the client's address layout."""
    port = _free_port()
    received: list[tuple[str, tuple[Any, ...]]] = []
    listener = AbletonOSCListener(port=port)
    listener.on_any(lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = AbletonOSCClient(port=port)
        client.play()
        client.set_tempo(128.5)
        client.fire_clip(track=2, scene=4)
        client.set_track_volume(track=1, volume=0.75)

        assert _wait_for(lambda: len(received) >= 4)
        addrs = [a for a, _ in received]
        assert "/live/song/start_playing" in addrs
        assert "/live/song/set/tempo" in addrs
        assert "/live/clip_slot/fire" in addrs
        assert "/live/track/set/volume" in addrs

        # Argument shapes
        by_addr = {a: args for a, args in received}
        assert by_addr["/live/song/set/tempo"] == (128.5,)
        assert by_addr["/live/clip_slot/fire"] == (2, 4)
        assert by_addr["/live/track/set/volume"] == (1, 0.75)
    finally:
        listener.stop()


def test_client_has_clip_and_launch_quant_addresses():
    """get_clip_slot_has_clip + set_clip_launch_quantization emit the right
    addresses/args (used by the preview create→fire race fix)."""
    port = _free_port()
    received: list[tuple[str, tuple[Any, ...]]] = []
    listener = AbletonOSCListener(port=port)
    listener.on_any(lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = AbletonOSCClient(port=port)
        client.get_clip_slot_has_clip(10, 0)
        client.set_clip_launch_quantization(10, 0, 0)
        assert _wait_for(lambda: len(received) >= 2)
        by_addr = {a: args for a, args in received}
        assert by_addr["/live/clip_slot/get/has_clip"] == (10, 0)
        assert by_addr["/live/clip/set/launch_quantization"] == (10, 0, 0)
        # launch_quant value must be a plain int for OSC.
        assert type(by_addr["/live/clip/set/launch_quantization"][2]) is int
    finally:
        listener.stop()


def test_client_track_and_clip_management_addresses():
    """Cover the new track/clip management commands added for Wave 3."""
    port = _free_port()
    received: list[tuple[str, tuple[Any, ...]]] = []
    listener = AbletonOSCListener(port=port)
    listener.on_any(lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = AbletonOSCClient(port=port)
        client.create_audio_track(-1)
        client.delete_track(7)
        client.create_scene(0)
        client.set_track_name(3, "Drums")
        client.set_track_color(3, 0xFF3030)
        client.create_clip(track=2, slot=0, length=16.0)
        client.delete_clip(track=2, slot=0)
        client.set_clip_warp(track=1, slot=0, warp=True)
        client.set_clip_loop(track=1, slot=0, start_beats=0.0, end_beats=32.0)
        client.set_clip_color(track=1, slot=0, color=12)
        client.set_clip_name(track=1, slot=0, name="loop A")
        client.get_num_tracks()
        client.show_message("hello")

        assert _wait_for(lambda: len(received) >= 13)
        by_addr = {a: args for a, args in received}
        assert by_addr["/live/song/create_audio_track"] == (-1,)
        assert by_addr["/live/song/delete_track"] == (7,)
        assert by_addr["/live/song/create_scene"] == (0,)
        assert by_addr["/live/track/set/name"] == (3, "Drums")
        assert by_addr["/live/track/set/color"] == (3, 0xFF3030)
        assert by_addr["/live/clip_slot/create_clip"] == (2, 0, 16.0)
        assert by_addr["/live/clip_slot/delete_clip"] == (2, 0)
        assert by_addr["/live/clip/set/warping"] == (1, 0, 1)
        # set_clip_loop sends two messages — the last-seen pair survives in by_addr.
        assert by_addr["/live/clip/set/loop_start"] == (1, 0, 0.0)
        assert by_addr["/live/clip/set/loop_end"] == (1, 0, 32.0)
        assert by_addr["/live/clip/set/color_index"] == (1, 0, 12)
        assert by_addr["/live/clip/set/name"] == (1, 0, "loop A")
        assert "/live/song/get/num_tracks" in by_addr
        assert by_addr["/live/api/show_message"] == ("hello",)
    finally:
        listener.stop()


def test_client_set_clip_warp_false_is_zero():
    port = _free_port()
    received: list[tuple[str, tuple[Any, ...]]] = []
    listener = AbletonOSCListener(port=port)
    listener.on_any(lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = AbletonOSCClient(port=port)
        client.set_clip_warp(track=0, slot=0, warp=False)
        assert _wait_for(lambda: any(a == "/live/clip/set/warping" for a, _ in received))
        by_addr = {a: args for a, args in received}
        assert by_addr["/live/clip/set/warping"] == (0, 0, 0)
    finally:
        listener.stop()


def test_client_bool_args_are_ints_for_osc():
    """OSC has no bool type — we encode mute/solo as 0/1."""
    port = _free_port()
    received: list[tuple[str, tuple[Any, ...]]] = []
    listener = AbletonOSCListener(port=port)
    listener.on_any(lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = AbletonOSCClient(port=port)
        client.set_track_mute(track=0, muted=True)
        client.set_track_solo(track=0, soloed=False)

        assert _wait_for(lambda: len(received) >= 2)
        by_addr = {a: args for a, args in received}
        assert by_addr["/live/track/set/mute"] == (0, 1)
        assert by_addr["/live/track/set/solo"] == (0, 0)
    finally:
        listener.stop()


# ---------------------------------------------------------------------------
# Bridge — wires state updates
# ---------------------------------------------------------------------------


def test_bridge_updates_state_on_tempo():
    """When Live pushes a tempo, the bridge's state.tempo updates."""
    listen_port = _free_port()
    # Send port doesn't matter — start() will try to subscribe but failure is benign.
    bridge = AbletonBridge(send_port=_free_port(), listen_port=listen_port)
    bridge.start()
    try:
        fake_live = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        fake_live.send_message("/live/song/get/tempo", [127.0])
        assert _wait_for(lambda: bridge.state.tempo == 127.0)
    finally:
        bridge.stop()


def test_bridge_updates_playing_clips():
    listen_port = _free_port()
    bridge = AbletonBridge(send_port=_free_port(), listen_port=listen_port)
    bridge.start()
    try:
        fake_live = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        fake_live.send_message("/live/track/get/playing_slot_index", [0, 3])
        fake_live.send_message("/live/track/get/playing_slot_index", [1, 5])
        assert _wait_for(lambda: bridge.state.playing_clips == {0: 3, 1: 5})
    finally:
        bridge.stop()


def test_bridge_broadcasts_to_subscribers():
    listen_port = _free_port()
    bridge = AbletonBridge(send_port=_free_port(), listen_port=listen_port)
    received: list[AbletonState] = []
    bridge.subscribe(lambda s: received.append(AbletonState(tempo=s.tempo)))
    bridge.start()
    try:
        fake_live = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        fake_live.send_message("/live/song/get/tempo", [124.0])
        assert _wait_for(lambda: len(received) >= 1 and received[-1].tempo == 124.0)
    finally:
        bridge.stop()


def test_bridge_get_num_tracks_roundtrip():
    """Bridge sends a query, fake Live replies, bridge surfaces the value."""
    listen_port = _free_port()
    send_port = _free_port()

    # Fake Live: listens on the bridge's send port, replies on the bridge's
    # listen port with /live/song/get/num_tracks.
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [12])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.start()

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        result = bridge.get_num_tracks(timeout=1.0)
        assert result == 12
    finally:
        bridge.stop()
        fake_live_listener.stop()


def test_bridge_get_num_tracks_times_out_when_live_silent():
    listen_port = _free_port()
    send_port = _free_port()
    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        assert bridge.get_num_tracks(timeout=0.1) is None
    finally:
        bridge.stop()


# Deck-column recovery — the 10 deck tracks in index order, used by both the
# prefixed (OSC-created) and bare (.als-writer) name tests below.
_DECK_COLUMN_ORDER = [
    "drums_a",
    "drums_b",
    "bass_a",
    "bass_b",
    "vocals_a",
    "vocals_b",
    "other_a",
    "other_b",
    "mix_a",
    "mix_b",
]


def _recover_with_track_names(names: list[str]) -> dict[str, int] | None:
    """Spin up a fake Live that replies to /live/song/get/track_names with
    ``names``, then run recover_deck_columns against it. Returns the
    recovered column map (or None)."""
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/track_names", names)

    fake_live_listener.on("/live/song/get/track_names", on_query)
    fake_live_listener.start()

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        return bridge.recover_deck_columns(timeout=1.0)
    finally:
        bridge.stop()
        fake_live_listener.stop()


def test_bridge_recover_adopts_bare_als_writer_names():
    """Opening an exported .als yields tracks named "Drums A" … "Mix B"
    (no "Deck " prefix — see dance/als/writer.py:_display_for). Recovery
    must adopt those so the deck grid populates after `dance export-als`.
    Regression for the prefix mismatch that left the app stuck on
    "Waiting for Ableton deck columns"."""
    bare_names = [
        "Drums A",
        "Drums B",
        "Bass A",
        "Bass B",
        "Vocals A",
        "Vocals B",
        "Other A",
        "Other B",
        "Mix A",
        "Mix B",
    ]
    recovered = _recover_with_track_names(bare_names)
    assert recovered is not None
    assert recovered == {kind: i for i, kind in enumerate(_DECK_COLUMN_ORDER)}


def test_bridge_recover_adopts_prefixed_deck_names():
    """The canonical OSC-created layout uses "Deck Drums A" … "Deck Mix B".
    Recovery must keep adopting those too (no regression from widening the
    accepted set to include the bare .als names)."""
    prefixed = [
        "Deck Drums A",
        "Deck Drums B",
        "Deck Bass A",
        "Deck Bass B",
        "Deck Vocals A",
        "Deck Vocals B",
        "Deck Other A",
        "Deck Other B",
        "Deck Mix A",
        "Deck Mix B",
    ]
    recovered = _recover_with_track_names(prefixed)
    assert recovered is not None
    assert recovered == {kind: i for i, kind in enumerate(_DECK_COLUMN_ORDER)}


def test_bridge_recover_returns_none_on_partial_layout():
    """If only some deck tracks are present (e.g. user renamed one), recovery
    bails rather than adopting a half-built grid."""
    partial = ["Drums A", "Drums B", "Bass A"]  # missing the rest
    assert _recover_with_track_names(partial) is None


def _stub_stem(kind: str, path: str):
    """Quack-typed stand-in for a StemFile row (avoids hitting the DB)."""
    class _S:
        pass
    s = _S()
    s.kind = kind
    s.path = path
    return s


def _stub_track(id: int = 1, title: str = "My Song", file_path: str = "/tmp/missing.wav"):
    class _T:
        pass
    t = _T()
    t.id = id
    t.title = title
    t.file_name = "song.wav"
    t.file_path = file_path
    return t


def test_bridge_push_track_to_live_creates_tracks_in_order():
    """Ten OSC create_audio_track calls — 8 stem decks (A/B × 4 roles) +
    mix_a + mix_b, in deck-pair order (stems first, mixes last so APC40's
    default 8-strip view maps 1:1 to the stem decks).

    Uses a stateful fake Live (10 pre-existing tracks) so the robust
    create helper's count-delta confirmation lands the new tracks at the
    real appended indices 10..19."""
    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            track = _stub_track(id=42, title="Test Track")
            stems = [
                _stub_stem("drums", "/tmp/d.wav"),
                _stub_stem("bass", "/tmp/b.wav"),
                _stub_stem("vocals", "/tmp/v.wav"),
                _stub_stem("other", "/tmp/o.wav"),
            ]
            result = bridge.push_track_to_live(track, stems, include_stems=True)

            # Indices should be the next 10 slots (10..19) in deck-pair order:
            # 8 stem decks first (A/B per role), then mix_a + mix_b. APC40's
            # default 8-strip view maps to the 8 stem decks; mixes live
            # beyond, reachable via bank-shift.
            assert result["scene_index"] == 0
            assert result["track_indices"]["drums_a"] == 10
            assert result["track_indices"]["drums_b"] == 11
            assert result["track_indices"]["bass_a"] == 12
            assert result["track_indices"]["bass_b"] == 13
            assert result["track_indices"]["vocals_a"] == 14
            assert result["track_indices"]["vocals_b"] == 15
            assert result["track_indices"]["other_a"] == 16
            assert result["track_indices"]["other_b"] == 17
            assert result["track_indices"]["mix_a"] == 18
            assert result["track_indices"]["mix_b"] == 19

            # Exactly 10 create_audio_track messages — no duplicates.
            assert live.count("/live/song/create_audio_track") == 10
            # All 10 deck tracks got named with the canonical "Deck …" form.
            joined = " | ".join(str(a[1]) for a in live.args_for("/live/track/set/name"))
            assert "Drums A" in joined and "Drums B" in joined
            assert "Bass A" in joined and "Bass B" in joined
            assert "Vocals A" in joined and "Vocals B" in joined
            assert "Other A" in joined and "Other B" in joined
            assert "Mix A" in joined and "Mix B" in joined
            # Both mix tracks are muted on creation (reference / parachute,
            # not double-summed audio). mix_a at idx 18, mix_b at idx 19.
            # Poll: styling sends trail the synchronous return on the fake's
            # listener thread.
            assert _wait_for(
                lambda: (18, 1) in live.args_for("/live/track/set/mute")
                and (19, 1) in live.args_for("/live/track/set/mute")
            )
        finally:
            bridge.stop()


def test_bridge_push_track_to_live_loads_mix_cell_on_full_song(tmp_path):
    """A whole-song load also drops the original mix file into the SONG
    cell. Without this the UI's SceneGrid renders the SONG column as empty
    after every load — looks broken, even though the stems were loaded.
    """
    # Real on-disk files so the existence guards pass.
    mix_path = tmp_path / "song.wav"
    mix_path.write_bytes(b"RIFF")
    drums_path = tmp_path / "drums.wav"
    drums_path.write_bytes(b"RIFF")

    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            track = _stub_track(id=7, title="My Song", file_path=str(mix_path))
            stems = [_stub_stem("drums", str(drums_path))]
            bridge.push_track_to_live(track, stems, include_stems=True)

            # mix_a is at idx 18 (8 stem decks + mix_a). _pick_side picks A
            # for the empty row, so the source mix file lands on mix_a not
            # mix_b. Poll: the fake records OSC on its listener thread, which
            # can lag the synchronous push_track_to_live return.
            assert _wait_for(
                lambda: any(
                    args[0] == 18 and args[2] == str(mix_path)
                    for args in live.args_for("/live/clip_slot/create_audio_clip")
                )
            )
            # _deck_cells records the side-matching mix cell.
            assert (0, "mix_a") in bridge._deck_cells
            assert bridge._deck_cells[(0, "mix_a")] == 7
            # mix_b stays empty — only one side gets the mix per load.
            assert (0, "mix_b") not in bridge._deck_cells
            # Drums landed on the A side (same side as the mix).
            assert (0, "drums_a") in bridge._deck_cells
            assert bridge._deck_cells[(0, "drums_a")] == 7
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Warp guard — forcing warp on load, and catching Live's bad auto-warp guesses
# ---------------------------------------------------------------------------

# Deck-column indices when the fake starts with 10 pre-existing tracks.
_A_STEM_IDX = {"drums_a": 10, "bass_a": 12, "vocals_a": 14, "other_a": 16}


def _full_song_stems(tmp_path):
    """Four real on-disk stem files + the mix, for a whole-song load."""
    paths = {}
    for kind in ("drums", "bass", "vocals", "other"):
        p = tmp_path / f"{kind}.wav"
        p.write_bytes(b"RIFF")
        paths[kind] = p
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"RIFF")
    return mix, [_stub_stem(k, str(v)) for k, v in paths.items()]


def _load_then_check_warp(live, *, expected_beats=None, tmp_path=None, track_id=7):
    """Whole-song load, then the warp audit — the real sequence.

    The audit is deliberately a SEPARATE step from the load: measured against
    Live 12.4.2, a fresh clip reports a placeholder length (all stems agree)
    for ~13-15s before the real auto-warp analysis lands.
    """
    mix, stems = _full_song_stems(tmp_path)
    bridge = live.make_bridge()
    bridge.start()
    try:
        track = _stub_track(id=track_id, file_path=str(mix))
        bridge.push_track_to_live(track, stems, include_stems=True)
        expected = {track_id: expected_beats} if expected_beats else None
        return bridge.check_warp_at(0, expected_beats_by_track=expected)
    finally:
        bridge.stop()


def test_bridge_load_forces_warp_beats_on_every_stem(tmp_path):
    """Every live-loaded stem gets Warp ON + Beats mode, explicitly.

    ``create_audio_clip`` alone leaves warp to Live's auto-warp preference,
    so the deck path used to disagree with the .als writer (which pins
    Warp=true / WarpMode=0). A deck you can't beatmatch is not a deck.
    """
    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        _load_then_check_warp(live, tmp_path=tmp_path)

        for deck_kind, idx in _A_STEM_IDX.items():
            assert _wait_for(
                lambda i=idx: (i, 0, 1) in live.args_for("/live/clip/set/warping")
            ), f"{deck_kind} was not forced to warp"
            assert _wait_for(
                lambda i=idx: (i, 0, 0) in live.args_for("/live/clip/set/warp_mode")
            ), f"{deck_kind} was not pinned to Beats mode"


def test_bridge_warp_check_flags_half_warped_stem(tmp_path):
    """One stem at half the others' beat-length = Live guessed half tempo.

    This is the failure that makes a beginner think they played the wrong
    thing: nothing errors, the vocal is just in a different universe. The
    check names the cell and the one-click fix in Live.
    """
    lengths = {(idx, 0): 700.0 for idx in _A_STEM_IDX.values()}
    lengths[(_A_STEM_IDX["vocals_a"], 0)] = 350.0  # Live halved this one
    with _FakeLive(
        initial_names=[f"Pre {i}" for i in range(10)], clip_lengths=lengths
    ) as live:
        result = _load_then_check_warp(live, tmp_path=tmp_path)

    warp_warnings = [w for w in result["warnings"] if "vocals_a" in w]
    assert len(warp_warnings) == 1
    assert "HALF" in warp_warnings[0]
    assert "*2" in warp_warnings[0]
    # The three that agreed are not accused.
    assert not [w for w in result["warnings"] if "drums_a" in w or "bass_a" in w]


def test_bridge_warp_check_flags_double_warped_stem(tmp_path):
    """Mirror case — one stem at 2x the others gets the ':2' instruction."""
    lengths = {(idx, 0): 700.0 for idx in _A_STEM_IDX.values()}
    lengths[(_A_STEM_IDX["bass_a"], 0)] = 1400.0
    with _FakeLive(
        initial_names=[f"Pre {i}" for i in range(10)], clip_lengths=lengths
    ) as live:
        result = _load_then_check_warp(live, tmp_path=tmp_path)

    warp_warnings = [w for w in result["warnings"] if "bass_a" in w]
    assert len(warp_warnings) == 1
    assert "DOUBLE" in warp_warnings[0]
    assert ":2" in warp_warnings[0]


def test_bridge_warp_check_silent_when_stems_agree(tmp_path):
    """The common case must be quiet — a warning the DJ learns to ignore is
    worse than no warning. Agreement within tolerance = nothing said."""
    lengths = {(idx, 0): 700.0 for idx in _A_STEM_IDX.values()}
    # 1% jitter is well inside _WARP_AGREE_TOL and must not trip anything.
    lengths[(_A_STEM_IDX["other_a"], 0)] = 707.0
    with _FakeLive(
        initial_names=[f"Pre {i}" for i in range(10)], clip_lengths=lengths
    ) as live:
        result = _load_then_check_warp(live, expected_beats=700.0, tmp_path=tmp_path)

    assert [w for w in result["warnings"] if "warp" in w.lower()] == []


def test_bridge_warp_check_flags_whole_track_octave_flip(tmp_path):
    """Stems can agree with each other and still all be wrong — Live
    octave-flipping the whole track. Only the analyzer can catch that, so
    it's phrased as a check, not an accusation (CLAUDE.md rule 3)."""
    lengths = {(idx, 0): 350.0 for idx in _A_STEM_IDX.values()}
    with _FakeLive(
        initial_names=[f"Pre {i}" for i in range(10)], clip_lengths=lengths
    ) as live:
        result = _load_then_check_warp(live, expected_beats=700.0, tmp_path=tmp_path)

    flips = [w for w in result["warnings"] if "consistently" in w]
    assert len(flips) == 1
    assert "half" in flips[0]
    # Framed as "sounds right? trust Live" — never as a hard error.
    assert "Trust Live" in flips[0]


def test_bridge_warp_check_gives_a_target_bpm_for_non_octave_drift(tmp_path):
    """A 9% drift is NOT an octave error, so ':2' / '*2' cannot fix it — the
    DJ has to type a segment BPM. Measured on the real rig: a bass stem read
    as 113.71 BPM against 124.98 drums. The message has to carry both numbers.
    """
    # 340.7s source. drums/vocals/other agree; bass sits ~9% low.
    dur = 340.7
    good = dur * 124.98 / 60.0
    bad = dur * 113.71 / 60.0
    lengths = {(idx, 0): good for idx in _A_STEM_IDX.values()}
    lengths[(_A_STEM_IDX["bass_a"], 0)] = bad
    with _FakeLive(
        initial_names=[f"Pre {i}" for i in range(10)], clip_lengths=lengths
    ) as live:
        mix, stems = _full_song_stems(tmp_path)
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=7, file_path=str(mix)), stems, include_stems=True
            )
            result = bridge.check_warp_at(0, duration_by_track={7: dur})
        finally:
            bridge.stop()

    hits = [w for w in result["warnings"] if "bass_a" in w]
    assert len(hits) == 1
    # Not misfiled as an octave error.
    assert "*2" not in hits[0] and ":2" not in hits[0]
    # Carries what Live thinks, what it should be, and where to type it.
    assert "113.7" in hits[0]
    assert "125.0" in hits[0]
    assert "Seg. BPM" in hits[0]


def test_bridge_warp_check_silent_when_live_never_answers(tmp_path):
    """No length replies (Live closed, or a fork without the handler) must
    degrade to silence, not to a false alarm."""
    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        result = _load_then_check_warp(live, expected_beats=700.0, tmp_path=tmp_path)

    assert [w for w in result["warnings"] if "warp" in w.lower()] == []


def test_load_itself_never_queries_length(tmp_path):
    """The load path must not run the check. Live reports a placeholder
    length until its analysis lands ~15s later, so an inline check reads the
    placeholder — where every stem agrees — and reports all-clear on a scene
    that is actually broken. Measured on Live 12.4.2."""
    with _FakeLive(
        initial_names=[f"Pre {i}" for i in range(10)],
        clip_lengths={(idx, 0): 700.0 for idx in _A_STEM_IDX.values()},
    ) as live:
        mix, stems = _full_song_stems(tmp_path)
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=7, file_path=str(mix)), stems, include_stems=True
            )
        finally:
            bridge.stop()
        # Warp IS pinned during load (per-clip, free, matches the .als writer)...
        for idx in _A_STEM_IDX.values():
            assert _wait_for(
                lambda i=idx: (i, 0, 1) in live.args_for("/live/clip/set/warping")
            )
        # ...but not one length was read.
        assert live.count("/live/clip/get/length") == 0


def test_warp_check_compares_only_within_one_source_track(tmp_path):
    """A scene in the live-remixing model holds stems from different songs.
    Those have different durations, so comparing them to each other is
    meaningless — the check must group by source track first."""
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    other = tmp_path / "other.wav"
    other.write_bytes(b"RIFF")
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"RIFF")
    lengths = {
        (_A_STEM_IDX["drums_a"], 0): 700.0,
        (_A_STEM_IDX["other_a"], 0): 350.0,  # different SONG, legitimately shorter
    }
    with _FakeLive(
        initial_names=[f"Pre {i}" for i in range(10)], clip_lengths=lengths
    ) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=7, file_path=str(mix)),
                [_stub_stem("drums", str(drums))],
                kinds=["drums"],
                side="a",
            )
            bridge.push_track_to_live(
                _stub_track(id=99, file_path=str(mix)),
                [_stub_stem("other", str(other))],
                kinds=["other"],
                side="a",
            )
            result = bridge.check_warp_at(0)
        finally:
            bridge.stop()

    # Two sources, one stem each — nothing is comparable, so nothing is said.
    assert result["checked"] == 0
    assert result["warnings"] == []


def test_load_clears_a_slot_live_already_has_a_clip_in(tmp_path):
    """Loading onto an occupied slot must DELETE first, or the rename lands on
    the stale clip and the app lies about what will play.

    Live raises "This clip slot already has a clip" and AbletonOSC has no
    negative-reply channel, so the failed create is invisible — but the
    set_clip_name that follows succeeds on the OLD clip. Card, clip name and
    deck map then all show the new track while the previous audio fires.
    Live's log on this rig recorded it 16 times.

    _deck_cells cannot be trusted for occupancy: it is the bridge's memory of
    what IT loaded, and an opened .als export puts a clip in slot 0 of every
    A-side deck track that the bridge never knew about.
    """
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"RIFF")

    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            # Pre-occupy drums_a @ scene 0 in LIVE only — the bridge's
            # _deck_cells stays empty, exactly like an opened .als export.
            bridge._deck_columns = bridge._create_deck_columns(start_index=10)
            drums_idx = bridge._deck_columns["drums_a"]
            live.received.clear()
            bridge.client.create_audio_clip(drums_idx, 0, str(drums))
            assert _wait_for(lambda: (drums_idx, 0) in live._present_clip
                             or (drums_idx, 0) in live._pending_clip)

            bridge.push_track_to_live(
                _stub_track(id=7, file_path=str(mix)),
                [_stub_stem("drums", str(drums))],
                kinds=["drums"],
                side="a",
                scene_index=0,
            )
        finally:
            bridge.stop()

        # The occupied slot was cleared before the new create.
        deletes = live.args_for("/live/clip_slot/delete_clip")
        assert (drums_idx, 0) in deletes, "occupied slot was not cleared"
        order = [a for a, _ in live.received]
        del_pos = max(i for i, a in enumerate(order) if a == "/live/clip_slot/delete_clip")
        create_positions = [
            i for i, (a, args) in enumerate(live.received)
            if a == "/live/clip_slot/create_audio_clip" and args[:2] == (drums_idx, 0)
        ]
        assert create_positions and create_positions[-1] > del_pos, (
            "delete must precede the replacing create"
        )


def test_bridge_push_track_to_live_skips_mix_on_single_stem_load(tmp_path):
    """Single-stem loads (kinds=['drums']) must NOT drop the mix file —
    the mix cell stays empty so we don't fight the user's remix in progress.
    """
    mix_path = tmp_path / "song.wav"
    mix_path.write_bytes(b"RIFF")
    drums_path = tmp_path / "drums.wav"
    drums_path.write_bytes(b"RIFF")

    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            track = _stub_track(id=7, file_path=str(mix_path))
            stems = [_stub_stem("drums", str(drums_path))]
            bridge.push_track_to_live(track, stems, kinds=["drums"])
            # No mix entry on either side — single-stem load doesn't touch
            # mix_a or mix_b.
            mix_cells = [k for k in bridge._deck_cells if k[1].startswith("mix")]
            assert mix_cells == []
            # The drums stem landed on the A side (empty row → 'a').
            assert (0, "drums_a") in bridge._deck_cells
            assert (0, "drums_b") not in bridge._deck_cells
        finally:
            bridge.stop()


def test_bridge_promote_forced_side_enqueues_not_clobbers(tmp_path):
    """Promoting a rec onto a side whose cell is already loaded (and maybe
    playing) must enqueue on the next free scene of THAT side — never
    overwrite the occupied cell. Reproduces the promote-clobber bug."""
    drums1 = tmp_path / "drums1.wav"
    drums1.write_bytes(b"RIFF")
    drums2 = tmp_path / "drums2.wav"
    drums2.write_bytes(b"RIFF")

    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            # First drums stem live-loaded onto Deck A at scene 0 (playing).
            r1 = bridge.push_track_to_live(
                _stub_track(id=1), [_stub_stem("drums", str(drums1))], kinds=["drums"], side="a"
            )
            assert r1["scene_index"] == 0
            assert bridge._deck_cells[(0, "drums_a")] == 1

            # Promote a different drums rec, forced onto Deck A again. Must land
            # on scene 1, NOT overwrite the scene-0 drums_a cell.
            r2 = bridge.push_track_to_live(
                _stub_track(id=2), [_stub_stem("drums", str(drums2))], kinds=["drums"], side="a"
            )
            assert r2["scene_index"] == 1
            assert r2["stems_loaded"] == 1
            # Scene-0 cell preserved (still track 1); new cell at scene 1.
            assert bridge._deck_cells[(0, "drums_a")] == 1
            assert bridge._deck_cells[(1, "drums_a")] == 2
            # Idempotency: the second load must NOT spawn a fresh deck set —
            # 10 creates total (just the one initial provisioning).
            assert live.count("/live/song/create_audio_track") == 10
        finally:
            bridge.stop()


def test_bridge_promote_auto_side_uses_free_side_at_scene_zero(tmp_path):
    """With Deck A busy at scene 0 and no forced side, a promote should fill
    the free B side at the SAME scene rather than overwriting A or skipping
    to a fresh scene."""
    drums1 = tmp_path / "drums1.wav"
    drums1.write_bytes(b"RIFF")
    drums2 = tmp_path / "drums2.wav"
    drums2.write_bytes(b"RIFF")

    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=1), [_stub_stem("drums", str(drums1))], kinds=["drums"], side="a"
            )
            # No forced side: should pick the free B side at scene 0.
            r2 = bridge.push_track_to_live(
                _stub_track(id=2), [_stub_stem("drums", str(drums2))], kinds=["drums"]
            )
            assert r2["scene_index"] == 0
            assert bridge._deck_cells[(0, "drums_a")] == 1
            assert bridge._deck_cells[(0, "drums_b")] == 2
        finally:
            bridge.stop()


def test_bridge_push_track_to_live_include_stems_false_loads_no_cells():
    """include_stems=False loads zero cells but still provisions the 10
    reusable deck columns (8 stem decks A/B × 4 roles + mix_a + mix_b)
    so future loads have somewhere to land."""
    with _FakeLive(initial_names=[]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.push_track_to_live(
                _stub_track(), [], include_stems=False
            )
            # All 10 deck columns get provisioned on first call.
            assert set(result["track_indices"].keys()) == {
                "drums_a", "drums_b",
                "bass_a", "bass_b",
                "vocals_a", "vocals_b",
                "other_a", "other_b",
                "mix_a", "mix_b",
            }
            # First call appends at 0..9 (empty session).
            assert result["track_indices"]["drums_a"] == 0
            assert result["track_indices"]["mix_b"] == 9
            # But nothing actually loaded since kinds resolved to [].
            assert result["stems_loaded"] == 0
            assert live.count("/live/song/create_audio_track") == 10
        finally:
            bridge.stop()


def test_bridge_push_track_to_live_records_warning_for_missing_stem_file():
    """If a stem path doesn't exist on disk, we don't crash — we warn."""
    with _FakeLive(initial_names=[]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            track = _stub_track(file_path="/definitely/does/not/exist.wav")
            stems = [_stub_stem("drums", "/also/missing.wav")]
            result = bridge.push_track_to_live(track, stems, include_stems=True)
            warns = " ".join(result["warnings"])
            assert "drums stem file missing" in warns
            # The "bass/vocals/other" stems aren't supplied -> their own warnings.
            assert "No bass stem" in warns
            assert "No vocals stem" in warns
            assert "No other stem" in warns
            # The full-mix file is never loaded as a clip, so its existence is
            # not checked here — that was an older song-mode assumption.
        finally:
            bridge.stop()


def test_bridge_push_track_to_live_proceeds_when_live_unreachable():
    """When num_tracks times out, we still push — just with a warning."""
    listen_port = _free_port()
    send_port = _free_port()
    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        result = bridge.push_track_to_live(
            _stub_track(), [], include_stems=False, num_tracks_timeout=0.05
        )
        # Live-unreachable path defaults start_index=0; mix_a and mix_b
        # are the LAST two deck columns so they land at idx 8 and 9
        # (after 8 stem decks).
        assert result["track_indices"]["drums_a"] == 0
        assert result["track_indices"]["mix_a"] == 8
        assert result["track_indices"]["mix_b"] == 9
        assert any("num_tracks" in w for w in result["warnings"])
    finally:
        bridge.stop()


def test_bridge_state_to_dict_is_json_safe():
    state = AbletonState(tempo=128.0, is_playing=True, beat=4.5)
    state.playing_clips[0] = 2
    state.track_volumes[1] = 0.8

    d = state.to_dict()
    import json
    json.dumps(d)  # would raise if non-JSON-safe
    assert d["tempo"] == 128.0
    assert d["playing_clips"] == {0: 2}


# ---------------------------------------------------------------------------
# Live-contract additions: crossfade_assign, soloed_kinds, deck_map_revision
# ---------------------------------------------------------------------------


def test_client_set_crossfade_assign_address_and_args():
    """set_track_crossfade_assign emits /live/track/set/crossfade_assign with
    an integer enum (0=A, 1=None, 2=B)."""
    port = _free_port()
    received: list[tuple[str, tuple[Any, ...]]] = []
    listener = AbletonOSCListener(port=port)
    listener.on_any(lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = AbletonOSCClient(port=port)
        client.set_track_crossfade_assign(3, client.CROSSFADE_A)
        client.set_track_crossfade_assign(4, client.CROSSFADE_B)
        client.set_track_crossfade_assign(8, client.CROSSFADE_NONE)
        client.get_track_crossfade_assign(3)

        assert _wait_for(lambda: len(received) >= 4)
        by_addr_all = [(a, args) for a, args in received]
        sets = [args for a, args in by_addr_all if a == "/live/track/set/crossfade_assign"]
        assert (3, 0) in sets
        assert (4, 2) in sets
        assert (8, 1) in sets
        gets = [args for a, args in by_addr_all if a == "/live/track/get/crossfade_assign"]
        assert (3,) in gets
        # OSC must carry plain ints, not bools/floats.
        for _t, v in sets:
            assert type(v) is int
    finally:
        listener.stop()


def test_client_solo_listen_addresses():
    """start/stop_listen_solo emit the generic track listen verbs."""
    port = _free_port()
    received: list[tuple[str, tuple[Any, ...]]] = []
    listener = AbletonOSCListener(port=port)
    listener.on_any(lambda addr, args: received.append((addr, args)))
    listener.start()
    try:
        client = AbletonOSCClient(port=port)
        client.start_listen_solo(5)
        client.stop_listen_solo(5)
        assert _wait_for(lambda: len(received) >= 2)
        by_addr = {a: args for a, args in received}
        assert by_addr["/live/track/start_listen/solo"] == (5,)
        assert by_addr["/live/track/stop_listen/solo"] == (5,)
    finally:
        listener.stop()


def test_bridge_crossfade_assign_value_for_each_deck_kind():
    """A-side decks → A(0), B-side decks → B(2), mix-less / unsided → None(1).
    Mirrors dance.als.writer._crossfade_value_for."""
    assert AbletonBridge._crossfade_assign_for("drums_a") == AbletonOSCClient.CROSSFADE_A
    assert AbletonBridge._crossfade_assign_for("vocals_b") == AbletonOSCClient.CROSSFADE_B
    assert AbletonBridge._crossfade_assign_for("mix_a") == AbletonOSCClient.CROSSFADE_A
    assert AbletonBridge._crossfade_assign_for("mix_b") == AbletonOSCClient.CROSSFADE_B
    # A bare/unsided kind (defensive) → None group.
    assert AbletonBridge._crossfade_assign_for("mix") == AbletonOSCClient.CROSSFADE_NONE


def test_bridge_crossfade_matches_als_writer_semantics():
    """The bridge's live-load routing must agree with the static .als writer
    for EVERY deck-kind so a live-loaded deck blends identically to an export.
    """
    from dance.als import writer

    for deck_kind in AbletonBridge._DECK_KINDS:
        bridge_val = AbletonBridge._crossfade_assign_for(deck_kind)
        writer_val = int(writer._crossfade_value_for(deck_kind))
        assert bridge_val == writer_val, f"mismatch on {deck_kind}"


def test_bridge_create_deck_columns_assigns_crossfade_per_side():
    """_create_deck_columns sends /live/track/set/crossfade_assign for each of
    the 10 deck tracks: A-side → 0, B-side → 2, mix follows its side."""
    with _FakeLive(initial_names=[]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(_stub_track(), [], include_stems=False)
            assert _wait_for(
                lambda: live.count("/live/track/set/crossfade_assign") >= 10
            )
            assigns = {
                args[0]: args[1]
                for args in live.args_for("/live/track/set/crossfade_assign")
            }
            # Indices 0..9 in _DECK_KINDS order; A-side at even, B-side at odd
            # for the 8 stem decks, then mix_a (8) / mix_b (9).
            # drums_a=0 → A, drums_b=1 → B, ... mix_a=8 → A, mix_b=9 → B.
            assert assigns[0] == AbletonOSCClient.CROSSFADE_A   # drums_a
            assert assigns[1] == AbletonOSCClient.CROSSFADE_B   # drums_b
            assert assigns[8] == AbletonOSCClient.CROSSFADE_A   # mix_a
            assert assigns[9] == AbletonOSCClient.CROSSFADE_B   # mix_b
        finally:
            bridge.stop()


def test_bridge_soloed_kinds_derived_from_solo_pushes():
    """A Live /live/track/get/solo push for a deck-column track index lands in
    AbletonState.soloed_kinds as the matching deck-kind; clearing it removes
    the kind. Indices that aren't deck columns are ignored."""
    listen_port = _free_port()
    send_port = _free_port()
    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        # Pretend the decks were created at indices 0..9.
        bridge._deck_columns = {
            kind: i for i, kind in enumerate(_DECK_COLUMN_ORDER)
        }
        fake_live = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        # Solo on drums_a (idx 0) and mix_b (idx 9); a non-deck idx 99 too.
        fake_live.send_message("/live/track/get/solo", [0, 1])
        fake_live.send_message("/live/track/get/solo", [9, 1])
        fake_live.send_message("/live/track/get/solo", [99, 1])
        assert _wait_for(
            lambda: bridge.state.soloed_kinds == ["drums_a", "mix_b"]
        )
        # Clear drums_a → only mix_b remains.
        fake_live.send_message("/live/track/get/solo", [0, 0])
        assert _wait_for(lambda: bridge.state.soloed_kinds == ["mix_b"])
    finally:
        bridge.stop()


def test_bridge_soloed_kinds_stable_canonical_order():
    """soloed_kinds is emitted in _DECK_KINDS order regardless of the order
    solo pushes arrive."""
    listen_port = _free_port()
    send_port = _free_port()
    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        bridge._deck_columns = {
            kind: i for i, kind in enumerate(_DECK_COLUMN_ORDER)
        }
        fake_live = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        # Push in reverse-ish order: mix_b(9), bass_a(2), drums_a(0).
        fake_live.send_message("/live/track/get/solo", [9, 1])
        fake_live.send_message("/live/track/get/solo", [2, 1])
        fake_live.send_message("/live/track/get/solo", [0, 1])
        assert _wait_for(
            lambda: bridge.state.soloed_kinds == ["drums_a", "bass_a", "mix_b"]
        )
    finally:
        bridge.stop()


def test_bridge_deck_map_revision_bumps_on_load(tmp_path):
    """deck_map_revision increments on each push_track_to_live (a deck-cell
    mutation), so the FE knows to refetch the deck map."""
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")

    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            assert bridge.state.deck_map_revision == 0
            bridge.push_track_to_live(
                _stub_track(id=1), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            rev_after_load = bridge.state.deck_map_revision
            assert rev_after_load >= 1
            # A second load bumps again.
            bridge.push_track_to_live(
                _stub_track(id=2), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            assert bridge.state.deck_map_revision > rev_after_load
        finally:
            bridge.stop()


def test_bridge_deck_map_revision_bumps_on_delete_and_reset(tmp_path):
    """delete_cell and reset_deck_columns each bump the revision."""
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")

    with _FakeLive(initial_names=[f"Pre {i}" for i in range(10)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=1), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            rev = bridge.state.deck_map_revision
            # drums_a deck track index is 10 (base 10 + drums_a at offset 0).
            bridge.delete_cell(track_index=10, slot_index=0)
            assert bridge.state.deck_map_revision == rev + 1
            rev = bridge.state.deck_map_revision
            bridge.reset_deck_columns()
            assert bridge.state.deck_map_revision == rev + 1
        finally:
            bridge.stop()


def test_bridge_deck_map_revision_bumps_on_adopt():
    """adopt_cells (the resync path) bumps the revision."""
    listen_port = _free_port()
    send_port = _free_port()
    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        rev = bridge.state.deck_map_revision
        bridge.adopt_cells({(0, "drums_a"): 1, (0, "bass_a"): 1})
        assert bridge.state.deck_map_revision == rev + 1
    finally:
        bridge.stop()


def test_bridge_state_to_dict_includes_new_contract_fields():
    """to_dict carries soloed_kinds + deck_map_revision (and crossfader) so
    the WebSocket broadcast surfaces the full live-state contract."""
    import json

    state = AbletonState(tempo=120.0)
    state.soloed_kinds = ["drums_a", "mix_b"]
    state.deck_map_revision = 7
    state.crossfader = -0.25
    d = state.to_dict()
    json.dumps(d)  # JSON-safe
    assert d["soloed_kinds"] == ["drums_a", "mix_b"]
    assert d["deck_map_revision"] == 7
    assert d["crossfader"] == -0.25


# ---------------------------------------------------------------------------
# Robust create helper — lookup-after-create, never trust a prediction.
# Regression for the duplicate-deck + preview-leak bugs (both predicted an
# index from get_num_tracks BEFORE create_audio_track, then named/routed at
# the guessed index).
# ---------------------------------------------------------------------------

# Canonical prefixed deck names the bridge CREATES, in _DECK_KINDS order.
_PREFIXED_DECK_NAMES = [
    "Deck Drums A", "Deck Drums B",
    "Deck Bass A", "Deck Bass B",
    "Deck Vocals A", "Deck Vocals B",
    "Deck Other A", "Deck Other B",
    "Deck Mix A", "Deck Mix B",
]
_BARE_DECK_NAMES = [
    "Drums A", "Drums B",
    "Bass A", "Bass B",
    "Vocals A", "Vocals B",
    "Other A", "Other B",
    "Mix A", "Mix B",
]


def test_create_helper_returns_real_appended_index():
    """The robust helper reads the count, appends, confirms the +1 delta, and
    returns the REAL appended index (== count-before), then names it there."""
    with _FakeLive(initial_names=[f"Pre {i}" for i in range(5)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            idx = bridge._create_track_and_get_index("Cue", timeout=1.0)
            assert idx == 5  # appended after the 5 pre-existing tracks
            # It named the track at the real index, not a guess. Poll: the
            # fake records the rename on its listener thread.
            assert _wait_for(
                lambda: (5, "Cue")
                in [(a[0], a[1]) for a in live.args_for("/live/track/set/name")]
            )
            # The fake's track list reflects the rename at index 5.
            assert _wait_for(lambda: len(live.names) > 5 and live.names[5] == "Cue")
        finally:
            bridge.stop()


def test_create_helper_returns_none_on_timeout():
    """If Live never confirms the count went up (silent / stuck), the helper
    refuses to name a guessed index and returns None."""
    # A non-stateful fake that answers num_tracks with a STUCK count and
    # never grows on create — models a Live that dropped the create.
    listen_port = _free_port()
    send_port = _free_port()
    fake = AbletonOSCListener(port=send_port)
    reply = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
    fake.on(
        "/live/song/get/num_tracks",
        lambda _a, _args: reply.send_message("/live/song/get/num_tracks", [3]),
    )
    fake.start()
    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        # Count is stuck at 3 forever → never reaches before+1 → None.
        assert bridge._create_track_and_get_index("Cue", timeout=0.3) is None
    finally:
        bridge.stop()
        fake.stop()


def test_create_helper_returns_none_when_live_unreachable():
    """No reply to num_tracks at all → helper returns None (can't even read
    the starting count)."""
    bridge = AbletonBridge(send_port=_free_port(), listen_port=_free_port())
    bridge.start()
    try:
        assert bridge._create_track_and_get_index("Cue", timeout=0.1) is None
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# Bug 1 — preview must use a dedicated "Cue" track, never leak to a deck.
# ---------------------------------------------------------------------------


def test_preview_creates_and_routes_cue_track_at_real_index(tmp_path):
    """With no "Cue" track present, preview creates one at the REAL appended
    index, routes it to Ext. Out 3/4, and fires the clip THERE — not into a
    guessed/deck index. Reproduces the preview-leak fix."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF")

    # Session already has the 10 deck tracks (so a naive prediction would
    # collide with deck indices). Cue must land at index 10.
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.preview_audio(str(audio), label="PREVIEW x")
            assert result["ok"] is True
            cue_idx = result["cue_track_idx"]
            assert cue_idx == 10  # appended after the 10 deck tracks
            # The new track was named "Cue" at the real index.
            assert _wait_for(lambda: len(live.names) > 10 and live.names[10] == "Cue")
            # Output routed to the headphone cue bus (Ext. Out / 3/4).
            assert _wait_for(
                lambda: any(
                    a == (10, "Ext. Out")
                    for a in live.args_for("/live/track/set/output_routing_type")
                )
            )
            assert _wait_for(
                lambda: any(
                    a == (10, "3/4")
                    for a in live.args_for("/live/track/set/output_routing_channel")
                )
            )
            # The clip was fired on the Cue track (idx 10), NEVER a deck index.
            assert _wait_for(
                lambda: (10, bridge._CUE_SLOT) in live.args_for("/live/clip_slot/fire")
            )
            fires = live.args_for("/live/clip_slot/fire")
            deck_indices = set(range(10))  # the 10 deck tracks
            assert not any(t in deck_indices for t, _slot in fires), (
                f"preview fired into a deck track: {fires}"
            )
        finally:
            bridge.stop()


def test_preview_adopts_existing_cue_track_without_creating(tmp_path):
    """If a track named exactly "Cue" already exists, preview ADOPTS it (no
    create_audio_track) and fires there."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF")

    # Cue track already present at index 10, after the 10 decks.
    names = list(_PREFIXED_DECK_NAMES) + ["Cue"]
    with _FakeLive(initial_names=names) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.preview_audio(str(audio))
            assert result["ok"] is True
            assert result["cue_track_idx"] == 10
            # Adoption → zero new tracks created for the cue.
            assert live.count("/live/song/create_audio_track") == 0
            # Fired on the adopted Cue track only.
            assert _wait_for(
                lambda: (10, bridge._CUE_SLOT) in live.args_for("/live/clip_slot/fire")
            )
            assert not any(
                t < 10 for t, _slot in live.args_for("/live/clip_slot/fire")
            )
        finally:
            bridge.stop()


def test_preview_refuses_when_live_unreachable_no_random_fire(tmp_path):
    """When Live can't be reached (no Cue track confirmable), preview returns
    ok=False with a warning and fires NOTHING — never blasts a guessed index
    onto the master."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF")

    received: list[tuple[str, tuple]] = []
    # A fake on the bridge's send port that RECORDS commands but never
    # replies, so num_tracks/track_names time out (Live "unreachable").
    send_port = _free_port()
    cmd_listener = AbletonOSCListener(port=send_port)
    cmd_listener.on_any(lambda a, args: received.append((a, args)))
    cmd_listener.start()
    bridge = AbletonBridge(send_port=send_port, listen_port=_free_port())
    bridge.start()
    try:
        result = bridge.preview_audio(str(audio))
        assert result["ok"] is False
        assert result["cue_track_idx"] is None
        assert result["warnings"]
        # No clip was ever fired — the whole point of the fix.
        assert not any(a == "/live/clip_slot/fire" for a, _ in received)
    finally:
        bridge.stop()
        cmd_listener.stop()


def test_preview_does_not_fire_into_known_deck_index(tmp_path):
    """End-to-end guard: across create + adopt, the preview clip is never
    fired into any index the bridge knows to be a deck column."""
    audio = tmp_path / "drums.wav"
    audio.write_bytes(b"RIFF")
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            # Adopt the 10 deck columns so the bridge knows their indices.
            cols = bridge.recover_deck_columns(timeout=1.0)
            assert cols is not None
            deck_indices = set(cols.values())
            bridge.preview_audio(str(audio))
            assert _wait_for(lambda: bool(live.args_for("/live/clip_slot/fire")))
            for t, _slot in live.args_for("/live/clip_slot/fire"):
                assert t not in deck_indices, f"preview fired into deck idx {t}"
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Follow-up bug — create_audio_clip is async; preview must WAIT for the clip
# to exist before firing, else fire no-ops on an empty slot (clip loads but
# never plays). Also: force instant launch so previews aren't 1-bar-quantized.
# ---------------------------------------------------------------------------


def test_preview_waits_for_clip_to_exist_then_fires(tmp_path):
    """The fire_clip must be sent only AFTER Live confirms the cue clip
    exists. With has_clip_delay>0 the clip isn't present on the first query,
    so a correct bridge polls has_clip until it flips True, THEN fires."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF")
    # Clip only appears after 2 has_clip queries return False — models Live's
    # async create finishing a beat later.
    with _FakeLive(
        initial_names=list(_PREFIXED_DECK_NAMES), has_clip_delay=2
    ) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.preview_audio(str(audio), label="PREVIEW y")
            assert result["ok"] is True
            assert result["warnings"] == []  # confirmed cleanly, no warning
            cue_idx = result["cue_track_idx"]
            assert cue_idx == 10

            # The fire was sent, and it came AFTER at least one has_clip query
            # (i.e. the bridge waited — did not fire blindly right after
            # create_audio_clip). Poll for the fire (recorded on the fake's
            # listener thread, may lag the synchronous return).
            assert _wait_for(
                lambda: "/live/clip_slot/fire" in [a for a, _ in live.received]
            )
            order = [a for a, _ in live.received]
            fire_pos = order.index("/live/clip_slot/fire")
            create_pos = order.index("/live/clip_slot/create_audio_clip")
            has_clip_positions = [
                i for i, a in enumerate(order) if a == "/live/clip_slot/get/has_clip"
            ]
            assert has_clip_positions, "bridge never queried has_clip before firing"
            # create → has_clip(s) → fire, in that order.
            assert create_pos < has_clip_positions[0] < fire_pos
            # Fired on the cue slot specifically.
            assert (cue_idx, bridge._CUE_SLOT) in live.args_for("/live/clip_slot/fire")
        finally:
            bridge.stop()


def test_preview_sets_instant_launch_quant_before_fire(tmp_path):
    """The cue clip's launch_quantization is set to None BEFORE the fire, so
    previews fire instantly instead of waiting for the 1-bar global quantize.

    Live's enum is **0=Global, 1=None** (AbletonOSC README:394). This asserted
    0 and passed, so every preview really was bar-quantized — up to ~2 s of
    apparent dead air after pressing ▶, which reads as "the button is broken".
    The cue track is dedicated, so None is permanent there and needs no
    restore (unlike the deck-clip seek path)."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF")
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.preview_audio(str(audio))
            cue_idx = result["cue_track_idx"]
            # 1 = None (instant). NOT 0 — that is Global, i.e. 1 Bar.
            assert _wait_for(
                lambda: (cue_idx, bridge._CUE_SLOT, 1)
                in live.args_for("/live/clip/set/launch_quantization")
            )
            # It was set BEFORE the fire.
            assert _wait_for(
                lambda: "/live/clip_slot/fire" in [a for a, _ in live.received]
            )
            order = [a for a, _ in live.received]
            lq_pos = order.index("/live/clip/set/launch_quantization")
            fire_pos = order.index("/live/clip_slot/fire")
            assert lq_pos < fire_pos
        finally:
            bridge.stop()


def test_preview_times_out_then_best_effort_fires_with_warning(tmp_path):
    """If the clip never appears (failed/slow create) but Live IS answering
    has_clip with False, the bridge fires best-effort after the timeout and
    appends a warning — it does not hang forever or silently skip the fire."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF")
    with _FakeLive(
        initial_names=list(_PREFIXED_DECK_NAMES), clip_never_appears=True
    ) as live:
        bridge = live.make_bridge()
        bridge.start()
        # Shrink both timeouts so the test is fast (clip never appears → both
        # the has_clip confirm and the is_playing re-fire loop time out).
        bridge._CUE_CREATE_CONFIRM_TIMEOUT = 0.4
        bridge._CUE_PLAY_CONFIRM_TIMEOUT = 0.4
        try:
            result = bridge.preview_audio(str(audio))
            assert result["ok"] is True  # best-effort fire still "ok"
            assert any("not confirmed present" in w for w in result["warnings"])
            # Fire WAS still sent (best-effort) despite no confirmation.
            assert _wait_for(
                lambda: (result["cue_track_idx"], bridge._CUE_SLOT)
                in live.args_for("/live/clip_slot/fire")
            )
        finally:
            bridge.stop()


def test_preview_best_effort_fires_when_has_clip_unanswered(tmp_path):
    """If the running AbletonOSC fork doesn't answer has_clip at all, the
    bridge can't confirm — it fires best-effort with a clear warning rather
    than hanging or refusing."""
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF")
    with _FakeLive(
        initial_names=list(_PREFIXED_DECK_NAMES), answer_has_clip=False
    ) as live:
        bridge = live.make_bridge()
        bridge.start()
        bridge._CUE_CREATE_CONFIRM_TIMEOUT = 0.4
        bridge._CUE_PLAY_CONFIRM_TIMEOUT = 0.4
        try:
            result = bridge.preview_audio(str(audio))
            assert result["ok"] is True
            assert any("Could not confirm" in w for w in result["warnings"])
            assert _wait_for(
                lambda: (result["cue_track_idx"], bridge._CUE_SLOT)
                in live.args_for("/live/clip_slot/fire")
            )
        finally:
            bridge.stop()


def test_preview_refires_until_clip_actually_plays(tmp_path):
    """has_clip can flip True before a compressed sample finishes decoding, so
    the first fire plays silence. The bridge must keep re-firing until Live
    reports is_playing — modelled here by is_playing_delay (the clip exists
    immediately but reports not-playing for the first 2 polls)."""
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"ID3")
    with _FakeLive(
        initial_names=list(_PREFIXED_DECK_NAMES), is_playing_delay=2
    ) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.preview_audio(str(audio), label="PREVIEW song")
            assert result["ok"] is True
            # It confirmed playing within the window, so no warning.
            assert result["warnings"] == []
            cue_idx = result["cue_track_idx"]
            # More than one fire was sent (initial + re-fires until playing).
            fires = live.count("/live/clip_slot/fire")
            assert fires >= 2, f"expected re-fires until playing, got {fires}"
            assert bridge._clip_is_playing(cue_idx, bridge._CUE_SLOT) is True
        finally:
            bridge.stop()


def test_cue_play_timeout_is_longer_for_compressed_sources():
    """A cold mp3 decodes lazily after the clip object exists, so it needs a
    longer play-confirm ceiling than an instant PCM stem — otherwise the
    re-fire loop gives up and a song/mix preview sits frozen at 0. PCM
    (wav/aiff) keeps the short ceiling so a genuine failure surfaces fast."""
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        short = bridge._CUE_PLAY_CONFIRM_TIMEOUT
        long = bridge._CUE_PLAY_CONFIRM_TIMEOUT_COMPRESSED
        assert long > short
        for pcm in ["drums.wav", "x.AIFF", "stem.aif"]:
            assert bridge._cue_play_timeout_for(Path(pcm)) == short
        for comp in ["song.mp3", "x.m4a", "x.flac", "x.ogg", "x.opus", "x.aac"]:
            assert bridge._cue_play_timeout_for(Path(comp)) == long


def test_preview_subscribes_cue_clip_position_for_playhead(tmp_path):
    """After a preview fires, the bridge subscribes the cue clip's
    playing_position so the companion's preview waveform gets a REAL playhead
    (track_index → beats in AbletonState.playing_positions) instead of falling
    back to the master beat. stop_preview unsubscribes and clears the ghost."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.preview_audio(str(audio), label="PREVIEW song")
            assert result["ok"] is True
            cue_idx = result["cue_track_idx"]

            # The preview clip is unwarped (so long mp3s fire fast + audition at
            # native tempo); position + seek are then in seconds, not beats.
            assert _wait_for(
                lambda: any(
                    int(a[0]) == cue_idx
                    and int(a[1]) == bridge._CUE_SLOT
                    and int(a[2]) == 0
                    for a in live.args_for("/live/clip/set/warping")
                )
            )

            # The cue slot's playing_position was subscribed.
            assert _wait_for(
                lambda: (cue_idx, bridge._CUE_SLOT)
                in [
                    (int(a[0]), int(a[1]))
                    for a in live.args_for(
                        "/live/clip/start_listen/playing_position"
                    )
                ]
            )

            # A pushed position lands in state.playing_positions (what the FE
            # reads to place the playhead).
            bridge._on_playing_position(
                "/live/clip/get/playing_position",
                (cue_idx, bridge._CUE_SLOT, 12.0),
            )
            assert bridge.state.playing_positions.get(cue_idx) == 12.0

            # Stopping unsubscribes and drops the stale playhead.
            bridge.stop_preview()
            assert _wait_for(
                lambda: (cue_idx, bridge._CUE_SLOT)
                in [
                    (int(a[0]), int(a[1]))
                    for a in live.args_for(
                        "/live/clip/stop_listen/playing_position"
                    )
                ]
            )
            assert cue_idx not in bridge.state.playing_positions
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Bug 2 — live-loading stems must ADOPT existing deck columns, never spawn a
# duplicate set. Idempotency across re-loads + restarts + opened .als exports.
# ---------------------------------------------------------------------------


def test_load_adopts_existing_prefixed_decks_creates_zero_tracks(tmp_path):
    """When all 10 "Deck …" columns already exist (prior load / restart), a
    load REUSES them and creates ZERO new tracks."""
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.push_track_to_live(
                _stub_track(id=1), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            # Adopted the existing decks at their real indices 0..9.
            assert result["track_indices"]["drums_a"] == 0
            assert result["track_indices"]["mix_b"] == 9
            # ZERO creates — the columns were adopted, not recreated.
            assert live.count("/live/song/create_audio_track") == 0
            # Total tracks unchanged (no duplicate "Deck …" set).
            assert len(live.names) == 10
            # Clip loaded into the adopted drums_a column (index 0).
            assert _wait_for(
                lambda: any(
                    a[0] == 0 for a in live.args_for("/live/clip_slot/create_audio_clip")
                )
            )
        finally:
            bridge.stop()


def test_load_adopts_existing_bare_als_decks_creates_zero_tracks(tmp_path):
    """Opening a static .als export yields BARE deck names ("Drums A" … "Mix
    B"). A load must adopt those too and create nothing."""
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    with _FakeLive(initial_names=list(_BARE_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.push_track_to_live(
                _stub_track(id=1), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            assert result["track_indices"]["drums_a"] == 0
            assert result["track_indices"]["mix_b"] == 9
            assert live.count("/live/song/create_audio_track") == 0
            assert len(live.names) == 10
        finally:
            bridge.stop()


def test_two_loads_do_not_duplicate_decks(tmp_path):
    """A second load (and a fresh bridge against the same session, modelling a
    backend restart) must NOT spawn a second set of "Deck …" tracks."""
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    # Start from a CLEAN session — the first load provisions the 10 decks.
    with _FakeLive(initial_names=[]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=1), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            assert live.count("/live/song/create_audio_track") == 10
            assert len(live.names) == 10
            # Second load on the same bridge — cached columns reused, no creates.
            bridge.push_track_to_live(
                _stub_track(id=2), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            assert live.count("/live/song/create_audio_track") == 10  # still 10
            assert len(live.names) == 10  # no duplicate set

            # Simulate a backend restart: brand-new bridge, same Live session
            # (the 10 deck tracks persist in `live.names`). Its first load
            # must ADOPT, not recreate.
            bridge.stop()
            bridge2 = live.make_bridge()
            bridge2.start()
            bridge2.push_track_to_live(
                _stub_track(id=3), [_stub_stem("drums", str(drums))], kinds=["drums"], side="a"
            )
            assert live.count("/live/song/create_audio_track") == 10  # STILL 10
            assert len(live.names) == 10  # no second "Deck …" set
            bridge2.stop()
        finally:
            pass


def test_create_deck_columns_fills_only_missing_columns(tmp_path):
    """Partial layout (some deck columns present, some missing) → adopt the
    present ones, create ONLY the missing ones."""
    # Only the 8 stem decks exist; both mix columns are missing.
    partial = list(_PREFIXED_DECK_NAMES[:8])  # drums..other A/B, no mixes
    with _FakeLive(initial_names=partial) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            cols = bridge._create_deck_columns(start_index=len(live.names))
            # The 8 existing stem decks adopted at 0..7.
            assert cols["drums_a"] == 0
            assert cols["other_b"] == 7
            # The 2 missing mix columns created → appended at 8 and 9.
            assert cols["mix_a"] == 8
            assert cols["mix_b"] == 9
            # Exactly 2 creates (just the missing mix columns).
            assert live.count("/live/song/create_audio_track") == 2
            assert _wait_for(
                lambda: len(live.names) > 9
                and live.names[8] == "Deck Mix A"
                and live.names[9] == "Deck Mix B"
            )
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Recording — the button captured nothing at all before this
# ---------------------------------------------------------------------------


def test_record_on_arms_a_resampling_recorder_track():
    """Record must ARM something, or Live captures silence.

    The old endpoint was a bare ``record_mode`` toggle. Live records what
    armed tracks hear; nothing in this project was ever armed or given an
    input source, so there was no path from the button to audio at all —
    which matters because "record every session and listen back" is the only
    honest feedback signal in a bedroom.

    Verified against real Live: the created track reads back name='Recorder',
    arm=True, input_routing_type='Resampling', current_monitoring_state=2.
    """
    with _FakeLive(initial_names=[f"Pre {i}" for i in range(3)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.set_record(True)
        finally:
            bridge.stop()

        assert result["ok"] is True
        assert result["recording"] is True
        idx = result["armed_track"]
        assert idx is not None, "record must confirm an armed track"

        assert _wait_for(
            lambda: (idx, "Resampling") in live.args_for("/live/track/set/input_routing_type")
        ), "recorder input must be Resampling (the master output)"
        # Monitoring OFF, or the capture is fed back into the master.
        assert _wait_for(lambda: (idx, 2) in live.args_for("/live/track/set/current_monitoring_state"))
        assert _wait_for(lambda: (idx, 1) in live.args_for("/live/track/set/arm"))
        assert _wait_for(lambda: (1,) in live.args_for("/live/song/set/record_mode"))

        # The arming must precede the record toggle — arming after the fact
        # would lose the head of the take.
        order = [a for a, _ in live.received]
        assert order.index("/live/track/set/arm") < order.index("/live/song/set/record_mode")


def test_record_on_adopts_an_existing_recorder_track():
    """Idempotent across a reopened Set — never stack up Recorder tracks."""
    with _FakeLive(initial_names=["Pre 0", "Recorder", "Pre 2"]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.set_record(True)
        finally:
            bridge.stop()
        assert result["armed_track"] == 1
        assert live.count("/live/song/create_audio_track") == 0


def test_record_off_does_not_provision_anything():
    """Stopping must not create or arm tracks."""
    with _FakeLive(initial_names=[f"Pre {i}" for i in range(3)]) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            result = bridge.set_record(False)
        finally:
            bridge.stop()
        assert result["recording"] is False
        assert result["armed_track"] is None
        assert live.count("/live/song/create_audio_track") == 0
        assert _wait_for(lambda: (0,) in live.args_for("/live/song/set/record_mode"))


def test_adopted_deck_columns_are_styled_not_just_created_ones():
    """Adopting existing deck columns must still mute the mix tracks.

    Styling used to live only on the CREATE path, so the full-adopt early
    return skipped the mix mute, the crossfade assignment and the colours —
    and full adopt is exactly the branch the documented workflow takes, since
    an exported .als already contains all ten named deck columns.

    An exported Set does not double on its own (its mix clips carry
    Disabled=true). The break is one step later: push_track_to_live drops a
    NEW, enabled mix clip into mix_a on every whole-song load, reasoning that
    "the mix track is muted at creation" — true only on the create path. So
    adopted + live-loaded plays the original full track over its own stems.
    """
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES)) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            live.received.clear()
            columns = bridge._create_deck_columns(start_index=0)
        finally:
            bridge.stop()

        # Nothing was created — this is the pure adopt path.
        assert live.count("/live/song/create_audio_track") == 0
        mix_a, mix_b = columns["mix_a"], columns["mix_b"]
        assert _wait_for(
            lambda: (mix_a, 1) in live.args_for("/live/track/set/mute")
        ), "adopted mix_a was left UNMUTED — it will double the stems"
        assert _wait_for(lambda: (mix_b, 1) in live.args_for("/live/track/set/mute"))
        # Stem decks must NOT be muted.
        drums_a = columns["drums_a"]
        assert (drums_a, 1) not in live.args_for("/live/track/set/mute")
        # Crossfade assignment is applied to adopted columns too.
        assert _wait_for(
            lambda: any(a[0] == drums_a for a in live.args_for("/live/track/set/crossfade_assign"))
        )


def test_load_past_the_last_scene_grows_the_set(tmp_path):
    """The 9th load onto a side must not aim at a slot that doesn't exist.

    Verified against real Live: creating a clip beyond the last scene raises
    "IndexError: Index out of range" internally, no clip appears, and because
    OSC carries no error reply the bridge recorded the cell anyway — the grid
    showed a loaded cell that was not there and firing it did nothing.
    `create_scene` existed on the client and was called from nowhere.
    """
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"RIFF")

    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES), num_scenes=8) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=7, file_path=str(mix)),
                [_stub_stem("drums", str(drums))],
                kinds=["drums"],
                side="a",
                scene_index=9,          # scenes 0..7 exist; 9 is two past the end
            )
        finally:
            bridge.stop()

        assert live.num_scenes >= 10, (
            f"Set was not grown to hold scene index 9 (num_scenes={live.num_scenes})"
        )
        assert live.count("/live/song/create_scene") >= 2


def test_load_within_the_existing_scenes_creates_none(tmp_path):
    """Don't grow the Set for a load that already fits — a DJ's scene list
    should not sprout empty rows on every load."""
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"RIFF")
    with _FakeLive(initial_names=list(_PREFIXED_DECK_NAMES), num_scenes=8) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.push_track_to_live(
                _stub_track(id=7, file_path=str(mix)),
                [_stub_stem("drums", str(drums))],
                kinds=["drums"], side="a", scene_index=3,
            )
        finally:
            bridge.stop()
        assert live.count("/live/song/create_scene") == 0
        assert live.num_scenes == 8


def test_stale_persisted_cue_index_is_discarded_not_trusted(tmp_path, monkeypatch):
    """A Cue index restored from disk must be confirmed against the CURRENT
    Live set before anything is fired into it.

    Getting this wrong is destructive, not just wrong: preview_audio deletes
    whatever clip occupies (cue_idx, _CUE_SLOT) before writing its own. A
    stale index pointing at a deck column would delete a loaded stem at scene
    1 and then play the preview through the MASTER — the exact "leaked
    previews into a deck track" failure _ensure_cue_track's docstring says it
    exists to prevent, arriving from disk instead of from a prediction.
    """
    names = list(_PREFIXED_DECK_NAMES) + ["Cue"]
    with _FakeLive(initial_names=names) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            # Simulate a restore from a PREVIOUS Live set: the Cue really
            # lives at index 10 here, but disk says 0 (a deck column).
            bridge._cue_track_idx = 0
            bridge._cue_idx_verified = False
            idx = bridge._ensure_cue_track()
        finally:
            bridge.stop()

        assert idx != 0, "fired into a deck column on a stale cached index"
        assert idx == names.index("Cue")


def test_verified_cue_index_is_reused_without_a_extra_lookup():
    """Once confirmed, don't pay a track-names roundtrip on every preview."""
    names = list(_PREFIXED_DECK_NAMES) + ["Cue"]
    with _FakeLive(initial_names=names) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            first = bridge._ensure_cue_track()
            before = live.count("/live/song/get/track_names")
            second = bridge._ensure_cue_track()
            after = live.count("/live/song/get/track_names")
        finally:
            bridge.stop()
        assert first == second == names.index("Cue")
        assert after == before, "re-verified an already-confirmed index"


def test_cue_index_refuses_to_guess_when_live_is_silent():
    """Unverified index + unreachable Live must return None, not the guess."""
    bridge = AbletonBridge(send_port=_free_port(), listen_port=_free_port())
    bridge._cue_track_idx = 4
    bridge._cue_idx_verified = False
    assert bridge._ensure_cue_track(num_tracks_timeout=0.05) is None


def test_stale_deck_columns_are_rederived_when_live_grew(tmp_path):
    """Cached deck-column indices must be checked by NAME, not by whether
    they still fit.

    The old test was `cached_max >= base` — re-derive only when Live SHRANK.
    If Live grew, or the user opened a different Set, or tracks were
    reordered, the stale map survived and clips were written to whatever now
    occupies those indices, potentially the user's own tracks. Growth is the
    normal case here: the bridge itself adds a Cue track and a Recorder track
    beyond the ten deck columns.
    """
    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"RIFF")

    # Live's layout: three of the user's own tracks FIRST, deck columns after.
    names = ["My Vocals", "My Bass", "My Perc"] + list(_PREFIXED_DECK_NAMES)
    with _FakeLive(initial_names=names) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            # Simulate a restore describing a PREVIOUS Set where the deck
            # columns were the first ten tracks.
            bridge._deck_columns = {
                k: i for i, k in enumerate(bridge._DECK_KINDS)
            }
            bridge._deck_columns_verified = False
            live.received.clear()
            bridge.push_track_to_live(
                _stub_track(id=7, file_path=str(mix)),
                [_stub_stem("drums", str(drums))],
                kinds=["drums"], side="a",
            )
        finally:
            bridge.stop()

        drums_a = bridge._deck_columns["drums_a"]
        assert drums_a >= 3, (
            f"drums_a resolved to {drums_a} — that is one of the user's own "
            f"tracks, not a deck column"
        )
        assert names[drums_a] in bridge._DECK_RECOVERY_NAMES["drums_a"]
        # And the clip really went to the corrected index.
        created = [a for a in live.args_for("/live/clip_slot/create_audio_clip")]
        assert any(a[0] == drums_a for a in created)


def test_deck_columns_are_left_alone_when_live_is_silent(tmp_path):
    """An unverifiable map must not be churned on a guess — the offline
    best-effort path other tests depend on."""
    bridge = AbletonBridge(send_port=_free_port(), listen_port=_free_port())
    bridge._deck_columns = {k: i for i, k in enumerate(bridge._DECK_KINDS)}
    assert bridge._deck_columns_match_live(timeout=0.05) is True


def test_clean_live_decks_also_removes_the_recorder_track():
    """The Recorder must be cleaned along with the decks and the Cue.

    It is appended AFTER the ten deck columns, so leaving it behind makes it
    the lowest-indexed survivor of a clean: it slides to index 0 and the decks
    recreated afterwards land at 1-10. The APC40's session ring still starts
    at track 0, so fader 1 would control a silent Resampling track and the
    last stem column would fall outside the ring entirely.
    """
    names = list(_PREFIXED_DECK_NAMES) + ["Cue", "Recorder", "My Own Track"]
    with _FakeLive(initial_names=names) as live:
        bridge = live.make_bridge()
        bridge.start()
        try:
            bridge.clean_live_decks(timeout=1.0)
        finally:
            bridge.stop()
        deleted = {a[0] for a in live.args_for("/live/song/delete_track")}
        assert names.index("Recorder") in deleted, "Recorder survived the clean"
        assert names.index("Cue") in deleted
        # The user's own track is never touched.
        assert names.index("My Own Track") not in deleted
