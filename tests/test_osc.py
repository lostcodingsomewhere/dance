"""OSC bridge tests — use real UDP loopback so we exercise the network code,
not a mock. Each test gets its own free port via ``port=0``.
"""

from __future__ import annotations

import socket
import time
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
    default 8-strip view maps 1:1 to the stem decks)."""
    listen_port = _free_port()
    send_port = _free_port()

    # Fake Live replies "10 existing tracks" so the new ones land at 10..14.
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
    received: list[tuple[str, tuple[Any, ...]]] = []

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [10])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.on_any(lambda addr, args: received.append((addr, args)))
    fake_live_listener.start()

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
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

        # We expect 10 create_audio_track messages.
        assert _wait_for(
            lambda: sum(
                1 for a, _ in received if a == "/live/song/create_audio_track"
            )
            >= 10
        )
        creates = [a for a, _ in received if a == "/live/song/create_audio_track"]
        assert len(creates) == 10
        # Wait for all 10 name messages — they race with create_audio_track
        # and may not all be in the buffer when the count check passes.
        assert _wait_for(
            lambda: sum(
                1 for addr, _ in received if addr == "/live/track/set/name"
            )
            >= 10
        )
        names = [
            args[1] for addr, args in received if addr == "/live/track/set/name"
        ]
        # Names include each stem role × side and both mix references.
        joined = " | ".join(names)
        assert "Drums A" in joined and "Drums B" in joined
        assert "Bass A" in joined and "Bass B" in joined
        assert "Vocals A" in joined and "Vocals B" in joined
        assert "Other A" in joined and "Other B" in joined
        assert "Mix A" in joined and "Mix B" in joined
        # Both mix tracks are muted on creation (reference / parachute,
        # not double-summed audio). mix_a is at idx 18, mix_b at idx 19.
        # Wait for the mute messages — they're sent during track creation,
        # which races with the create_audio_track count check above.
        def saw_both_mutes() -> bool:
            mutes = [
                args for addr, args in received
                if addr == "/live/track/set/mute"
            ]
            return (18, 1) in mutes and (19, 1) in mutes
        assert _wait_for(saw_both_mutes)
    finally:
        bridge.stop()
        fake_live_listener.stop()


def test_bridge_push_track_to_live_loads_mix_cell_on_full_song(tmp_path):
    """A whole-song load also drops the original mix file into the SONG
    cell. Without this the UI's SceneGrid renders the SONG column as empty
    after every load — looks broken, even though the stems were loaded.
    """
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
    received: list[tuple[str, tuple[Any, ...]]] = []

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [10])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.on_any(lambda a, args: received.append((a, args)))
    fake_live_listener.start()

    # Real on-disk files so the existence guards pass.
    mix_path = tmp_path / "song.wav"
    mix_path.write_bytes(b"RIFF")
    drums_path = tmp_path / "drums.wav"
    drums_path.write_bytes(b"RIFF")

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        track = _stub_track(id=7, title="My Song", file_path=str(mix_path))
        stems = [_stub_stem("drums", str(drums_path))]
        bridge.push_track_to_live(track, stems, include_stems=True)

        # mix_a is at idx 18 (8 stem decks + mix_a). _pick_side picks A
        # for the empty row, so the source mix file lands on mix_a not
        # mix_b. Re-scan ``received`` each poll so we see messages that
        # arrive after the assert is set up.
        def saw_mix_clip() -> bool:
            return any(
                addr == "/live/clip_slot/create_audio_clip"
                and args[0] == 18
                and args[2] == str(mix_path)
                for addr, args in received
            )
        assert _wait_for(saw_mix_clip)
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
        fake_live_listener.stop()


def test_bridge_push_track_to_live_skips_mix_on_single_stem_load(tmp_path):
    """Single-stem loads (kinds=['drums']) must NOT drop the mix file —
    the mix cell stays empty so we don't fight the user's remix in progress.
    """
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [10])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.start()

    mix_path = tmp_path / "song.wav"
    mix_path.write_bytes(b"RIFF")
    drums_path = tmp_path / "drums.wav"
    drums_path.write_bytes(b"RIFF")

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
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
        fake_live_listener.stop()


def test_bridge_promote_forced_side_enqueues_not_clobbers(tmp_path):
    """Promoting a rec onto a side whose cell is already loaded (and maybe
    playing) must enqueue on the next free scene of THAT side — never
    overwrite the occupied cell. Reproduces the promote-clobber bug."""
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [10])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.start()

    drums1 = tmp_path / "drums1.wav"
    drums1.write_bytes(b"RIFF")
    drums2 = tmp_path / "drums2.wav"
    drums2.write_bytes(b"RIFF")

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        # First drums stem live-loaded onto Deck A at scene 0 (now "playing").
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
    finally:
        bridge.stop()
        fake_live_listener.stop()


def test_bridge_promote_auto_side_uses_free_side_at_scene_zero(tmp_path):
    """With Deck A busy at scene 0 and no forced side, a promote should fill
    the free B side at the SAME scene rather than overwriting A or skipping
    to a fresh scene."""
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [10])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.start()

    drums1 = tmp_path / "drums1.wav"
    drums1.write_bytes(b"RIFF")
    drums2 = tmp_path / "drums2.wav"
    drums2.write_bytes(b"RIFF")

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
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
        fake_live_listener.stop()


def test_bridge_push_track_to_live_include_stems_false_loads_no_cells():
    """include_stems=False loads zero cells but still provisions the 10
    reusable deck columns (8 stem decks A/B × 4 roles + mix_a + mix_b)
    so future loads have somewhere to land."""
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
    received: list[tuple[str, tuple[Any, ...]]] = []

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [0])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.on_any(lambda a, args: received.append((a, args)))
    fake_live_listener.start()

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
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
        # But nothing actually loaded since kinds resolved to [].
        assert result["stems_loaded"] == 0
        assert _wait_for(
            lambda: sum(
                1 for a, _ in received if a == "/live/song/create_audio_track"
            )
            == 10
        )
    finally:
        bridge.stop()
        fake_live_listener.stop()


def test_bridge_push_track_to_live_records_warning_for_missing_stem_file():
    """If a stem path doesn't exist on disk, we don't crash — we warn."""
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [0])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.start()

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
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
        fake_live_listener.stop()


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
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
    received: list[tuple[str, tuple[Any, ...]]] = []

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [0])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.on_any(lambda a, args: received.append((a, args)))
    fake_live_listener.start()

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
    bridge.start()
    try:
        bridge.push_track_to_live(_stub_track(), [], include_stems=False)
        assert _wait_for(
            lambda: sum(
                1 for a, _ in received if a == "/live/track/set/crossfade_assign"
            )
            >= 10
        )
        assigns = {
            args[0]: args[1]
            for a, args in received
            if a == "/live/track/set/crossfade_assign"
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
        fake_live_listener.stop()


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
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [10])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.start()

    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
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
        fake_live_listener.stop()


def test_bridge_deck_map_revision_bumps_on_delete_and_reset(tmp_path):
    """delete_cell and reset_deck_columns each bump the revision."""
    listen_port = _free_port()
    send_port = _free_port()
    fake_live_listener = AbletonOSCListener(port=send_port)
    reply_client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)

    def on_query(_addr, _args):
        reply_client.send_message("/live/song/get/num_tracks", [10])

    fake_live_listener.on("/live/song/get/num_tracks", on_query)
    fake_live_listener.start()

    drums = tmp_path / "drums.wav"
    drums.write_bytes(b"RIFF")

    bridge = AbletonBridge(send_port=send_port, listen_port=listen_port)
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
        fake_live_listener.stop()


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
