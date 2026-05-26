"""OSC bridge tests — use real UDP loopback so we exercise the network code,
not a mock. Each test gets its own free port via ``port=0``.
"""

from __future__ import annotations

import socket
import time
from typing import Any

import pytest
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
        names = [
            args[1] for addr, args in received if addr == "/live/track/set/name"
        ]
        assert len(creates) == 10
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
