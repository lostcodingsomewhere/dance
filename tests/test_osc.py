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


def test_bridge_state_to_dict_is_json_safe():
    state = AbletonState(tempo=128.0, is_playing=True, beat=4.5)
    state.playing_clips[0] = 2
    state.track_volumes[1] = 0.8

    d = state.to_dict()
    import json
    json.dumps(d)  # would raise if non-JSON-safe
    assert d["tempo"] == 128.0
    assert d["playing_clips"] == {0: 2}
