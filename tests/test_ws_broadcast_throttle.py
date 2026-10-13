"""The /ws AbletonState push is coalesced to a bounded rate, last state wins.

The OSC bridge calls its listeners once per inbound OSC message (clip
positions, meters, beat) — hundreds of times a second mid-set. These tests pin
the contract of ``LatestValueThrottle`` and its wiring into ``WSManager`` /
``create_app``:

- bursts collapse to at most ``max_hz`` emits per second,
- the LAST value of a burst is always emitted (trailing edge),
- a change after a quiet spell is emitted immediately (leading edge),
- values are never emitted out of order,
- the payload that reaches the browser is still the full AbletonState dict,
- the job-event channel stays unthrottled (every event is a distinct job).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from dance.api import create_app
from dance.api.broadcast_throttle import STATE_BROADCAST_MAX_HZ, LatestValueThrottle
from dance.api.routers.ws import WSManager
from dance.osc.bridge import AbletonState

# ---------------------------------------------------------------------------
# Deterministic harness: a fake clock + a fake event loop that runs callbacks
# in due-time order. No real time passes, so the bounds are exact.
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeLoop:
    """The two loop methods the throttle uses, driven by ``_FakeClock``."""

    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self._seq = itertools.count()
        self._queue: list[tuple[float, int, Callable[..., None], tuple[Any, ...]]] = []
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    def call_soon_threadsafe(self, cb: Callable[..., None], *args: Any) -> None:
        if self.closed:
            raise RuntimeError("Event loop is closed")
        heapq.heappush(self._queue, (self.clock.now, next(self._seq), cb, args))

    def call_later(self, delay: float, cb: Callable[..., None], *args: Any) -> None:
        heapq.heappush(self._queue, (self.clock.now + delay, next(self._seq), cb, args))

    def run_until(self, t: float) -> None:
        """Run every callback due at or before ``t``, advancing the clock."""
        while self._queue and self._queue[0][0] <= t:
            due, _, cb, args = heapq.heappop(self._queue)
            self.clock.now = max(self.clock.now, due)
            cb(*args)
        self.clock.now = max(self.clock.now, t)

    def drain(self) -> None:
        while self._queue:
            self.run_until(self._queue[0][0])


def _harness(max_hz: float = 30.0):
    clock = _FakeClock()
    loop = _FakeLoop(clock)
    emitted: list[tuple[float, int]] = []
    throttle: LatestValueThrottle[int] = LatestValueThrottle(
        lambda v: emitted.append((clock.now, v)), max_hz=max_hz, clock=clock
    )
    return clock, loop, throttle, emitted


def test_burst_is_bounded_and_ends_on_the_last_value() -> None:
    clock, loop, throttle, emitted = _harness(max_hz=30.0)

    # 1 kHz producer for one simulated second — e.g. 8 decks x ~60 Hz meters
    # and clip positions, plus the beat.
    for i in range(1000):
        loop.run_until(i / 1000)
        throttle.submit_threadsafe(i, loop)  # type: ignore[arg-type]
    loop.drain()

    values = [v for _, v in emitted]
    # One leading emit + at most one per interval over the second. Without
    # the throttle this would be 1000.
    assert len(values) <= 30 + 1
    assert len(values) >= 25  # still a steady stream, not a single late emit
    assert values[-1] == 999  # trailing edge: the final state always lands
    assert values == sorted(values)  # never an older state after a newer one
    times = [t for t, _ in emitted]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert min(gaps) >= throttle.interval - 1e-9


def test_lone_change_after_quiet_goes_out_without_delay() -> None:
    clock, loop, throttle, emitted = _harness()

    throttle.submit_threadsafe(1, loop)  # type: ignore[arg-type]
    loop.run_until(0.0)
    assert emitted == [(0.0, 1)]

    # Long after: no waiting out an interval that has already passed.
    loop.run_until(5.0)
    throttle.submit_threadsafe(2, loop)  # type: ignore[arg-type]
    loop.run_until(5.0)
    assert emitted[-1] == (5.0, 2)


def test_change_inside_the_window_waits_for_the_trailing_edge() -> None:
    clock, loop, throttle, emitted = _harness(max_hz=10.0)  # 100 ms window

    throttle.submit_threadsafe(1, loop)  # type: ignore[arg-type]
    loop.run_until(0.0)
    loop.run_until(0.02)
    throttle.submit_threadsafe(2, loop)  # type: ignore[arg-type]
    throttle.submit_threadsafe(3, loop)  # type: ignore[arg-type]
    loop.run_until(0.099)
    assert [v for _, v in emitted] == [1]  # held back inside the window
    loop.run_until(0.1 + 1e-9)
    assert emitted[-1] == (pytest.approx(0.1), 3)  # 2 was superseded, 3 lands
    loop.drain()
    assert [v for _, v in emitted] == [1, 3]  # nothing left over, no repeat


def test_no_emit_without_a_new_value() -> None:
    clock, loop, throttle, emitted = _harness()
    throttle.submit_threadsafe(7, loop)  # type: ignore[arg-type]
    loop.drain()
    loop.run_until(10.0)
    assert [v for _, v in emitted] == [7]


def test_rearms_on_a_new_loop_when_the_old_one_closed() -> None:
    """An app whose lifespan restarted must not stay muted forever."""
    clock = _FakeClock()
    old, new = _FakeLoop(clock), _FakeLoop(clock)
    emitted: list[int] = []
    throttle: LatestValueThrottle[int] = LatestValueThrottle(
        emitted.append, max_hz=30.0, clock=clock
    )
    throttle.submit_threadsafe(1, old)  # type: ignore[arg-type]
    old.closed = True  # flush never ran — the loop went away
    throttle.submit_threadsafe(2, new)  # type: ignore[arg-type]
    new.drain()
    assert emitted == [2]


def test_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        LatestValueThrottle(lambda _v: None, max_hz=0)


# ---------------------------------------------------------------------------
# Real asyncio loop, real producer thread — the production threading shape.
# ---------------------------------------------------------------------------


def test_threaded_producer_on_a_real_loop_is_bounded_and_ends_on_last() -> None:
    received: list[int] = []
    done = threading.Event()
    total = 3000

    async def main() -> float:
        loop = asyncio.get_running_loop()

        def emit(v: int) -> None:
            received.append(v)
            if v == total - 1:
                done.set()

        throttle: LatestValueThrottle[int] = LatestValueThrottle(emit, max_hz=30.0)

        def produce() -> None:
            for i in range(total):
                throttle.submit_threadsafe(i, loop)
                if i % 10 == 0:
                    time.sleep(0.001)  # ~ a few hundred Hz, like the OSC thread

        t0 = time.monotonic()
        producer = threading.Thread(target=produce)
        producer.start()
        while not done.is_set():
            await asyncio.sleep(0.005)
            assert time.monotonic() - t0 < 10, "final value never emitted"
        producer.join()
        return time.monotonic() - t0

    elapsed = asyncio.run(main())

    assert received[-1] == total - 1
    assert received == sorted(received)
    # At most one emit per 1/30 s window, plus the leading one. A loaded CI
    # box only makes this smaller, never larger.
    assert len(received) <= int(elapsed * 30.0) + 2
    assert len(received) < total / 10


# ---------------------------------------------------------------------------
# Wiring: create_app -> bridge listener -> WSManager -> /ws
# ---------------------------------------------------------------------------


class _BurstBridge:
    """Minimal bridge stand-in: a state object and a subscriber list."""

    def __init__(self) -> None:
        self.state = AbletonState(tempo=120.0, is_playing=True, beat=0.0)
        self._subscribers: list[Callable[[AbletonState], None]] = []

    def subscribe(self, listener: Callable[[AbletonState], None]) -> None:
        self._subscribers.append(listener)

    def start(self) -> None:  # pragma: no cover - lifespan only calls it for own bridges
        pass

    def stop(self) -> None:  # pragma: no cover
        pass

    def emit(self, **changes: Any) -> None:
        for k, v in changes.items():
            setattr(self.state, k, v)
        for sub in list(self._subscribers):
            sub(self.state)


@pytest.fixture
def burst_bridge() -> _BurstBridge:
    return _BurstBridge()


@pytest.fixture
def ws_client(session_factory: sessionmaker, burst_bridge: _BurstBridge, tmp_path):
    from dance.config import Settings

    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )
    app = create_app(settings=settings, bridge=burst_bridge, session_factory=session_factory)
    with TestClient(app) as c:
        yield c


def test_ws_state_burst_is_coalesced_and_final_state_arrives(
    ws_client: TestClient, burst_bridge: _BurstBridge
) -> None:
    total = 600
    frames: list[dict[str, Any]] = []
    finished = threading.Event()

    with ws_client.websocket_connect("/ws") as ws:
        initial = ws.receive_json()
        assert initial["beat"] == 0.0

        def receive() -> None:
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["beat"] == float(total - 1):
                    finished.set()
                    return

        rx = threading.Thread(target=receive, daemon=True)
        rx.start()

        # The OSC listener thread: one full-state broadcast per message.
        t0 = time.monotonic()

        def osc_thread() -> None:
            for i in range(total):
                burst_bridge.emit(
                    beat=float(i),
                    playing_positions={1: i / 4, 2: i / 4},
                    track_meters={1: (i % 10) / 10},
                )
                if i % 20 == 0:
                    time.sleep(0.001)

        producer = threading.Thread(target=osc_thread)
        producer.start()
        producer.join()
        assert finished.wait(5.0), "the final state never reached the client"
        elapsed = time.monotonic() - t0

    # Bounded: one leading frame + one per 1/30 s window. Pre-throttle this
    # was exactly ``total`` frames.
    assert len(frames) <= int(elapsed * STATE_BROADCAST_MAX_HZ) + 2
    assert len(frames) < total / 5
    # Newest state wins and the shape is unchanged — the full AbletonState.
    last = frames[-1]
    assert last["beat"] == float(total - 1)
    assert last["playing_positions"] == {"1": (total - 1) / 4, "2": (total - 1) / 4}
    assert set(last) == set(AbletonState().to_dict())
    beats = [f["beat"] for f in frames]
    assert beats == sorted(beats)


def test_ws_single_change_is_pushed_promptly(
    ws_client: TestClient, burst_bridge: _BurstBridge
) -> None:
    with ws_client.websocket_connect("/ws") as ws:
        ws.receive_json()

        def fire() -> None:
            time.sleep(0.05)
            burst_bridge.emit(tempo=128.0)

        threading.Thread(target=fire, daemon=True).start()
        t0 = time.monotonic()
        update = ws.receive_json()
        assert update["tempo"] == 128.0
        # Leading edge: no artificial 1/30 s hold on an isolated change
        # (generous bound — this only guards against a regression to a
        # fixed trailing-only delay of whole seconds).
        assert time.monotonic() - t0 < 2.0


# ---------------------------------------------------------------------------
# The unthrottled path (job events) still delivers every payload.
# ---------------------------------------------------------------------------


class _RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def test_plain_broadcast_threadsafe_is_not_coalesced() -> None:
    sock = _RecordingSocket()

    async def main() -> None:
        manager = WSManager()
        manager.loop = asyncio.get_running_loop()
        manager.connections.add(sock)  # type: ignore[arg-type]

        def produce() -> None:
            for i in range(50):
                manager.broadcast_threadsafe({"id": f"job-{i}"})

        t = threading.Thread(target=produce)
        t.start()
        t.join()
        for _ in range(200):
            if len(sock.sent) == 50:
                break
            await asyncio.sleep(0.01)

    asyncio.run(main())
    assert sorted(p["id"] for p in sock.sent) == sorted(f"job-{i}" for i in range(50))


def test_latest_broadcast_reaches_every_connection() -> None:
    socks = [_RecordingSocket() for _ in range(3)]

    async def main() -> None:
        manager = WSManager(state_max_hz=30.0)
        manager.loop = asyncio.get_running_loop()
        for s in socks:
            manager.connections.add(s)  # type: ignore[arg-type]
        for i in range(100):
            manager.broadcast_latest_threadsafe(lambda i=i: {"beat": i})
        for _ in range(200):
            if all(s.sent and s.sent[-1]["beat"] == 99 for s in socks):
                break
            await asyncio.sleep(0.01)

    asyncio.run(main())
    for s in socks:
        assert s.sent[-1] == {"beat": 99}
        assert len(s.sent) <= 2  # leading {beat: 0} + trailing {beat: 99}


def test_latest_broadcast_snapshots_at_send_time() -> None:
    """The OSC server handles each datagram on its own thread, so two
    handlers can mutate-then-submit in either order. Taking the snapshot at
    send time means a submit that lost that race can't ship a stale state."""
    sock = _RecordingSocket()
    live = {"beat": 0}
    calls = 0

    def snapshot() -> dict:
        nonlocal calls
        calls += 1
        return dict(live)

    async def main() -> None:
        manager = WSManager()
        manager.loop = asyncio.get_running_loop()
        manager.broadcast_latest_threadsafe(snapshot)  # nobody connected yet
        await asyncio.sleep(0.05)
        assert calls == 0  # no clients: no snapshot, no serialization

        manager.connections.add(sock)  # type: ignore[arg-type]
        manager.broadcast_latest_threadsafe(snapshot)
        live["beat"] = 42  # mutated after the submit, before the send
        for _ in range(100):
            if sock.sent:
                break
            await asyncio.sleep(0.01)

    asyncio.run(main())
    assert sock.sent == [{"beat": 42}]
