"""Coalesce a high-rate stream of snapshots into a bounded broadcast rate.

The OSC bridge calls its state listeners on EVERY inbound OSC message: each
playing clip's ``playing_position`` (Live pushes those at roughly display rate,
per track), each deck's output meter, the transport beat. With two decks of
four stems firing that is several hundred full-``AbletonState`` broadcasts a
second, each one serialized and written to every connected browser, which then
re-renders for each. Nobody can see more than one state per frame, and every
consumer only ever wants the NEWEST state.

``LatestValueThrottle`` keeps the newest submitted value and emits it at most
once per ``1 / max_hz`` seconds:

- **Leading edge** — a change after a quiet spell goes out on the next loop
  tick, so a lone tempo nudge or a clip launch is not delayed.
- **Trailing edge** — the last value of a burst is always emitted, at most one
  interval late. A value is never dropped in favour of an older one: each emit
  takes whatever is newest at that moment.

Threading: ``submit_threadsafe`` may be called from any thread (the OSC
listener thread in production). The emit callback always runs on the asyncio
loop passed in, so it can create tasks / write to WebSockets directly.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default cap for AbletonState pushes on ``/ws``. Live itself reports clip
#: positions at about this rate, and the companion app renders at most once
#: per animation frame, so more than this is pure overhead.
STATE_BROADCAST_MAX_HZ = 30.0


class LatestValueThrottle(Generic[T]):
    """Emit the newest submitted value at most ``max_hz`` times a second."""

    def __init__(
        self,
        emit: Callable[[T], None],
        *,
        max_hz: float = STATE_BROADCAST_MAX_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_hz <= 0:
            raise ValueError("max_hz must be positive")
        self._emit = emit
        self._interval = 1.0 / max_hz
        self._clock = clock
        self._lock = threading.Lock()
        # Guarded by _lock: the newest value not yet emitted, and the loop a
        # flush is currently scheduled on (None = nothing scheduled).
        self._latest: T | None = None
        self._has_latest = False
        self._scheduled_on: asyncio.AbstractEventLoop | None = None
        # Only touched on the loop thread.
        self._last_emit = float("-inf")

    @property
    def interval(self) -> float:
        return self._interval

    def submit_threadsafe(self, value: T, loop: asyncio.AbstractEventLoop) -> None:
        """Record ``value`` as the newest and make sure a flush is pending.

        Cheap when a flush is already scheduled: it just swaps the stored
        value under a lock, so a 500 Hz producer costs 500 assignments, not
        500 broadcasts.
        """
        with self._lock:
            self._latest = value
            self._has_latest = True
            pending = self._scheduled_on
            # A flush parked on a loop that has since closed (an app whose
            # lifespan restarted) will never run — re-arm on the live loop.
            if pending is loop and not loop.is_closed():
                return
            self._scheduled_on = loop
        try:
            loop.call_soon_threadsafe(self._arm, loop)
        except RuntimeError:
            # Loop closed between the check and the call (shutdown). Nothing
            # to deliver to; leave the value for a future loop.
            with self._lock:
                if self._scheduled_on is loop:
                    self._scheduled_on = None

    # -- loop thread ---------------------------------------------------

    def _arm(self, loop: asyncio.AbstractEventLoop) -> None:
        delay = self._last_emit + self._interval - self._clock()
        if delay <= 0:
            self._flush()
        else:
            loop.call_later(delay, self._flush)

    def _flush(self) -> None:
        with self._lock:
            value = self._latest
            has_value = self._has_latest
            self._latest = None
            self._has_latest = False
            self._scheduled_on = None
        if not has_value:
            return
        self._last_emit = self._clock()
        try:
            self._emit(cast(T, value))
        except Exception:  # noqa: BLE001
            logger.exception("Throttled broadcast failed")


__all__ = ["STATE_BROADCAST_MAX_HZ", "LatestValueThrottle"]
