"""Read Live's warp grid over AbletonOSC — shared by the one-off scripts.

``scripts/warp_probe.py`` (report / repair a scene's stem grids) and
``scripts/grid_diff.py --live`` (judge dance's beat grid against the grid
Live is actually playing) both need the same three things: a blocking
request/reply helper, the deck-column map, and a parser for the fork's
``/live/clip/get/warp_marker_times`` reply. They live here so the two scripts
cannot drift apart.

Requires the patched AbletonOSC fork (see docs/proposals/warp-guard.md) and
exclusive use of the reply port — stop the backend first, it owns 11001.
Nothing in the app imports this: per CLAUDE.md rule 2, an unverified Live
capability stays in scripts.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Any

from dance.osc.bridge import AbletonBridge
from dance.osc.client import ABLETON_RECEIVE_PORT, ABLETON_SEND_PORT, AbletonOSCClient
from dance.osc.listener import AbletonOSCListener
from dance.pipeline.utils.grid_metrics import WarpMarker

# Deck columns as the bridge provisions them (drums_a=0 … mix_b=9).
DECK_COLUMNS: dict[str, int] = {k: i for i, k in enumerate(AbletonBridge._DECK_KINDS)}

WARP_MARKER_TIMES = "/live/clip/get/warp_marker_times"
CLIP_FILE_PATH = "/live/clip/get/file_path"
CLIP_LENGTH = "/live/clip/get/length"


class OscAsker:
    """Blocking ask-and-wait over AbletonOSC, for scripts only.

    ``ask`` clears the reply buffer, sends, sleeps ``wait`` seconds and
    returns the argument tuples of every reply on the same address. Crude,
    but it is exactly what the probe has always done and it needs no
    correlation IDs from Live.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = ABLETON_RECEIVE_PORT,
        reply_port: int = ABLETON_SEND_PORT,
        *,
        settle: float = 0.3,
    ) -> None:
        self.client = AbletonOSCClient(host=host, port=port)
        self.listener = AbletonOSCListener(port=reply_port)
        self.settle = settle
        self._replies: list[tuple[str, tuple[Any, ...]]] = []
        self._lock = threading.Lock()
        self.listener.on_any(self._on_reply)

    def _on_reply(self, address: str, args: tuple[Any, ...]) -> None:
        with self._lock:
            self._replies.append((address, args))

    def start(self) -> None:
        """Bind the reply port. Raises ``OSError`` when it is busy."""
        self.listener.start()
        time.sleep(self.settle)

    def stop(self) -> None:
        self.listener.stop()

    def __enter__(self) -> OscAsker:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def ask(self, address: str, *args: Any, wait: float = 0.8) -> list[tuple[Any, ...]]:
        with self._lock:
            self._replies.clear()
        self.client._send(address, *args)
        time.sleep(wait)
        with self._lock:
            return [a for (addr, a) in self._replies if addr == address]


def parse_warp_marker_times(args: Sequence[Any]) -> list[WarpMarker]:
    """Decode a ``warp_marker_times`` reply: ``(track, slot, beat, sec, …)``."""
    vals = [float(v) for v in args[2:]]
    return [WarpMarker(sec=s, beat=b) for b, s in zip(vals[0::2], vals[1::2])]


def read_warp_markers(
    asker: OscAsker, track: int, slot: int, *, wait: float = 0.8
) -> list[WarpMarker] | None:
    """The clip's warp markers, or None when the handler did not answer."""
    replies = asker.ask(WARP_MARKER_TIMES, track, slot, wait=wait)
    if not replies:
        return None
    return parse_warp_marker_times(replies[0])


def read_clip_file_path(asker: OscAsker, track: int, slot: int, *, wait: float = 0.8) -> str | None:
    """The clip's sample path (stock AbletonOSC getter), or None."""
    replies = asker.ask(CLIP_FILE_PATH, track, slot, wait=wait)
    if not replies or len(replies[0]) < 3:
        return None
    path = str(replies[0][2])
    return path or None


__all__ = [
    "CLIP_FILE_PATH",
    "CLIP_LENGTH",
    "DECK_COLUMNS",
    "OscAsker",
    "WARP_MARKER_TIMES",
    "parse_warp_marker_times",
    "read_clip_file_path",
    "read_warp_markers",
]
