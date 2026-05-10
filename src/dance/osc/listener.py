"""OSC listener — receives state pushes from AbletonOSC.

Runs a UDP server in a background thread, dispatches incoming OSC messages
to registered handlers, and exposes the latest observed state.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server

from dance.osc.client import ABLETON_SEND_PORT

logger = logging.getLogger(__name__)


Handler = Callable[[str, tuple[Any, ...]], None]


class AbletonOSCListener:
    """Background OSC server.

    Usage:
        listener = AbletonOSCListener()
        listener.on("/live/song/get/tempo", lambda addr, args: print(args[0]))
        listener.start()
        ...
        listener.stop()

    Handlers run on the server thread — keep them fast (push to a queue or
    update an in-memory dict, don't block).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = ABLETON_SEND_PORT) -> None:
        self.host = host
        self.port = port
        self._dispatcher = osc_dispatcher.Dispatcher()
        self._dispatcher.set_default_handler(self._default_handler)
        self._server: osc_server.ThreadingOSCUDPServer | None = None
        self._thread: threading.Thread | None = None
        self._handlers: dict[str, list[Handler]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Listener already started")
        self._server = osc_server.ThreadingOSCUDPServer(
            (self.host, self.port), self._dispatcher
        )
        # Re-read bound port in case caller passed 0 (auto-assign).
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="osc-listener", daemon=True
        )
        self._thread.start()
        logger.info("OSC listener bound on %s:%d", self.host, self.port)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def on(self, address: str, handler: Handler) -> None:
        """Register a handler for an OSC address."""
        self._handlers.setdefault(address, []).append(handler)

    def on_any(self, handler: Handler) -> None:
        """Register a handler that receives every message."""
        # Reuse the default handler list — entries here are called for all messages.
        self._handlers.setdefault("*", []).append(handler)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _default_handler(self, address: str, *args: Any) -> None:
        for handler in self._handlers.get(address, []):
            try:
                handler(address, tuple(args))
            except Exception:  # noqa: BLE001
                logger.exception("OSC handler %s on %s crashed", handler, address)
        for handler in self._handlers.get("*", []):
            try:
                handler(address, tuple(args))
            except Exception:  # noqa: BLE001
                logger.exception("OSC catch-all handler %s crashed", handler)
