"""WebSocket endpoint that streams AbletonState changes to clients.

The bridge runs its subscribe-callback on the OSC LISTENER THREAD, not on
the asyncio event loop. We capture the loop on app startup and use
``asyncio.run_coroutine_threadsafe`` to hop back when we need to send.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class WSManager:
    """Tracks the live WebSocket connections and pushes state to them.

    A single instance is stored on ``app.state.ws_manager`` by ``create_app``.
    """

    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.connections.discard(ws)

    async def _send_one(self, ws: WebSocket, payload: dict) -> None:
        try:
            await ws.send_json(payload)
        except Exception:  # noqa: BLE001
            # Likely client gone — drop it.
            self.connections.discard(ws)

    def broadcast_threadsafe(self, payload: dict) -> None:
        """Schedule a broadcast from a non-asyncio thread."""
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        for ws in list(self.connections):
            asyncio.run_coroutine_threadsafe(self._send_one(ws, payload), loop)


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    request_app = websocket.app
    manager: WSManager = request_app.state.ws_manager
    bridge = request_app.state.bridge

    await manager.connect(websocket)
    try:
        # Send the current snapshot once on connect.
        await websocket.send_json(bridge.state.to_dict())
        # Hold the connection open. We don't expect inbound messages, but we
        # must read so disconnects are noticed promptly.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


__all__ = ["WSManager", "router"]
