"""WebSocket endpoints.

Two channels:

- ``/ws`` — AbletonState pushes (tempo / playing-clips / fader values).
  Hooks the OSC bridge's subscribe-callback. The bridge runs that on
  the OSC LISTENER thread, not the asyncio loop, so we capture the
  loop on app startup and use ``asyncio.run_coroutine_threadsafe`` to
  hop back when sending.

- ``/ws/pipeline`` — Job-registry pushes (CSV ingest progress: items
  flipping from ``pending`` → ``ok`` / ``skip`` / ``fail`` as yt-dlp
  downloads complete). Same threadsafe pattern: the JobRegistry calls
  its subscribers from a download worker thread.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from dance.api.jobs import Job, get_job_registry

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


class PipelineWSManager(WSManager):
    """WebSocket manager for the ``/ws/pipeline`` channel.

    Subscribes to the global JobRegistry on init; every job mutation triggers
    a threadsafe broadcast of the job's serialized state to all connected
    clients. UI uses this for real-time progress without 1-2 s poll latency.
    """

    def __init__(self) -> None:
        super().__init__()
        # Subscribe once at construction. JobRegistry holds a reference.
        get_job_registry().subscribe(self._on_job_change)

    def _on_job_change(self, job: Job) -> None:
        self.broadcast_threadsafe(job.to_dict())


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


@router.websocket("/ws/pipeline")
async def ws_pipeline_endpoint(websocket: WebSocket) -> None:
    """Pipeline-event WebSocket. Broadcasts on every job mutation.

    On connect, replays the current state of the most-recent ~5 jobs so
    the UI doesn't need a separate REST call to bootstrap.
    """
    request_app = websocket.app
    manager: PipelineWSManager = request_app.state.pipeline_ws_manager

    await manager.connect(websocket)
    try:
        # Replay recent jobs so the UI has something to show immediately.
        for job in get_job_registry().list(limit=5):
            await websocket.send_json(job.to_dict())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


__all__ = ["PipelineWSManager", "WSManager", "router"]
