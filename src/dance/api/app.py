"""FastAPI app factory and lifespan management."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker

from dance.api.routers import ableton, recommend, sessions, tracks, ws
from dance.config import Settings, get_settings
from dance.core.database import get_session_factory
from dance.osc.bridge import AbletonBridge, AbletonState

logger = logging.getLogger(__name__)


API_PREFIX = "/api/v1"


def create_app(
    settings: Settings | None = None,
    bridge: AbletonBridge | None = None,
    session_factory: sessionmaker | None = None,
) -> FastAPI:
    """Build a FastAPI app.

    Parameters
    ----------
    settings: dependency-injectable Settings; defaults to ``get_settings()``.
    bridge:   pre-built AbletonBridge; if ``None``, a fresh one is created and
              its lifecycle is managed by the lifespan.
    session_factory: SQLAlchemy sessionmaker; if ``None``, derived from settings.
    """

    settings = settings or get_settings()
    own_bridge = bridge is None
    bridge = bridge or AbletonBridge()
    session_factory = session_factory or get_session_factory(settings.db_url)

    ws_manager = ws.WSManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Capture the running loop so the OSC listener thread can hop back.
        ws_manager.loop = asyncio.get_running_loop()

        def _on_state(state: AbletonState) -> None:
            ws_manager.broadcast_threadsafe(state.to_dict())

        bridge.subscribe(_on_state)
        if own_bridge:
            try:
                bridge.start()
            except Exception:  # noqa: BLE001
                logger.exception("Bridge failed to start; continuing without it")
        try:
            yield
        finally:
            if own_bridge:
                try:
                    bridge.stop()
                except Exception:  # noqa: BLE001
                    logger.exception("Bridge failed to stop cleanly")

    app = FastAPI(title="Dance API", version="0.1.0", lifespan=lifespan)

    # Stash singletons on app.state so dependencies can fetch them.
    app.state.settings = settings
    app.state.bridge = bridge
    app.state.session_factory = session_factory
    app.state.ws_manager = ws_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(tracks.router, prefix=API_PREFIX)
    app.include_router(recommend.router, prefix=API_PREFIX)
    app.include_router(sessions.router, prefix=API_PREFIX)
    app.include_router(ableton.router, prefix=API_PREFIX)
    # WebSocket is unversioned at the moment — the spec says /ws.
    app.include_router(ws.router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"ok": True}

    return app


__all__ = ["API_PREFIX", "create_app"]
