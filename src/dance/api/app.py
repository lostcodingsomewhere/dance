"""FastAPI app factory and lifespan management."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker

from dance.api.jobs import init_persisted_registry
from dance.api.routers import (
    ableton,
    files,
    pipeline,
    recommend,
    sessions,
    sets,
    spotify,
    tracks,
    ws,
)
from dance.api.routers.pipeline import start_watch_thread
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
    # Persist deck-column / cell / cue-track state inside the user's data
    # dir so a backend restart can restore what was loaded in Live. Path
    # is plumbed through here so tests using a tmp data_dir don't share
    # the production ``~/.dance/deck_state.json``.
    bridge = bridge or AbletonBridge(state_file=settings.data_dir / "deck_state.json")
    session_factory = session_factory or get_session_factory(settings.db_url)

    # Swap the global JobRegistry for one that persists to data_dir.
    # Order matters: this MUST happen before PipelineWSManager() so the
    # manager subscribes to the persisted instance.
    init_persisted_registry(settings.data_dir / "jobs.json")

    # Start the watch-mode polling thread. Reads its persisted
    # enabled flag from data_dir/watch.json. Idempotent.
    start_watch_thread(settings)

    ws_manager = ws.WSManager()
    pipeline_ws_manager = ws.PipelineWSManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Capture the running loop so the OSC listener thread can hop back.
        loop = asyncio.get_running_loop()
        ws_manager.loop = loop
        pipeline_ws_manager.loop = loop

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
    app.state.pipeline_ws_manager = pipeline_ws_manager
    # CLAP text encoder is lazy — first call to /recommend/text loads the model.
    app.state.embedding_stage = None

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
    app.include_router(tracks.stems_router, prefix=API_PREFIX)
    app.include_router(recommend.router, prefix=API_PREFIX)
    app.include_router(sessions.router, prefix=API_PREFIX)
    app.include_router(sets.router, prefix=API_PREFIX)
    app.include_router(spotify.router, prefix=API_PREFIX)
    app.include_router(ableton.router, prefix=API_PREFIX)
    app.include_router(files.router, prefix=API_PREFIX)
    app.include_router(pipeline.router, prefix=API_PREFIX)
    # WebSocket is unversioned at the moment — the spec says /ws.
    app.include_router(ws.router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"ok": True}

    from fastapi import Depends as _Depends

    from dance.api.deps import get_settings as _dep_get_settings

    @app.get(f"{API_PREFIX}/health/deps", tags=["meta"])
    def health_deps(s: Settings = _Depends(_dep_get_settings)) -> dict:
        """Report the status of every host-level + per-user dep the
        ingest path needs. Surfaced as a chip in MasterStrip so the
        user sees a problem before they try to ingest."""
        return _deps_status(s)

    return app


def _deps_status(s) -> dict:  # noqa: ANN001
    """Inspect host tools + per-user creds. Returns a flat report the FE
    can render as a checklist. Each ``status`` is "ok" / "missing" /
    "optional"."""
    from pathlib import Path

    # Match the resolution logic in ``download_track`` so the chip never
    # green-lights a tool that the worker would then fail to find.
    from dance.spotify.csv_importer import _resolve_ffmpeg, _resolve_yt_dlp

    yt_dlp_path = _resolve_yt_dlp()
    ff = _resolve_ffmpeg(None)
    ffmpeg_path = ff if Path(ff).exists() else None
    cookies = (
        s.youtube_cookies_file
        if s.youtube_cookies_file
        and Path(s.youtube_cookies_file).exists()
        else None
    )
    spotify_ok = bool(s.spotify_client_id and s.spotify_client_secret)

    checks = [
        {
            "key": "yt_dlp",
            "label": "yt-dlp",
            "status": "ok" if yt_dlp_path else "missing",
            "detail": yt_dlp_path
            or "Required for Spotify/YouTube ingest. brew install yt-dlp",
            "required": True,
        },
        {
            "key": "ffmpeg",
            "label": "ffmpeg",
            "status": "ok" if ffmpeg_path else "missing",
            "detail": ffmpeg_path
            or "Required by yt-dlp for audio extraction. brew install ffmpeg",
            "required": True,
        },
        {
            "key": "cookies",
            "label": "YouTube cookies",
            "status": "ok" if cookies else "optional",
            "detail": str(cookies)
            if cookies
            else (
                "Export to ~/.dance/cookies.txt via the 'Get cookies.txt "
                "LOCALLY' Chrome extension — without it, YouTube will throttle "
                "or block downloads."
            ),
            "required": False,
        },
        {
            "key": "spotify_creds",
            "label": "Spotify search",
            "status": "ok" if spotify_ok else "optional",
            "detail": (
                "Configured"
                if spotify_ok
                else (
                    "Add DANCE_SPOTIFY_CLIENT_ID + SECRET to ~/.dance/.env "
                    "to enable Cmd-K 'Add from Spotify'."
                )
            ),
            "required": False,
        },
    ]
    required_ok = all(c["status"] == "ok" for c in checks if c["required"])
    optional_ok = all(c["status"] == "ok" for c in checks if not c["required"])
    return {
        "ok": required_ok,
        "all_green": required_ok and optional_ok,
        "checks": checks,
    }


__all__ = ["API_PREFIX", "create_app"]
