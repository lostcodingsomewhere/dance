"""Spotify catalog search endpoint.

The pipeline router has the *ingest* endpoint that actually creates a Track
row and queues a download job; this module is just the lightweight catalog
search that the Cmd-K palette calls as the user types. Auth is the Client
Credentials flow — no user OAuth needed for public catalog reads.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from dance.api.deps import get_settings
from dance.config import Settings
from dance.spotify.search import (
    SpotifyAuthError,
    SpotifySearchError,
    get_default_client,
)

router = APIRouter(prefix="/spotify", tags=["spotify"])


@router.get("/search")
def search_tracks(
    q: str = Query(..., min_length=1, description="Free-text catalog query"),
    limit: int = Query(8, ge=1, le=20),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return Spotify catalog hits for ``q``. Used by the Cmd-K palette's
    "ADD FROM SPOTIFY" section when the local library has no matches.

    Returns 503 if credentials aren't configured (server-side), 502 if
    Spotify itself errors. The FE renders an inline hint in either case.
    """
    client = get_default_client(settings)
    if not client.configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "Spotify search is not configured. Add DANCE_SPOTIFY_CLIENT_ID "
                "and DANCE_SPOTIFY_CLIENT_SECRET to ~/.dance/.env."
            ),
        )
    try:
        hits = client.search_tracks(q, limit=limit)
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=503, detail=f"Spotify auth: {exc}") from exc
    except SpotifySearchError as exc:
        raise HTTPException(status_code=502, detail=f"Spotify error: {exc}") from exc
    return {
        "query": q,
        "hits": [
            {
                "spotify_id": h.spotify_id,
                "title": h.title,
                "artist": h.artist,
                "album": h.album,
                "duration_ms": h.duration_ms,
                "preview_url": h.preview_url,
                "image_url": h.image_url,
                "explicit": h.explicit,
                "popularity": h.popularity,
            }
            for h in hits
        ],
    }
