"""Spotify catalog reads: free-text search and a song's other versions.

The pipeline router has the *ingest* endpoint that actually creates a Track
row and queues a download job; this module only looks things up. Auth is the
Client Credentials flow — no user OAuth needed for public catalog reads.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from dance.api.deps import get_session, get_settings
from dance.api.schemas import SpotifyVersionsOut
from dance.config import Settings
from dance.core.database import Track
from dance.spotify.search import (
    SpotifyAuthError,
    SpotifySearchClient,
    SpotifySearchError,
    get_default_client,
)
from dance.spotify.versions import (
    NothingToSearchWithError,
    held_in_library,
    versions_on_spotify,
)

router = APIRouter(prefix="/spotify", tags=["spotify"])

_NOT_CONFIGURED = (
    "Spotify search is not configured. Add DANCE_SPOTIFY_CLIENT_ID "
    "and DANCE_SPOTIFY_CLIENT_SECRET to ~/.dance/.env."
)


def _configured_client(settings: Settings) -> SpotifySearchClient:
    client = get_default_client(settings)
    if not client.configured:
        raise HTTPException(status_code=503, detail=_NOT_CONFIGURED)
    return client


@router.get("/search")
def search_tracks(
    q: str = Query(..., min_length=1, description="Free-text catalog query"),
    limit: int = Query(8, ge=1, le=20),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return Spotify catalog hits for ``q``. Used by the Cmd-K palette's
    "ADD FROM SPOTIFY" section when the local library has no matches.

    A hit the library already holds carries that track's id and state in
    ``library_track_id`` and ``library_state``; both are null otherwise.

    Returns 503 if credentials aren't configured (server-side), 502 if
    Spotify itself errors. The FE renders an inline hint in either case.
    """
    client = _configured_client(settings)
    try:
        hits = client.search_tracks(q, limit=limit)
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=503, detail=f"Spotify auth: {exc}") from exc
    except SpotifySearchError as exc:
        raise HTTPException(status_code=502, detail=f"Spotify error: {exc}") from exc
    held = held_in_library(session, hits)
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
                "library_track_id": mine.track_id if mine else None,
                "library_state": mine.state if mine else None,
            }
            for h, mine in zip(hits, held, strict=True)
        ],
    }


@router.get("/versions", response_model=SpotifyVersionsOut)
def find_versions(
    track_id: int = Query(..., description="The library track whose versions to look for"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    track = session.get(Track, track_id)
    if track is None:
        raise HTTPException(status_code=404, detail="track not found")
    client = _configured_client(settings)
    try:
        found = versions_on_spotify(session, client, track)
    except NothingToSearchWithError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpotifyAuthError as exc:
        raise HTTPException(status_code=503, detail=f"Spotify auth: {exc}") from exc
    except SpotifySearchError as exc:
        raise HTTPException(status_code=502, detail=f"Spotify error: {exc}") from exc
    return {
        "track_id": track_id,
        "query": found.query,
        "versions": [
            {
                "spotify_id": version.hit.spotify_id,
                "title": version.hit.title,
                "artist": version.hit.artist,
                "album": version.hit.album,
                "duration_ms": version.hit.duration_ms,
                "image_url": version.hit.image_url,
                "explicit": version.hit.explicit,
                "label": version.label,
                "library_track_id": version.in_library.track_id if version.in_library else None,
                "library_state": version.in_library.state if version.in_library else None,
                "download_failed": bool(version.in_library and version.in_library.download_failed),
            }
            for version in found.versions
        ],
    }
