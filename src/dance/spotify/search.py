"""Spotify Web API search — Client Credentials flow.

Powers Cmd-K's "ADD FROM SPOTIFY" section. spotDL handles its own auth for
downloads; this module is just for *finding* tracks the user can then
ingest. Client Credentials is the right flow for catalog search: no user
OAuth, no scopes, no redirect URIs — just a (client_id, client_secret)
exchanged for a 1-hour Bearer token.

The token is cached in memory with a small safety margin before expiry.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import httpx

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SEARCH_URL = "https://api.spotify.com/v1/search"
_TRACK_URL = "https://api.spotify.com/v1/tracks"


class SpotifyAuthError(RuntimeError):
    """Credentials missing, invalid, or refused by Spotify."""


class SpotifySearchError(RuntimeError):
    """Search request failed for non-auth reasons (rate limit, network)."""


@dataclass
class SpotifyTrackHit:
    """One Spotify search result, normalized for the FE."""

    spotify_id: str
    title: str
    artist: str
    album: str | None
    duration_ms: int | None
    preview_url: str | None
    image_url: str | None
    explicit: bool
    popularity: int | None


class SpotifySearchClient:
    """Cached-token Spotify catalog client. One per app, thread-safe enough
    for FastAPI's threadpool: token refresh is idempotent and a torn race
    only costs an extra token fetch."""

    def __init__(self, client_id: str | None, client_secret: str | None) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        # epoch seconds at which the current token expires; 0 = no token.
        self._expires_at: float = 0.0
        # Inject-able for tests; defaults to a real httpx client.
        self._http = httpx.Client(timeout=10.0)

    @property
    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_tracks(self, query: str, limit: int = 8) -> list[SpotifyTrackHit]:
        """Run a catalog search. Returns up to ``limit`` track hits."""
        if not query.strip():
            return []
        if not self.configured:
            raise SpotifyAuthError(
                "Spotify search not configured — set DANCE_SPOTIFY_CLIENT_ID "
                "and DANCE_SPOTIFY_CLIENT_SECRET in ~/.dance/.env"
            )
        token = self._get_token()
        try:
            r = self._http.get(
                _SEARCH_URL,
                params={"q": query, "type": "track", "limit": max(1, min(limit, 50))},
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise SpotifySearchError(f"Spotify search transport error: {exc}") from exc
        if r.status_code == 401:
            # Token might've expired mid-flight; force a refresh + retry once.
            self._token = None
            self._expires_at = 0
            token = self._get_token()
            r = self._http.get(
                _SEARCH_URL,
                params={"q": query, "type": "track", "limit": max(1, min(limit, 50))},
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code >= 400:
            raise SpotifySearchError(
                f"Spotify search HTTP {r.status_code}: {r.text[:200]}"
            )
        return [_parse_hit(it) for it in r.json().get("tracks", {}).get("items", [])]

    def get_track(self, spotify_id: str) -> SpotifyTrackHit:
        """Fetch a single track by Spotify ID. Used by ingest to fill in
        metadata when the caller only gives us the ID."""
        if not self.configured:
            raise SpotifyAuthError("Spotify search not configured")
        token = self._get_token()
        r = self._http.get(
            f"{_TRACK_URL}/{spotify_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code == 404:
            raise SpotifySearchError(f"Spotify track {spotify_id} not found")
        if r.status_code >= 400:
            raise SpotifySearchError(
                f"Spotify get-track HTTP {r.status_code}: {r.text[:200]}"
            )
        return _parse_hit(r.json())

    # ------------------------------------------------------------------
    # Token cache
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        # Refresh 30 s before expiry so an in-flight request can't catch
        # a token that's about to lapse.
        if self._token and self._expires_at > time.time() + 30:
            return self._token
        return self._refresh_token()

    def _refresh_token(self) -> str:
        assert self._client_id and self._client_secret  # configured-checked
        basic = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode("ascii")
        ).decode("ascii")
        try:
            r = self._http.post(
                _TOKEN_URL,
                data={"grant_type": "client_credentials"},
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        except httpx.HTTPError as exc:
            raise SpotifyAuthError(f"Spotify token transport error: {exc}") from exc
        if r.status_code != 200:
            raise SpotifyAuthError(
                f"Spotify token HTTP {r.status_code}: {r.text[:200]}"
            )
        data = r.json()
        self._token = str(data["access_token"])
        self._expires_at = time.time() + float(data.get("expires_in", 3600))
        return self._token


def _parse_hit(item: dict) -> SpotifyTrackHit:
    artists = item.get("artists") or []
    album = item.get("album") or {}
    images = album.get("images") or []
    # Spotify returns images sorted largest-first; the smallest (typically
    # 64×64) is plenty for our row thumbnail.
    image_url = images[-1]["url"] if images else None
    return SpotifyTrackHit(
        spotify_id=str(item["id"]),
        title=str(item.get("name") or "(untitled)"),
        artist=", ".join(a.get("name", "") for a in artists) or "(unknown)",
        album=album.get("name") or None,
        duration_ms=item.get("duration_ms"),
        preview_url=item.get("preview_url"),
        image_url=image_url,
        explicit=bool(item.get("explicit", False)),
        popularity=item.get("popularity"),
    )


# Module-level singleton — initialized lazily on first request via the API
# dep. Tests can inject a mock client instead.
_default_client: SpotifySearchClient | None = None


def get_default_client(settings) -> SpotifySearchClient:  # noqa: ANN001
    """Return the process-wide client, building it lazily from ``settings``."""
    global _default_client
    if _default_client is None:
        _default_client = SpotifySearchClient(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
        )
    return _default_client


def _reset_default_client_for_tests() -> None:
    global _default_client
    _default_client = None
