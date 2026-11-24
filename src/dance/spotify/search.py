"""Spotify Web API reads: catalog search, one track, and a playlist's songs.

Search and track lookups use the Client Credentials flow: a (client_id,
client_secret) pair exchanged for a 1-hour Bearer token, cached in memory with
a small safety margin before expiry. A playlist's songs need a user's token.

What Spotify answered on 2026-11-24, after its February 2026 Web API change:
search takes at most 10 results a request, ``/playlists/{id}/tracks`` is gone
(403), and ``/playlists/{id}/items`` answers only a user token, and only for a
playlist that user owns or collaborates on.
"""

from __future__ import annotations

import base64
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SEARCH_URL = "https://api.spotify.com/v1/search"
_TRACK_URL = "https://api.spotify.com/v1/tracks"
_PLAYLIST_URL = "https://api.spotify.com/v1/playlists"

SEARCH_PAGE_SIZE = 10
PLAYLIST_PAGE_SIZE = 50

PLAYLIST_NEEDS_A_USER = (
    "Spotify lists a playlist's songs only for a signed-in user, so this needs a "
    "Spotify user token: paste one in the Pipeline form, or set "
    "DANCE_SPOTIFY_USER_TOKEN in ~/.dance/.env. A playlist exported as a CSV and "
    "pasted in the Pipeline view needs no token."
)
PLAYLIST_BELONGS_TO_SOMEONE_ELSE = (
    "Spotify refused this playlist. It lists a playlist's songs only for the "
    "account that owns it or collaborates on it. Use a token from that account, "
    "or export the playlist as a CSV and paste that in the Pipeline view."
)

# HTTP 429 handling. Spotify rate-limits per app over a rolling 30 s window
# and says how long to back off in ``Retry-After`` (seconds). A 300-track
# playlist pages + per-track lookups can trip it; before this the first 429
# surfaced as a hard error. We honour the header, add jitter so concurrent
# callers don't retry in lockstep, and give up after a few tries — or at
# once when Spotify asks for a wait too long to hold an API request open
# (its penalty box can be hours).
_MAX_429_RETRIES = 3
_MAX_RETRY_AFTER_S = 30.0
_RETRY_JITTER_S = 1.0
# Search answered 502 "Please try again later" once in about 85 real calls on 2026-11-24.
_PASSING_SERVER_ERRORS = frozenset({500, 502, 503, 504})
_MAX_SERVER_ERROR_RETRIES = 2


def _retry_after_seconds(header: str | None, attempt: int) -> float:
    """Seconds to wait from a ``Retry-After`` header (delta or HTTP date).

    Missing or unparseable → exponential backoff 1, 2, 4 s by attempt.
    """
    if header:
        value = header.strip()
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(value)
            return max(0.0, when.timestamp() - time.time())
        except (TypeError, ValueError, IndexError):
            pass
    return float(2**attempt)


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

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        user_token: str | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        # A user's own token, needed to read a playlist's songs.
        self._user_token = user_token
        self._token: str | None = None
        # epoch seconds at which the current token expires; 0 = no token.
        self._expires_at: float = 0.0
        # Inject-able for tests; defaults to a real httpx client.
        self._http = httpx.Client(timeout=10.0)
        # Inject-able for tests so 429 back-off doesn't really sleep.
        self._sleep: Callable[[float], None] = time.sleep

    @property
    def configured(self) -> bool:
        """True when *any* credential is set. Search and track lookups need
        Client Credentials; a playlist's songs need a user token."""
        return bool(self._client_id and self._client_secret) or bool(
            self._user_token
        )

    @property
    def has_user_token(self) -> bool:
        return bool(self._user_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_tracks(self, query: str, limit: int = 8) -> list[SpotifyTrackHit]:
        """Run a catalog search. Returns up to ``limit`` track hits, reading as
        many pages as that takes."""
        if not query.strip():
            return []
        if not self.configured:
            raise SpotifyAuthError(
                "Spotify search not configured — set DANCE_SPOTIFY_CLIENT_ID "
                "and DANCE_SPOTIFY_CLIENT_SECRET in ~/.dance/.env"
            )
        hits: list[SpotifyTrackHit] = []
        while len(hits) < limit:
            asked_for = min(SEARCH_PAGE_SIZE, limit - len(hits))
            page = self._search_page(query, asked_for, offset=len(hits))
            hits += page
            if len(page) < asked_for:
                break
        return hits

    def _search_page(self, query: str, size: int, offset: int) -> list[SpotifyTrackHit]:
        params = {"q": query, "type": "track", "limit": size, "offset": offset}
        token = self._get_token()
        try:
            r = self._get(_SEARCH_URL, params=params, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise SpotifySearchError(f"Spotify search transport error: {exc}") from exc
        if r.status_code == 401:
            # Token might've expired mid-flight; force a refresh + retry once.
            self._token = None
            self._expires_at = 0
            token = self._get_token()
            r = self._get(_SEARCH_URL, params=params, headers={"Authorization": f"Bearer {token}"})
        if r.status_code >= 400:
            raise SpotifySearchError(f"Spotify search HTTP {r.status_code}: {r.text[:200]}")
        return [_parse_hit(it) for it in r.json().get("tracks", {}).get("items", [])]

    def get_track(self, spotify_id: str) -> SpotifyTrackHit:
        """Fetch a single track by Spotify ID. Used by ingest to fill in
        metadata when the caller only gives us the ID."""
        if not self.configured:
            raise SpotifyAuthError("Spotify search not configured")
        token = self._get_token()
        r = self._get(
            f"{_TRACK_URL}/{spotify_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code == 404:
            raise SpotifySearchError(f"Spotify track {spotify_id} not found")
        if r.status_code >= 400:
            raise SpotifySearchError(f"Spotify get-track HTTP {r.status_code}: {r.text[:200]}")
        return _parse_hit(r.json())

    def get_playlist_tracks(
        self,
        playlist_url_or_id: str,
        user_token: str | None = None,
    ) -> list[SpotifyTrackHit]:
        """Every song in a playlist the token's user owns or collaborates on.

        Accepts a raw playlist ID or any of Spotify's URL forms. ``user_token``
        is used in preference to the stored ``DANCE_SPOTIFY_USER_TOKEN``.
        Podcast episodes, local files and repeats of a song are left out, so
        every returned hit has a usable ``spotify_id``.
        """
        bearer = user_token or self._user_token
        if bearer is None:
            raise SpotifyAuthError(PLAYLIST_NEEDS_A_USER)
        playlist_id = extract_playlist_id(playlist_url_or_id)
        if not playlist_id:
            raise SpotifySearchError(f"Could not extract playlist ID from {playlist_url_or_id!r}")

        hits: list[SpotifyTrackHit] = []
        seen_ids: set[str] = set()
        offset = 0
        while True:
            try:
                r = self._get(
                    f"{_PLAYLIST_URL}/{playlist_id}/items",
                    params={"limit": PLAYLIST_PAGE_SIZE, "offset": offset},
                    headers={"Authorization": f"Bearer {bearer}"},
                )
            except httpx.HTTPError as exc:
                raise SpotifySearchError(f"Spotify playlist transport error: {exc}") from exc
            if r.status_code == 401:
                raise SpotifyAuthError("Spotify user token expired (or invalid). Paste a fresh one.")
            if r.status_code == 403:
                raise SpotifyAuthError(
                    f'{PLAYLIST_BELONGS_TO_SOMEONE_ELSE} Spotify said "{_what_spotify_said(r)}".'
                )
            if r.status_code == 404:
                raise SpotifySearchError(
                    f"Spotify playlist {playlist_id} not found (private or wrong URL?)"
                )
            if r.status_code >= 400:
                raise SpotifySearchError(f"Spotify playlist HTTP {r.status_code}: {r.text[:200]}")
            payload = r.json()
            entries = payload.get("items") or []
            for entry in entries:
                # Spotify renamed the entry's "track" to "item" and still documents both.
                track = entry.get("item") or entry.get("track")
                if not track or track.get("type") != "track":
                    continue
                tid = track.get("id")
                if not tid or tid in seen_ids:
                    continue
                seen_ids.add(tid)
                hits.append(_parse_hit(track))
            offset += len(entries)
            if not entries or not payload.get("next"):
                return hits

    # ------------------------------------------------------------------
    # Transport with 429 back-off
    # ------------------------------------------------------------------

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._send(self._http.get, url, **kwargs)

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._send(self._http.post, url, **kwargs)

    def _send(
        self, method: Callable[..., httpx.Response], url: str, **kwargs: Any
    ) -> httpx.Response:
        """Issue a request, retrying HTTP 429 per ``Retry-After`` and a
        server error twice.

        Returns the final response — a 429 or server error after the last
        retry is handed back for the caller's normal ``>= 400`` handling. Raises
        :class:`SpotifySearchError` straight away when Spotify asks for a
        wait longer than ``_MAX_RETRY_AFTER_S``.
        """
        attempt = 0
        server_errors = 0
        while True:
            r = method(url, **kwargs)
            if r.status_code in _PASSING_SERVER_ERRORS:
                if server_errors >= _MAX_SERVER_ERROR_RETRIES:
                    return r
                self._sleep(float(2**server_errors) + random.uniform(0.0, _RETRY_JITTER_S))
                server_errors += 1
                continue
            if r.status_code != 429 or attempt >= _MAX_429_RETRIES:
                return r
            wait = _retry_after_seconds(r.headers.get("Retry-After"), attempt)
            if wait > _MAX_RETRY_AFTER_S:
                raise SpotifySearchError(
                    f"Spotify rate limit (HTTP 429): asked to wait {wait:.0f}s, "
                    f"more than {_MAX_RETRY_AFTER_S:.0f}s — try again later"
                )
            self._sleep(wait + random.uniform(0.0, _RETRY_JITTER_S))
            attempt += 1

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
        basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode("ascii")).decode(
            "ascii"
        )
        try:
            r = self._post(
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
            raise SpotifyAuthError(f"Spotify token HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        self._token = str(data["access_token"])
        self._expires_at = time.time() + float(data.get("expires_in", 3600))
        return self._token


_PLAYLIST_ID_RE = re.compile(r"playlist[/:]([A-Za-z0-9]+)")


def extract_playlist_id(value: str) -> str | None:
    """Pull a playlist ID out of any Spotify reference shape.

    Accepts:
      - ``"37i9dQZF1DXcBWIGoYBM5M"`` (raw ID, 22-char base62)
      - ``"https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=..."``
      - ``"spotify:playlist:37i9dQZF1DXcBWIGoYBM5M"``

    Returns ``None`` when no plausible ID can be recovered — caller decides
    what to do (the API endpoint returns 400).
    """
    if not value:
        return None
    value = value.strip()
    m = _PLAYLIST_ID_RE.search(value)
    if m:
        return m.group(1)
    # Bare ID — Spotify IDs are URL-safe base62, typically 22 chars but the
    # length isn't part of the public spec. Accept any alphanumeric run of
    # 10+ chars on its own line as a raw ID.
    if re.fullmatch(r"[A-Za-z0-9]{10,}", value):
        return value
    return None


def _what_spotify_said(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return response.text[:100]


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
    """Return the process-wide client, building it lazily from ``settings``.

    We rebuild when the env-supplied user token changes — tokens last ~1
    hour, so the user will re-paste into ``~/.dance/.env`` periodically.
    Without this check, a stale client would keep using the original
    token even after the file changed (the API process picks up the new
    Settings on each request but the singleton client wouldn't refresh)."""
    global _default_client
    if (
        _default_client is None
        or getattr(_default_client, "_user_token", None) != settings.spotify_user_token
    ):
        _default_client = SpotifySearchClient(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            user_token=settings.spotify_user_token,
        )
    return _default_client


def _reset_default_client_for_tests() -> None:
    global _default_client
    _default_client = None
