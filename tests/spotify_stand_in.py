from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from dance.spotify import search as spotify_search
from dance.spotify.search import SpotifySearchClient

APP_ID = "test-app-id"
APP_SECRET = "test-app-secret"
APP_TOKEN = "app-token"
API = "https://api.spotify.com"
SEARCH_LIMIT_MAX = 10
SEARCH_LIMIT_DEFAULT = 5
PLAYLIST_LIMIT_MAX = 50
PLAYLIST_LIMIT_DEFAULT = 20

_FIELD_FILTER = re.compile(r'(\w+):"([^"]*)"')
_PLAYLIST_PATH = re.compile(r"^/v1/playlists/([A-Za-z0-9]+)(?:/(tracks|items))?$")


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"status": status, "message": message}})


@dataclass(frozen=True)
class CatalogTrack:
    spotify_id: str
    name: str
    artists: tuple[str, ...]
    seconds: float
    album: str = "Single"
    kind: str = "track"

    def as_spotify_sends_it(self) -> dict:
        return {
            "id": self.spotify_id,
            "name": self.name,
            "type": self.kind,
            "uri": f"spotify:{self.kind}:{self.spotify_id}",
            "artists": [{"name": name, "type": "artist"} for name in self.artists],
            "album": {
                "name": self.album,
                "album_type": "single",
                "images": [{"url": f"https://img.test/{self.spotify_id}", "height": 64}],
            },
            "duration_ms": int(self.seconds * 1000),
            "explicit": False,
            "external_ids": {"isrc": f"ISRC{self.spotify_id}"},
            "is_local": False,
            "is_playable": True,
        }


@dataclass
class Playlist:
    owner: str
    entries: list[CatalogTrack | None]


@dataclass
class SpotifyStandIn:
    catalog: list[CatalogTrack] = field(default_factory=list)
    playlists: dict[str, Playlist] = field(default_factory=dict)
    user_of_token: dict[str, str] = field(default_factory=dict)
    most_playlist_entries_a_page: int | None = None
    search_fails_with: int | None = None
    passing_search_failures: int = 0
    requests: list[httpx.Request] = field(default_factory=list)

    def client(self) -> SpotifySearchClient:
        client = SpotifySearchClient(APP_ID, APP_SECRET)
        client._http = httpx.Client(transport=httpx.MockTransport(self.handle))
        client._sleep = lambda _seconds: None
        return client

    def stands_in_for_spotify(self, monkeypatch) -> None:
        monkeypatch.setattr(spotify_search, "_default_client", self.client())

    def asked(self, path_start: str) -> list[dict[str, str]]:
        return [
            dict(request.url.params)
            for request in self.requests
            if request.url.path.startswith(path_start)
        ]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/token":
            return httpx.Response(200, json={"access_token": APP_TOKEN, "expires_in": 3600})
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if path == "/v1/search":
            return self._search(request, bearer)
        if path.startswith("/v1/tracks/"):
            return self._track(path.rsplit("/", 1)[1], bearer)
        playlist = _PLAYLIST_PATH.match(path)
        if playlist:
            return self._playlist(request, playlist.group(1), playlist.group(2), bearer)
        return _error(404, "Service not found")

    def _is_signed_in_app_or_user(self, bearer: str) -> bool:
        return bearer == APP_TOKEN or bearer in self.user_of_token

    def _search(self, request: httpx.Request, bearer: str) -> httpx.Response:
        if not self._is_signed_in_app_or_user(bearer):
            return _error(401, "Invalid access token")
        if self.search_fails_with is not None:
            return _error(self.search_fails_with, "Server error")
        if self.passing_search_failures > 0:
            self.passing_search_failures -= 1
            return _error(502, "An unexpected error occurred. Please try again later.")
        params = request.url.params
        limit = int(params.get("limit", SEARCH_LIMIT_DEFAULT))
        if not 1 <= limit <= SEARCH_LIMIT_MAX:
            return _error(400, "Invalid limit")
        offset = int(params.get("offset", 0))
        found = [track for track in self.catalog if _matches(track, params.get("q", ""))]
        page = found[offset : offset + limit]
        return httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [track.as_spotify_sends_it() for track in page],
                    "limit": limit,
                    "offset": offset,
                    "total": len(found),
                }
            },
        )

    def _track(self, spotify_id: str, bearer: str) -> httpx.Response:
        if not self._is_signed_in_app_or_user(bearer):
            return _error(401, "Invalid access token")
        for track in self.catalog:
            if track.spotify_id == spotify_id:
                return httpx.Response(200, json=track.as_spotify_sends_it())
        return _error(404, "Resource not found")

    def _playlist(
        self, request: httpx.Request, playlist_id: str, part: str | None, bearer: str
    ) -> httpx.Response:
        if part == "tracks":
            return _error(403, "Forbidden")
        if part is None:
            if playlist_id not in self.playlists:
                return _error(404, "Resource not found")
            return httpx.Response(200, json={"id": playlist_id, "name": "A playlist"})
        user = self.user_of_token.get(bearer)
        if user is None:
            return _error(401, "Valid user authentication required")
        playlist = self.playlists.get(playlist_id)
        if playlist is None:
            return _error(404, "Resource not found")
        if playlist.owner != user:
            return _error(403, "Forbidden")
        params = request.url.params
        limit = int(params.get("limit", PLAYLIST_LIMIT_DEFAULT))
        if not 1 <= limit <= PLAYLIST_LIMIT_MAX:
            return _error(400, "Invalid limit")
        size = min(limit, self.most_playlist_entries_a_page or limit)
        offset = int(params.get("offset", 0))
        page = playlist.entries[offset : offset + size]
        more = offset + size < len(playlist.entries)
        next_page = (
            f"{API}/v1/playlists/{playlist_id}/items?offset={offset + size}&limit={limit}"
            if more
            else None
        )
        return httpx.Response(
            200,
            json={
                "href": str(request.url),
                "items": [_playlist_entry(entry) for entry in page],
                "limit": limit,
                "next": next_page,
                "offset": offset,
                "previous": None,
                "total": len(playlist.entries),
            },
        )


def _playlist_entry(entry: CatalogTrack | None) -> dict:
    if entry is None:
        local_file = {"id": None, "name": "A file on someone's disk", "type": "track"}
        return {"added_at": "2026-01-01T00:00:00Z", "is_local": True, "item": local_file}
    return {
        "added_at": "2026-01-01T00:00:00Z",
        "is_local": False,
        "item": entry.as_spotify_sends_it(),
    }


def _matches(track: CatalogTrack, query: str) -> bool:
    name = track.name.lower()
    artists = [artist.lower() for artist in track.artists]
    for field_name, phrase in _FIELD_FILTER.findall(query):
        wanted = phrase.lower()
        if field_name == "track" and wanted not in name:
            return False
        if field_name == "artist" and not any(wanted in artist for artist in artists):
            return False
    everything = " ".join([name, *artists])
    return all(word in everything for word in _FIELD_FILTER.sub(" ", query).lower().split())
