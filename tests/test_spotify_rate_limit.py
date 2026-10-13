"""Spotify HTTP 429: honour Retry-After with bounded, jittered retries.

Spotify rate-limits per app over a rolling window. Before, the first 429
surfaced as a hard error and a playlist ingest died part-way through.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from dance.spotify.search import SpotifySearchClient, SpotifySearchError

_SEARCH_OK = {
    "tracks": {
        "items": [
            {
                "id": "t1",
                "name": "Strobe",
                "artists": [{"name": "deadmau5"}],
                "album": {"name": "For Lack", "images": []},
                "duration_ms": 600_000,
            }
        ]
    }
}
_TOKEN_OK = {"access_token": "tok", "expires_in": 3600}


def _client(responses: list[httpx.Response]) -> tuple[SpotifySearchClient, list[float], list]:
    """Client whose transport replays ``responses`` (token POSTs answered OK)."""
    calls: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/token":
            if queue and getattr(queue[0], "_for_token", False):
                return queue.pop(0)
            return httpx.Response(200, json=_TOKEN_OK)
        return queue.pop(0)

    c = SpotifySearchClient("id", "secret")
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    sleeps: list[float] = []
    c._sleep = sleeps.append
    return c, sleeps, calls


def _r429(retry_after: str | None = None, for_token: bool = False) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    r = httpx.Response(429, headers=headers, text="rate limited")
    r._for_token = for_token  # type: ignore[attr-defined]
    return r


def test_429_honours_retry_after_then_succeeds() -> None:
    c, sleeps, _ = _client([_r429("2"), httpx.Response(200, json=_SEARCH_OK)])
    hits = c.search_tracks("strobe")
    assert [h.title for h in hits] == ["Strobe"]
    (slept,) = sleeps
    assert 2.0 <= slept <= 3.0  # Retry-After + up to 1 s jitter


def test_429_without_header_backs_off_exponentially() -> None:
    c, sleeps, _ = _client([_r429(), _r429(), httpx.Response(200, json=_SEARCH_OK)])
    assert c.search_tracks("strobe")
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 2.0 and 2.0 <= sleeps[1] <= 3.0


def test_429_retries_are_bounded() -> None:
    c, sleeps, calls = _client([_r429("1") for _ in range(4)])
    with pytest.raises(SpotifySearchError, match="429"):
        c.search_tracks("strobe")
    assert len(sleeps) == 3
    assert sum(1 for r in calls if r.url.path == "/v1/search") == 4


def test_429_with_a_long_penalty_fails_fast() -> None:
    """Spotify's penalty box can be hours; never hold a request open for it."""
    c, sleeps, calls = _client([_r429("3600")])
    with pytest.raises(SpotifySearchError, match="3600s"):
        c.search_tracks("strobe")
    assert sleeps == []
    assert sum(1 for r in calls if r.url.path == "/v1/search") == 1


def test_429_retry_after_as_http_date() -> None:
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=5), usegmt=True)
    c, sleeps, _ = _client([_r429(when), httpx.Response(200, json=_SEARCH_OK)])
    assert c.search_tracks("strobe")
    (slept,) = sleeps
    assert 3.0 <= slept <= 6.5


def test_429_on_playlist_pages_is_retried() -> None:
    page = {
        "items": [
            {
                "track": {
                    "id": "p1",
                    "name": "One",
                    "type": "track",
                    "artists": [{"name": "A"}],
                    "album": {"name": "X"},
                    "duration_ms": 1000,
                }
            }
        ],
        "next": None,
    }
    c, sleeps, _ = _client([_r429("1"), httpx.Response(200, json=page)])
    hits = c.get_playlist_tracks("https://open.spotify.com/playlist/abcdefghij", user_token="u")
    assert [h.spotify_id for h in hits] == ["p1"]
    assert len(sleeps) == 1


def test_429_on_token_endpoint_is_retried() -> None:
    c, sleeps, calls = _client([_r429("1", for_token=True), httpx.Response(200, json=_SEARCH_OK)])
    assert c.search_tracks("strobe")
    assert len(sleeps) == 1
    assert [r.url.path for r in calls] == ["/api/token", "/api/token", "/v1/search"]
