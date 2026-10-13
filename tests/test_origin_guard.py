"""Origin / Host guard on the local API (``dance.api.origin_guard``).

Uses the real ``create_app`` factory with a socket-free fake bridge, so the
guard is exercised in its real position in front of CORS and the routers.
The app here pins the production host allowlist explicitly (conftest adds
``testserver`` for the rest of the suite) and talks to it as
``http://127.0.0.1:8000`` the way curl / the Vite proxy would.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from starlette.websockets import WebSocketDisconnect

from dance.api import create_app
from dance.api.origin_guard import OriginGuardMiddleware, host_name
from dance.config import (
    DEFAULT_API_ALLOWED_HOSTS,
    DEFAULT_API_ALLOWED_ORIGINS,
    Settings,
)
from dance.osc.bridge import AbletonState

BASE = "http://127.0.0.1:8000"
WS_BASE = "ws://127.0.0.1:8000"
COMPANION = "http://localhost:5173"
EVIL = "https://evil.example"


class _RecordingClient:
    """Records every OSC primitive the routers call, whatever its name."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Callable[..., None]:
        def _record(*_args: Any, **_kwargs: Any) -> None:
            self.calls.append(name)

        return _record


class _FakeBridge:
    def __init__(self) -> None:
        self.client = _RecordingClient()
        self.state = AbletonState(tempo=120.0, is_playing=False, beat=0.0)

    def subscribe(self, _listener: Callable[[AbletonState], None]) -> None:
        pass


def _settings(tmp_path, **overrides: Any) -> Settings:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = dict(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        # Production defaults, not conftest's testserver-widened list.
        api_allowed_hosts=list(DEFAULT_API_ALLOWED_HOSTS),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture
def bridge() -> _FakeBridge:
    return _FakeBridge()


@pytest.fixture
def make_client(tmp_path, session_factory: sessionmaker, bridge: _FakeBridge):
    clients: list[TestClient] = []

    def _make(**overrides: Any) -> TestClient:
        app = create_app(
            settings=_settings(tmp_path, **overrides),
            bridge=bridge,
            session_factory=session_factory,
        )
        c = TestClient(app, base_url=BASE)
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


# ---------------------------------------------------------------------------
# State-changing requests
# ---------------------------------------------------------------------------

# Empty-body POSTs that drive Live — exactly what a "simple" cross-origin
# request (no preflight) can reach.
_LIVE_POSTS = [
    ("/api/v1/ableton/play", "play"),
    ("/api/v1/ableton/transport/stop-all", "stop_all_clips"),
    ("/api/v1/ableton/transport/fire-clip/1/2", "fire_clip"),
]


@pytest.mark.parametrize("path,primitive", _LIVE_POSTS)
@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/x-www-form-urlencoded", None],
)
def test_cross_origin_simple_post_is_blocked_before_live(
    client: TestClient,
    bridge: _FakeBridge,
    path: str,
    primitive: str,
    content_type: str | None,
) -> None:
    headers = {"Origin": EVIL}
    if content_type:
        headers["Content-Type"] = content_type
    r = client.post(path, headers=headers, content=b"" if content_type else None)
    assert r.status_code == 403
    assert "DANCE_API_ALLOWED_ORIGINS" in r.json()["detail"]
    assert "access-control-allow-origin" not in r.headers
    assert bridge.client.calls == []


@pytest.mark.parametrize("path,primitive", _LIVE_POSTS)
def test_allowlisted_origin_post_reaches_live(
    client: TestClient, bridge: _FakeBridge, path: str, primitive: str
) -> None:
    r = client.post(path, headers={"Origin": COMPANION})
    assert r.status_code == 200, r.text
    assert bridge.client.calls == [primitive]
    # CORS still answers for the allowlisted origin.
    assert r.headers["access-control-allow-origin"] == COMPANION


def test_every_default_companion_origin_is_allowed(client: TestClient, bridge: _FakeBridge) -> None:
    for origin in DEFAULT_API_ALLOWED_ORIGINS:
        r = client.post("/api/v1/ableton/stop", headers={"Origin": origin})
        assert r.status_code == 200, (origin, r.text)
    assert bridge.client.calls == ["stop"] * len(DEFAULT_API_ALLOWED_ORIGINS)


def test_post_without_origin_is_allowed(client: TestClient, bridge: _FakeBridge) -> None:
    """curl, scripts, the CLI, a future MCP server: no Origin header."""
    r = client.post("/api/v1/ableton/transport/stop-all")
    assert r.status_code == 200
    assert bridge.client.calls == ["stop_all_clips"]


def test_null_origin_is_blocked(client: TestClient, bridge: _FakeBridge) -> None:
    """Sandboxed iframes and file:// pages send ``Origin: null``."""
    r = client.post("/api/v1/ableton/play", headers={"Origin": "null"})
    assert r.status_code == 403
    assert bridge.client.calls == []


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_other_mutating_methods_are_guarded(client: TestClient, method: str) -> None:
    r = client.request(method.upper(), "/api/v1/sets/1", headers={"Origin": EVIL})
    assert r.status_code == 403


def test_same_origin_post_is_allowed(client: TestClient, bridge: _FakeBridge) -> None:
    """FastAPI's own /docs page ("Try it out") posts with Origin == Host."""
    r = client.post("/api/v1/ableton/play", headers={"Origin": BASE})
    assert r.status_code == 200
    assert bridge.client.calls == ["play"]


def test_lookalike_origins_are_blocked(client: TestClient) -> None:
    for origin in (
        "http://localhost:5173.evil.example",
        "http://evil.localhost:5173",
        "https://localhost:5173",  # scheme matters
        "http://localhost:5175",  # port matters
        "http://127.0.0.1:8001",  # same host, other port: not same-origin
    ):
        r = client.post("/api/v1/ableton/play", headers={"Origin": origin})
        assert r.status_code == 403, origin


def test_origin_match_is_case_insensitive(client: TestClient) -> None:
    r = client.post("/api/v1/ableton/play", headers={"Origin": "HTTP://LocalHost:5173"})
    assert r.status_code == 200


def test_configured_origin_tolerates_trailing_slash(make_client) -> None:
    c = make_client(api_allowed_origins=["http://localhost:5175/"])
    r = c.post("/api/v1/ableton/play", headers={"Origin": "http://localhost:5175"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Host (DNS rebinding)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1:8000", "127.0.0.1", "localhost:8000", "LOCALHOST:5173", "[::1]:8000", "[::1]"],
)
def test_loopback_hosts_are_allowed(client: TestClient, host: str) -> None:
    r = client.get("/health", headers={"Host": host})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize(
    "host",
    ["evil.example:8000", "evil.example", "192.168.1.20:8000", "0.0.0.0:8000", "testserver"],
)
def test_foreign_host_is_blocked(client: TestClient, bridge: _FakeBridge, host: str) -> None:
    assert client.get("/health", headers={"Host": host}).status_code == 403
    r = client.post("/api/v1/ableton/play", headers={"Host": host})
    assert r.status_code == 403
    assert "DANCE_API_ALLOWED_HOSTS" in r.json()["detail"]
    assert bridge.client.calls == []


def test_dns_rebinding_is_blocked_even_though_it_looks_same_origin(
    client: TestClient, bridge: _FakeBridge
) -> None:
    """After rebinding, the attacker's page is "same-origin" with the API under
    the attacker's hostname. Host is what gives it away."""
    r = client.post(
        "/api/v1/ableton/transport/stop-all",
        headers={"Host": "rebind.evil.example:8000", "Origin": "http://rebind.evil.example:8000"},
    )
    assert r.status_code == 403
    assert bridge.client.calls == []


def test_extra_allowed_host_is_configurable(make_client) -> None:
    c = make_client(api_allowed_hosts=[*DEFAULT_API_ALLOWED_HOSTS, "dance-mac.local"])
    assert c.get("/health", headers={"Host": "dance-mac.local:8000"}).status_code == 200
    assert c.get("/health", headers={"Host": "evil.example"}).status_code == 403


# ---------------------------------------------------------------------------
# Cross-origin reads — decided: blocked too (see origin_guard docstring)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/health", "/api/v1/tracks", "/api/v1/ableton/state"])
def test_cross_origin_get_is_blocked(client: TestClient, path: str) -> None:
    r = client.get(path, headers={"Origin": EVIL})
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers


@pytest.mark.parametrize("path", ["/health", "/api/v1/tracks", "/api/v1/ableton/state"])
def test_allowlisted_and_originless_gets_are_allowed(client: TestClient, path: str) -> None:
    r = client.get(path, headers={"Origin": COMPANION})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == COMPANION
    assert client.get(path).status_code == 200


def test_cors_preflight_from_companion_still_works(client: TestClient) -> None:
    preflight = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    r = client.options("/api/v1/ableton/play", headers={"Origin": COMPANION, **preflight})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == COMPANION

    r = client.options("/api/v1/ableton/play", headers={"Origin": EVIL, **preflight})
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers


# ---------------------------------------------------------------------------
# WebSockets — not covered by CORS at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/ws", "/ws/pipeline"])
def test_websocket_with_foreign_origin_is_refused(client: TestClient, path: str) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"{WS_BASE}{path}", headers={"Origin": EVIL}):
            pass
    assert exc.value.code == 1008


def test_websocket_with_foreign_host_is_refused(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "ws://rebind.evil.example:8000/ws",
            headers={"Origin": "http://rebind.evil.example:8000"},
        ):
            pass
    assert exc.value.code == 1008


def test_websocket_with_companion_origin_is_accepted(client: TestClient) -> None:
    # Through the Vite proxy the backend sees Host localhost:5173 (ws proxy
    # keeps the Host) and the page's Origin.
    with client.websocket_connect("ws://localhost:5173/ws", headers={"Origin": COMPANION}) as ws:
        snapshot = ws.receive_json()
    assert snapshot["tempo"] == 120.0


def test_websocket_without_origin_is_accepted(client: TestClient) -> None:
    with client.websocket_connect(f"{WS_BASE}/ws") as ws:
        assert ws.receive_json()["tempo"] == 120.0


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_defaults_are_loopback_and_companion_only(monkeypatch) -> None:
    monkeypatch.delenv("DANCE_API_ALLOWED_HOSTS", raising=False)
    s = Settings()
    assert s.api_allowed_hosts == ["127.0.0.1", "localhost", "::1"]
    assert s.api_allowed_origins == list(DEFAULT_API_ALLOWED_ORIGINS)


def test_env_lists_accept_csv_and_json(monkeypatch) -> None:
    monkeypatch.setenv(
        "DANCE_API_ALLOWED_ORIGINS", " http://localhost:5173 , http://localhost:5175 ,"
    )
    monkeypatch.setenv("DANCE_API_ALLOWED_HOSTS", '["127.0.0.1", "dance-mac.local"]')
    s = Settings()
    assert s.api_allowed_origins == ["http://localhost:5173", "http://localhost:5175"]
    assert s.api_allowed_hosts == ["127.0.0.1", "dance-mac.local"]


def test_configured_origin_is_allowed_by_guard_and_cors(make_client, monkeypatch) -> None:
    monkeypatch.setenv("DANCE_API_ALLOWED_ORIGINS", "http://localhost:5175")
    c = make_client(api_allowed_origins=Settings().api_allowed_origins)
    r = c.post("/api/v1/ableton/play", headers={"Origin": "http://localhost:5175"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5175"
    # The env value replaces the defaults.
    assert c.post("/api/v1/ableton/play", headers={"Origin": COMPANION}).status_code == 403


def test_wildcards_switch_checks_off(make_client) -> None:
    c = make_client(api_allowed_hosts=["*"], api_allowed_origins=["*"])
    r = c.post("/api/v1/ableton/play", headers={"Host": "evil.example", "Origin": EVIL})
    assert r.status_code == 200


def test_host_wildcard_drops_the_same_origin_allowance(make_client, bridge: _FakeBridge) -> None:
    """With the host check off, a rebinding page would be "same-origin" with
    its own foreign Host — the origin check must still refuse it."""
    c = make_client(api_allowed_hosts=["*"])
    rebound = {"Host": "rebind.evil.example:8000", "Origin": "http://rebind.evil.example:8000"}
    assert c.post("/api/v1/ableton/play", headers=rebound).status_code == 403
    assert c.post("/api/v1/ableton/play", headers={"Origin": BASE}).status_code == 403
    assert c.post("/api/v1/ableton/play", headers={"Origin": COMPANION}).status_code == 200
    assert bridge.client.calls == ["play"]


# ---------------------------------------------------------------------------
# Unit: Host parsing and non-http scopes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("127.0.0.1:8000", "127.0.0.1"),
        ("LocalHost", "localhost"),
        ("localhost.:8000", "localhost"),
        ("[::1]:8000", "::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),
        ("evil.example:8000", "evil.example"),
        ("", None),
        ("[::1", None),
        ("[::1]x", None),
        ("localhost:http", None),
        (":8000", None),
    ],
)
def test_host_name(value: str, expected: str | None) -> None:
    assert host_name(value) == expected


def test_lifespan_scope_passes_through() -> None:
    import asyncio

    seen: list[str] = []

    async def inner(scope, receive, send) -> None:  # noqa: ANN001
        seen.append(scope["type"])

    guard = OriginGuardMiddleware(inner)

    async def _noop_receive() -> dict:
        return {}

    async def _noop_send(_message) -> None:  # noqa: ANN001
        pass

    asyncio.run(guard({"type": "lifespan"}, _noop_receive, _noop_send))
    assert seen == ["lifespan"]


def test_missing_host_header_is_not_a_browser_and_passes() -> None:
    guard = OriginGuardMiddleware(lambda *_: None)
    assert guard.check({"type": "http", "headers": []}) is None
    # ...but a foreign Origin is still refused without a Host to be same-origin with.
    assert guard.check({"type": "http", "headers": [(b"origin", EVIL.encode())]}) is not None
