"""Origin / Host guard for the local API and its websockets.

Why this exists
---------------
``CORSMiddleware`` only decides whether a browser lets a page *read* a
cross-origin response. It does not stop the browser from *sending* the
request: a "simple" cross-origin POST (no body, or a text/plain / form
body) goes out without a preflight, so any web page the user visits could
``fetch("http://127.0.0.1:8000/api/v1/ableton/transport/stop-all",
{method: "POST", mode: "no-cors"})`` and drive Live mid-set. WebSockets are
not covered by CORS at all, and DNS rebinding lets a foreign page reach the
loopback port under its own hostname, which a browser then sends as ``Host``.

Policy (applies to every HTTP request and every websocket handshake)
--------------------------------------------------------------------
1. **Host must be loopback.** The hostname part of ``Host`` (any port) must
   be in ``allowed_hosts`` (default ``127.0.0.1``, ``localhost``, ``::1``).
   A rebinding attack always presents the attacker's hostname, so it stops
   here. A request with no ``Host`` at all passes: every browser sends one,
   so a Host-less request is a raw script, not a web page.
2. **Origin, when present, must be trusted.** Either listed in
   ``allowed_origins`` (the companion dev origins; also used as the CORS
   allowlist) or same-origin with the loopback ``Host`` it arrived on (e.g.
   FastAPI's own ``/docs`` page "Try it out"). ``Origin: null`` (sandboxed
   iframes, ``file://`` pages) is never trusted.
3. **No Origin passes.** curl, scripts, the CLI, a future MCP server: none
   of them send ``Origin``. Browsers omit it only on same-origin GET/HEAD and
   on no-cors subresource loads (``<img>``, ``<audio>``), whose responses the
   page cannot read anyway.

Cross-origin reads are blocked too, not only POST/PUT/PATCH/DELETE: several
GETs do real work (waveform computation and caching, Spotify searches on the
user's credentials, OSC queries against Live), so a foreign page must not be
able to trigger them from script either. The companion app never sends a
foreign Origin — Vite proxies ``/api`` and ``/ws`` same-origin, and the page
origin (``http://localhost:5173``) is allowlisted.

Rejections are 403 for HTTP (JSON ``detail`` names the setting to change)
and close-before-accept (code 1008, which the server turns into an HTTP 403
handshake response) for websockets. Both are logged at WARNING.

A ``"*"`` entry in either list switches that check off.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dance.config import DEFAULT_API_ALLOWED_HOSTS, DEFAULT_API_ALLOWED_ORIGINS

logger = logging.getLogger(__name__)

# RFC 6455 "policy violation" — the websocket analog of 403.
WS_POLICY_VIOLATION = 1008


def host_name(host_header: str) -> str | None:
    """Return the lower-cased hostname of a ``Host`` header value.

    Strips the port and IPv6 brackets: ``"[::1]:8000"`` -> ``"::1"``,
    ``"LocalHost:5173"`` -> ``"localhost"``. Returns ``None`` when the value
    is malformed (empty, unterminated bracket, non-numeric port), which the
    guard treats as not allowed.
    """

    value = host_header.strip().lower()
    if not value:
        return None
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return None
        rest = value[end + 1 :]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return None
        return value[1:end] or None
    if value.count(":") > 1:
        # Unbracketed IPv6 is not a valid Host value, but take it literally:
        # "::1" still matches the allowlist and anything with a port won't.
        return value
    name, sep, port = value.partition(":")
    if sep and not port.isdigit():
        return None
    # "localhost." is the same name as "localhost".
    return name.rstrip(".") or None


def _normalize_origin(origin: str) -> str:
    return origin.strip().lower().rstrip("/")


class OriginGuardMiddleware:
    """Pure-ASGI middleware enforcing the policy in the module docstring.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so it also sees websocket
    scopes, and so it adds no task/stream wrapping around every response.
    Add it *after* ``CORSMiddleware`` so it runs first (outermost).
    """

    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: Iterable[str] = DEFAULT_API_ALLOWED_HOSTS,
        allowed_origins: Iterable[str] = DEFAULT_API_ALLOWED_ORIGINS,
    ) -> None:
        self.app = app
        hosts = [h.strip().lower().strip("[]") for h in allowed_hosts if h.strip()]
        origins = [_normalize_origin(o) for o in allowed_origins if o.strip()]
        self.any_host = "*" in hosts
        self.any_origin = "*" in origins
        self.allowed_hosts = frozenset(hosts)
        self.allowed_origins = frozenset(origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        reason = self.check(scope)
        if reason is None:
            await self.app(scope, receive, send)
            return

        if scope["type"] == "http":
            logger.warning(
                "origin guard: blocked %s %s: %s",
                scope.get("method", "?"),
                scope.get("path", "?"),
                reason,
            )
            await _send_403(send, reason)
        else:
            logger.warning("origin guard: refused websocket %s: %s", scope.get("path", "?"), reason)
            # ASGI: answer the connect event with a close instead of an accept;
            # the server rejects the handshake with HTTP 403.
            message = await receive()
            if message["type"] == "websocket.connect":
                await send({"type": "websocket.close", "code": WS_POLICY_VIOLATION})

    def check(self, scope: Scope) -> str | None:
        """Return why ``scope`` must be refused, or ``None`` to let it through."""

        hosts: list[str] = []
        origins: list[str] = []
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name == b"host":
                hosts.append(raw_value.decode("latin-1"))
            elif raw_name == b"origin":
                origins.append(raw_value.decode("latin-1"))

        if not self.any_host:
            for host in hosts:
                if host_name(host) not in self.allowed_hosts:
                    return (
                        f"Host {host!r} is not a loopback address. Add the hostname "
                        "to DANCE_API_ALLOWED_HOSTS if you serve the API on purpose "
                        "under another name."
                    )

        if not self.any_origin:
            # Same-origin only means "a page this server served" once Host has
            # been vetted above. With the host check off, a rebinding page is
            # same-origin with its own foreign Host, so the allowance goes too.
            same_origin = (
                set()
                if self.any_host
                else {
                    f"{scheme}://{host.strip().lower()}"
                    for host in hosts
                    for scheme in ("http", "https")
                }
            )
            for origin in origins:
                normalized = _normalize_origin(origin)
                if normalized in self.allowed_origins or normalized in same_origin:
                    continue
                return (
                    f"Origin {origin!r} is not allowed to call this API. Add it to "
                    "DANCE_API_ALLOWED_ORIGINS if it is a companion app you run."
                )
        return None


async def _send_403(send: Send, reason: str) -> None:
    body = json.dumps({"detail": reason}).encode("utf-8")
    start: Message = {
        "type": "http.response.start",
        "status": 403,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    }
    await send(start)
    await send({"type": "http.response.body", "body": body})


__all__ = ["OriginGuardMiddleware", "host_name"]
