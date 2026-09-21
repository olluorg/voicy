"""Optional API key, the way OpenAI clients already send one.

Off by default: a server on localhost has nobody to keep out. Set
`VOICY_API_KEY` — one key or several, comma-separated — and every `/v1/`
route, WebSocket included, wants

    Authorization: Bearer <key>

which is exactly what `OpenAI(api_key=...)` sends. Browsers cannot put headers
on an EventSource or a WebSocket, so `?api_key=<key>` is accepted as well;
a header stays preferable, since query strings end up in logs.

`/health`, the console page and its static files stay open: the page asks for
the key itself, and the health check must work for docker and the CLI.
"""
from __future__ import annotations

import hmac
import os
from urllib.parse import parse_qs

import errors

KEYS = tuple(k.strip() for k in os.environ.get("VOICY_API_KEY", "").split(",") if k.strip())


def enabled() -> bool:
    return bool(KEYS)


def _valid(key: str) -> bool:
    return any(hmac.compare_digest(key.encode(), k.encode()) for k in KEYS)


def _key(scope) -> str:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            v = value.decode("latin-1")
            return v[7:].strip() if v.lower().startswith("bearer ") else v.strip()
        if name == b"x-api-key":
            return value.decode("latin-1").strip()
    q = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    return (q.get("api_key") or [""])[0]


class Middleware:
    """Pure ASGI, so it guards WebSockets too — HTTP middleware never sees them."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (not KEYS or scope["type"] not in ("http", "websocket")
                or not scope.get("path", "").startswith("/v1/")
                or scope.get("method") == "OPTIONS"):
            return await self.app(scope, receive, send)
        key = _key(scope)
        if key and _valid(key):
            return await self.app(scope, receive, send)

        message = ("Incorrect API key provided" if key else
                   "Missing API key: pass 'Authorization: Bearer <key>'")
        if scope["type"] == "websocket":
            # закрытие до accept сервер отдаёт клиенту как HTTP 403
            await send({"type": "websocket.close", "code": 1008, "reason": message})
            return
        resp = errors.response(401, message, headers={"WWW-Authenticate": "Bearer"})
        await resp(scope, receive, send)
