"""Errors in the shape OpenAI clients expect.

FastAPI answers `{"detail": "..."}`; the official clients look for
`{"error": {"message", "type", "param", "code"}}` and, not finding it, show the
caller a bare status line. So every error — raised on purpose, a validation
failure, an unexpected exception — leaves the server in OpenAI's shape.

Validation failures become 400, as OpenAI returns them, not FastAPI's 422.
"""
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("voicy")

# статус → (type, code), как у OpenAI
KINDS = {
    400: ("invalid_request_error", None),
    401: ("invalid_request_error", "invalid_api_key"),
    404: ("invalid_request_error", "not_found"),
    405: ("invalid_request_error", "method_not_allowed"),
    409: ("invalid_request_error", "conflict"),
    410: ("invalid_request_error", "gone"),
    413: ("invalid_request_error", "too_large"),
    429: ("requests", "rate_limit_exceeded"),
}


def body(status: int, message: str, *, param: str | None = None,
         code: str | None = None) -> dict:
    kind, default_code = KINDS.get(status, ("server_error" if status >= 500
                                            else "invalid_request_error", None))
    return {"error": {"message": message, "type": kind, "param": param,
                      "code": code or default_code}}


def response(status: int, message: str, headers: dict | None = None, **kw) -> JSONResponse:
    return JSONResponse(body(status, message, **kw), status_code=status, headers=headers)


def attach(app) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return response(exc.status_code, message, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(x) for x in first.get("loc", ()) if x not in ("body", "query", "form")]
        param = ".".join(loc) or None
        message = first.get("msg", "invalid request")
        return response(400, f"{param}: {message}" if param else message, param=param)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return response(500, str(exc) or type(exc).__name__)
