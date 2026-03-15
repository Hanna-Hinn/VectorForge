"""Request logging and error handling middleware for the API.

Implemented as pure ASGI middleware (not ``BaseHTTPMiddleware``) to avoid
the body-streaming deadlock that ``BaseHTTPMiddleware`` causes with
multipart file uploads when multiple middleware layers are stacked.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vectorforge.exceptions import (
    ConfigurationError,
    DatabaseError,
    DuplicateError,
    EmbeddingError,
    LLMError,
    NotFoundError,
    VectorForgeError,
)

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """Pure ASGI middleware — logs every HTTP request with method, path, status, and latency."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Log the request after the response is sent."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-Id", request_id)
            await send(message)

        await self.app(scope, receive, send_wrapper)

        latency_ms = (time.perf_counter() - start) * 1000
        method = scope.get("method", "?")
        path = scope.get("path", "?")
        logger.info(
            "%s %s → %d (%.1fms) [%s]",
            method,
            path,
            status_code,
            latency_ms,
            request_id,
        )


# ---------------------------------------------------------------------------
# Exception → HTTP status mapping
# ---------------------------------------------------------------------------

_EXCEPTION_STATUS_MAP: dict[type[VectorForgeError], int] = {
    NotFoundError: 404,
    DuplicateError: 409,
    ConfigurationError: 500,
    DatabaseError: 500,
    EmbeddingError: 502,
    LLMError: 502,
}

_EXCEPTION_CODE_MAP: dict[type[VectorForgeError], str] = {
    NotFoundError: "not_found",
    DuplicateError: "duplicate",
    ConfigurationError: "configuration_error",
    DatabaseError: "database_error",
    EmbeddingError: "embedding_error",
    LLMError: "llm_error",
}


class ErrorHandlerMiddleware:
    """Pure ASGI middleware — catches VectorForgeError exceptions and returns structured JSON."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wrap the downstream app, converting known exceptions to JSON."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except VectorForgeError as exc:
            if response_started:
                raise
            status_code = 400
            error_code = "error"
            for exc_cls, code in _EXCEPTION_STATUS_MAP.items():
                if isinstance(exc, exc_cls):
                    status_code = code
                    error_code = _EXCEPTION_CODE_MAP[exc_cls]
                    break
            logger.warning(
                "VectorForgeError (%s): %s", error_code, exc,
            )
            response = JSONResponse(
                status_code=status_code,
                content={
                    "error": error_code,
                    "message": str(exc),
                },
            )
            await response(scope, receive, send)
        except Exception as exc:
            if response_started:
                raise
            logger.exception("Unhandled exception: %s", exc)
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal",
                    "message": "An unexpected error occurred",
                },
            )
            await response(scope, receive, send)
