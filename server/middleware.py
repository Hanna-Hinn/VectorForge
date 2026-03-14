"""Request logging and error handling middleware for the API."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

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


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and latency."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request and log timing information.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler.

        Returns:
            The HTTP response.
        """
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s → %d (%.1fms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request_id,
        )
        response.headers["X-Request-Id"] = request_id
        return response


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


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch VectorForgeError exceptions and return structured JSON errors."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request, converting exceptions to JSON responses.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler.

        Returns:
            A JSON error response or the normal response.
        """
        try:
            return await call_next(request)
        except VectorForgeError as exc:
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
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": error_code,
                    "message": str(exc),
                },
            )
        except Exception as exc:
            logger.exception("Unhandled exception: %s", exc)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal",
                    "message": "An unexpected error occurred",
                },
            )
