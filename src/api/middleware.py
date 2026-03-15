from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from monitoring.metrics import HTTP_REQUEST_LATENCY, REQUEST_COUNT
from utils.logging import get_logger

log = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        correlation_id = request.headers.get("x-correlation-id", request_id)
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        bind_contextvars(request_id=request_id, correlation_id=correlation_id)
        start = time.perf_counter()
        response: Response | None = None
        exc: Exception | None = None
        try:
            response = await call_next(request)
        except Exception as e:  # pragma: no cover - passthrough path
            exc = e
        finally:
            elapsed = time.perf_counter() - start
            status_code = str(response.status_code) if response is not None else "500"
            REQUEST_COUNT.labels(path=str(request.url.path), status_code=status_code).inc()
            HTTP_REQUEST_LATENCY.labels(
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
            ).observe(elapsed)
            log.info(
                "http.request",
                request_id=request_id,
                correlation_id=correlation_id,
                method=request.method,
                path=str(request.url.path),
                elapsed_s=elapsed,
                status_code=status_code,
            )
            clear_contextvars()
        if exc is not None:
            raise exc
        if response is None:  # pragma: no cover - defensive
            raise RuntimeError("Request handling failed before a response was created")
        response.headers["x-request-id"] = request_id
        response.headers["x-correlation-id"] = correlation_id
        return response
