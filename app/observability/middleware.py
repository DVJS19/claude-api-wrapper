import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability.logger import get_logger

log = get_logger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Writes one structured audit log event per request covering:
    method, path, status_code, latency_ms, client_id (from JWT if present),
    and request_id for correlation with route-level logs.

    Runs after the route handler — status_code is the actual response code.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Attach request_id to request state so route handlers can read it
        request.state.request_id = request_id

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Extract client_id from response headers if the route set it
        # (routes add X-Client-ID header for audit correlation)
        client_id = response.headers.get("X-Client-ID", "unauthenticated")

        log.info(
            "audit_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            client_id=client_id,
        )

        # Add tracing headers to every response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-MS"] = str(latency_ms)

        return response
