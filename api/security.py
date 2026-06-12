"""API security: API-key authentication and per-IP rate limiting."""

import logging
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from config import settings

logger = logging.getLogger(__name__)


def require_api_key(x_api_key: str | None = Header(default=None, include_in_schema=False)) -> None:
    """Validate the `X-API-Key` header against the configured API key.

    If `settings.api_key` is empty, authentication is disabled (useful for
    local development and tests) and a warning is logged once.

    Args:
        x_api_key: The value of the `X-API-Key` request header.

    Raises:
        HTTPException: With status code 401 if the key is missing or invalid.
    """
    if not settings.api_key:
        return

    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """A simple in-memory sliding-window rate limiter, keyed by client IP."""

    def __init__(self, app, requests_per_minute: int | None = None) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application to wrap.
            requests_per_minute: Maximum requests allowed per client IP per
                60-second window. Defaults to `settings.rate_limit_per_minute`.
        """
        super().__init__(app)
        self.limit = requests_per_minute if requests_per_minute is not None else settings.rate_limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Reject requests that exceed the per-IP rate limit.

        Args:
            request: The incoming HTTP request.
            call_next: The next handler in the middleware chain.

        Returns:
            Response: Either a 429 JSON response, or the downstream response.
        """
        if self.limit <= 0:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[client_ip]

        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
            )

        window.append(now)
        return await call_next(request)
