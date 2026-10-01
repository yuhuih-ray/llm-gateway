import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from google.genai.errors import APIError

from llm_gateway.errors import GatewayError

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def with_retries(
    operation: Callable[[], Awaitable[T]],  # One asynchronous upstream attempt.
    timeout: float,  # Per-attempt deadline in seconds.
) -> T:
    for attempt in range(3):
        try:
            async with asyncio.timeout(timeout):
                return await operation()
        except (TimeoutError, ConnectionError, httpx.TransportError, APIError) as exc:
            # Read/deadline timeouts may follow billed generation; never repeat it.
            if isinstance(exc, (TimeoutError, httpx.ReadTimeout)):
                raise GatewayError(
                    504, "Upstream request timed out", "upstream_error"
                ) from None
            timed_out = isinstance(exc, httpx.ConnectTimeout)
            status = exc.code if isinstance(exc, APIError) else None
            if status in (401, 403):
                logger.error("Gateway provider credentials are invalid")
                raise GatewayError(
                    502, "Upstream authentication failed", "upstream_error"
                ) from None
            retryable = (
                timed_out
                or isinstance(exc, (ConnectionError, httpx.NetworkError))
                or status in (429, 500, 502, 503, 504)
            )
            if not retryable:
                if status == 400:
                    raise GatewayError(
                        400, "Invalid upstream request", "invalid_request_error"
                    ) from None
                raise GatewayError(
                    502, "Upstream request failed", "upstream_error"
                ) from None
            if attempt == 2:
                raise GatewayError(
                    504 if timed_out else 502,
                    "Upstream request timed out"
                    if timed_out
                    else "Upstream request failed",
                    "upstream_error",
                ) from None
            reason = (
                "timeout"
                if timed_out
                else f"HTTP {status}"
                if status
                else "connection error"
            )
            logger.warning("Upstream retry attempt=%s reason=%s", attempt + 2, reason)
            await asyncio.sleep(random.uniform(0, min(4.0, 0.5 * 2**attempt)))
    raise AssertionError("Unreachable")
