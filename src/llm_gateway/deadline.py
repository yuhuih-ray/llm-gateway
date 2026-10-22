import asyncio
from contextvars import ContextVar

from llm_gateway.errors import GatewayError

# One absolute deadline follows retries and fallback attempts in this request.
current_deadline: ContextVar[asyncio.Timeout | None] = ContextVar(
    "request_deadline", default=None
)


def exceeded() -> GatewayError:
    return GatewayError(
        504, "Request deadline exceeded", "upstream_timeout", breaker_failure=True
    )


def remaining() -> float | None:
    deadline = current_deadline.get()
    when = deadline.when() if deadline is not None else None
    return None if when is None else when - asyncio.get_running_loop().time()


def check_deadline() -> None:
    left = remaining()
    if left is not None and left <= 0:
        raise exceeded()


def limited_timeout(timeout: float) -> float:  # Local wait limit in seconds.
    check_deadline()
    left = remaining()
    return timeout if left is None else min(timeout, left)


def commit() -> None:
    # After the first event/full completion, only the stream idle timeout applies.
    check_deadline()
    deadline = current_deadline.get()
    if deadline is not None:
        deadline.reschedule(None)
