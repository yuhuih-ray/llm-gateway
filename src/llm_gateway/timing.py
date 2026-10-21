"""Optional request timings, with a fixed set of non-identifying stage names."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.config import get_settings

Stage = Literal["auth", "ratelimit", "budget", "cache", "provider"]
STAGES = ("auth", "ratelimit", "budget", "cache", "provider")


@contextmanager
def measure(
    request: Request,  # Request-local state; never include identifiers in headers.
    stage: Stage,  # Only the fixed stage names are accepted by callers.
) -> Iterator[None]:
    durations = request.scope.get("state", {}).get("server_timing")
    if durations is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        durations[stage] += (time.perf_counter() - started) * 1000


class ServerTimingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"] != "/v1/chat/completions"
            or not get_settings().server_timing
        ):
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        durations = dict.fromkeys(STAGES, 0.0)
        scope.setdefault("state", {})["server_timing"] = durations

        async def send_timed(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Headers precede body delivery and post-response usage accounting.
                # For SSE, provider measures opening through the first event only.
                values = {**durations, "total": (time.perf_counter() - started) * 1000}
                header = ", ".join(
                    f"{name};dur={value:.3f}" for name, value in values.items()
                )
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"server-timing", header.encode("ascii")),
                    ],
                }
            await send(message)

        await self.app(scope, receive, send_timed)
