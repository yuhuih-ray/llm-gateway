import asyncio
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from time import monotonic
from typing import Any, Literal
from uuid import UUID, uuid4

import anyio
import httpx
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from pydantic import BaseModel
from starlette.datastructures import State
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.auth import AuthContext
from llm_gateway.config import get_settings
from llm_gateway.errors import GatewayError
from llm_gateway.registry import MODELS, ModelEntry
from llm_gateway.schemas import Usage

logger = logging.getLogger(__name__)


def calculate_cost(
    entry: ModelEntry,  # Verified registry prices in USD per million.
    prompt_tokens: int | None,  # Unknown counts must not become zero.
    completion_tokens: int | None,  # Already includes reasoning tokens.
) -> Decimal | None:
    if prompt_tokens is None or completion_tokens is None:
        return None
    input_price, output_price = entry.input_price, entry.output_price
    if prompt_tokens > 200_000 and entry.long_input_price is not None:
        input_price = entry.long_input_price
        assert entry.long_output_price is not None
        output_price = entry.long_output_price
    return (
        (
            Decimal(prompt_tokens) * input_price
            + Decimal(completion_tokens) * output_price
        )
        / Decimal(1_000_000)
    ).quantize(Decimal("0.00000001"))


class UsageRecord(BaseModel):
    request_id: UUID
    tenant_id: UUID
    api_key_id: UUID
    model_requested: str
    model_selected: str
    upstream_model: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: Decimal | None = None
    latency_ms: int = 0
    ttft_ms: int | None = None
    status: Literal["success", "error", "cancelled"] = "cancelled"
    error_type: str | None = None
    created_at: datetime


async def create_usage_pool() -> ArqRedis | None:
    settings = RedisSettings.from_dsn(get_settings().redis_url)
    settings.conn_retries = 0
    try:
        async with asyncio.timeout(0.5):
            return await create_pool(settings)
    except Exception:
        logger.error("Usage queue unavailable")
        return None


async def enqueue_usage(
    app_state: State,  # Shared lifespan state, including recovery throttle.
    record: UsageRecord,  # Fully priced in the gateway, serialized without floats.
) -> None:
    try:
        async with asyncio.timeout(0.5):
            pool = getattr(app_state, "usage_pool", None)
            if pool is None:
                now = monotonic()
                last_attempt = getattr(
                    app_state, "usage_pool_last_attempt", -float("inf")
                )
                if now - last_attempt >= 10:
                    # Set before awaiting so concurrent requests cannot start more attempts.
                    app_state.usage_pool_last_attempt = now
                    pool = await create_usage_pool()
                    app_state.usage_pool = pool
            if pool is None:
                raise ConnectionError("Queue unavailable")
            await pool.enqueue_job(
                "write_usage",
                record.model_dump(mode="json"),
                _job_id=str(record.request_id),
            )
    except Exception:
        logger.error("usage_record_dropped %s", record.model_dump_json())


class UsageTracker:
    def __init__(
        self,
        state: dict[str, Any],  # Request start metadata from ASGI middleware.
        auth: AuthContext,  # Authenticated identifiers only, never the API key.
        model: str,  # Validated registry name.
        app_state: State,  # Shared queue state, not a snapshot of the pool.
    ) -> None:
        self.started = state["started"]
        self.app_state = app_state
        self.emitted = False
        self.record = UsageRecord(
            request_id=state["request_id"],
            created_at=state["created_at"],
            tenant_id=auth.tenant_id,
            api_key_id=auth.api_key_id,
            model_requested=model,
            model_selected=model,
            upstream_model=MODELS[model].upstream_model,
        )

    def first_text(self) -> None:
        if self.record.ttft_ms is None:
            self.record.ttft_ms = int((time.monotonic() - self.started) * 1000)

    def success(self, usage: Usage | None) -> None:  # Only confirmed provider totals.
        self.record.status = "success"
        if usage is not None:
            self.record.prompt_tokens = usage.prompt_tokens
            self.record.completion_tokens = usage.completion_tokens
            self.record.reasoning_tokens = (
                usage.completion_tokens_details.reasoning_tokens
            )
        self.record.cost = calculate_cost(
            MODELS[self.record.model_selected],
            self.record.prompt_tokens,
            self.record.completion_tokens,
        )

    def failure(self, exc: BaseException) -> None:  # Sanitized classification only.
        self.record.status = "error"
        self.record.error_type = (
            "upstream_timeout"
            if isinstance(exc, (TimeoutError, httpx.TimeoutException))
            or isinstance(exc, GatewayError)
            and exc.status == 504
            else "upstream_error"
        )

    async def emit(self) -> None:
        if self.emitted:
            return
        self.emitted = True
        self.record.latency_ms = int((time.monotonic() - self.started) * 1000)
        with anyio.CancelScope(shield=True):
            await enqueue_usage(self.app_state, self.record)


class UsageMiddleware:
    def __init__(
        self, app: ASGIApp
    ) -> None:  # Pure ASGI preserves streaming/cancellation.
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state.update(
            request_id=uuid4(),
            created_at=datetime.now(timezone.utc),
            started=time.monotonic(),
        )

        async def send_with_id(message: Message) -> None:  # Add header before delivery.
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"x-request-id", str(state["request_id"]).encode()),
                    ],
                }
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            tracker = state.get("usage_tracker")
            if tracker is not None:
                await tracker.emit()
