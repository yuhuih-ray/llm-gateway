"""Disposable exact responses; independent of durable queue/rate-limit Redis."""

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from time import monotonic
from uuid import UUID, uuid4

from pydantic import BaseModel
from redis.asyncio import Redis

from llm_gateway.config import get_settings
from llm_gateway.providers import FinalEvent, StreamEvent, TextEvent
from llm_gateway.registry import MODELS
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger(__name__)
last_warning = -float("inf")


def create_cache_client() -> Redis:
    return Redis.from_url(get_settings().cache_redis_url)


def cache_key(
    tenant_id: UUID,  # Tenant isolation even for identical prompts.
    request: ChatCompletionRequest,  # Full ordered context and applied token limits.
) -> str:
    entry = MODELS[request.model]
    # Gateway model separates public configurations; ordered roles/content preserve
    # the full context. Temperature and the applied max_tokens affect generation.
    # Include every generation field; transport options do not change output.
    payload = request.model_dump(exclude={"stream", "stream_options"})
    payload.update(
        tenant_id=str(tenant_id),
        # Registry changes must invalidate answers, including thinking variants.
        provider=entry.provider,
        upstream_model=entry.upstream_model,
        thinking_level=entry.thinking_level,
        thinking_budget=entry.thinking_budget,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "cache:v1:" + hashlib.sha256(canonical.encode()).hexdigest()


def cacheable(request: ChatCompletionRequest, cache_control: str) -> bool:
    return request.temperature == 0 and not any(
        directive.strip().split("=", 1)[0].lower() == "no-cache"
        for directive in cache_control.split(",")
    )


class CachedAnswer(BaseModel):
    response: ChatCompletionResponse
    upstream_model: str | None

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self.response.model_copy(update={"id": f"chatcmpl-{uuid4()}"})

    async def stream(
        self,
        request: ChatCompletionRequest,  # Replay through the normal SSE encoder.
    ) -> AsyncGenerator[StreamEvent, None]:
        choice = self.response.choices[0]
        yield TextEvent(choice.message.content)
        yield FinalEvent(choice.finish_reason, self.response.usage)


def warn_unavailable() -> None:
    global last_warning
    now = monotonic()
    if now - last_warning >= 30:
        last_warning = now
        logger.warning("Response cache unavailable; treating as miss")


async def read_cache(client: Redis | None, key: str) -> CachedAnswer | None:
    try:
        async with asyncio.timeout(0.05):
            if client is None:
                return None
            value = await client.get(key)
            if value is not None:
                answer = CachedAnswer.model_validate_json(value)
                if valid_response(answer.response):
                    return answer
    except Exception:
        warn_unavailable()
    return None


def valid_response(response: ChatCompletionResponse) -> bool:
    return len(response.choices) == 1 and response.choices[0].finish_reason in {
        "stop",
        "length",
    }


async def write_cache(
    client: Redis | None,  # Lifespan-owned disposable Redis connection pool.
    key: str,  # Digest only; never raw prompt text.
    answer: CachedAnswer,  # Only confirmed successful completions.
) -> None:
    try:
        async with asyncio.timeout(0.05):
            if client is not None and valid_response(answer.response):
                await client.set(key, answer.model_dump_json(), ex=86400)
    except Exception:
        warn_unavailable()
