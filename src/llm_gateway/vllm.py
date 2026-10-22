import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, Literal, cast
from uuid import uuid4

import anyio
import httpx
from pydantic import ValidationError

from llm_gateway.config import get_settings
from llm_gateway.deadline import limited_timeout
from llm_gateway.errors import GatewayError
from llm_gateway.providers import FinalEvent, StreamEvent, TextEvent
from llm_gateway.retries import with_retries
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse, Usage

_shared_client: httpx.AsyncClient | None = None


def get_vllm_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        settings = get_settings()
        if not settings.vllm_base_url or settings.vllm_api_key is None:
            raise GatewayError(502, "vLLM provider is not configured", "upstream_error")
        # A sleeping or offline machine must fail fast. Registry deadlines govern
        # reads; disabling HTTPX's read timer avoids conflicting first/idle limits.
        _shared_client = httpx.AsyncClient(
            base_url=settings.vllm_base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {settings.vllm_api_key.get_secret_value()}"
            },
            timeout=httpx.Timeout(connect=2, read=None, write=2, pool=2),
        )
    return _shared_client


async def close_vllm_client() -> None:
    global _shared_client
    client, _shared_client = _shared_client, None
    if client is not None:
        await client.aclose()


def finish_reason(value: Any) -> Literal["stop", "length", "content_filter"]:
    if value not in ("stop", "length", "content_filter"):
        raise GatewayError(
            502, "Upstream returned an unsupported completion", "upstream_error"
        )
    return cast(Literal["stop", "length", "content_filter"], value)


def map_usage(
    value: dict[str, Any] | None,  # Upstream totals; absent counts remain unknown.
) -> Usage:
    if value is None:
        return Usage(
            prompt_tokens=None,
            completion_tokens=None,
            completion_tokens_details={"reasoning_tokens": None},
        )
    return Usage(
        prompt_tokens=value.get("prompt_tokens"),
        completion_tokens=value.get("completion_tokens"),
        total_tokens=value.get("total_tokens") or 0,
        completion_tokens_details=value.get("completion_tokens_details") or {},
    )


class VllmProvider:
    def __init__(
        self,
        model_id: str,  # Registry's served model name, not the gateway alias.
        client: httpx.AsyncClient | None = None,  # Injectable MockTransport client.
    ) -> None:
        self.model_id = model_id
        self.client = client

    def payload(self, request: ChatCompletionRequest, stream: bool) -> dict[str, Any]:
        value = request.model_dump(exclude_none=True, exclude={"stream_options"})
        value.update(model=self.model_id, stream=stream)
        if stream:
            value["stream_options"] = {"include_usage": True}
        return value

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        from llm_gateway.registry import MODELS

        client = self.client if self.client is not None else get_vllm_client()

        async def attempt() -> ChatCompletionResponse:
            response = await client.post(
                "chat/completions", json=self.payload(request, False)
            )
            try:
                response.raise_for_status()
                value = response.json()
                if not value.get("choices"):
                    raise ValueError("Missing choices")
                choices = [
                    dict(
                        message={
                            "role": "assistant",
                            "content": choice["message"].get("content") or "",
                        },
                        finish_reason=finish_reason(choice.get("finish_reason")),
                    )
                    for choice in value["choices"]
                ]
                return ChatCompletionResponse(
                    id=value.get("id") or f"chatcmpl-{uuid4()}",
                    model=request.model,
                    choices=choices,
                    usage=map_usage(value.get("usage")),
                )
            except (ValueError, TypeError, KeyError, AttributeError, ValidationError):
                raise GatewayError(
                    502, "Upstream returned an invalid completion", "upstream_error"
                ) from None
            finally:
                with anyio.CancelScope(shield=True):
                    await response.aclose()

        return await with_retries(attempt, MODELS[request.model].first_event_timeout)

    async def stream(
        self,
        request: ChatCompletionRequest,  # Gateway alias and applied generation limits.
    ) -> AsyncGenerator[StreamEvent, None]:
        # stream_response owns retries until the first event; never retry after text.
        from llm_gateway.registry import MODELS

        entry = MODELS[request.model]
        client = self.client if self.client is not None else get_vllm_client()
        upstream = None
        deadline = asyncio.get_running_loop().time() + limited_timeout(
            entry.first_event_timeout
        )
        try:
            async with asyncio.timeout(limited_timeout(entry.first_event_timeout)):
                upstream = await client.send(
                    client.build_request(
                        "POST", "chat/completions", json=self.payload(request, True)
                    ),
                    stream=True,
                )
            upstream.raise_for_status()
            lines = upstream.aiter_lines()
            usage = None
            reason = None
            while True:
                async with asyncio.timeout(
                    max(0, deadline - asyncio.get_running_loop().time())
                ):
                    line = await anext(lines, None)
                if line is None:
                    raise GatewayError(
                        502, "Upstream stream ended unexpectedly", "upstream_error"
                    )
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    if reason is None:
                        raise GatewayError(
                            502, "Upstream stream ended unexpectedly", "upstream_error"
                        )
                    yield FinalEvent(reason, usage)
                    return
                value = json.loads(data)
                if "error" in value:
                    raise GatewayError(502, "Upstream stream failed", "upstream_error")
                if value.get("usage") is not None:
                    usage = map_usage(value["usage"])
                for choice in value.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    if choice.get("finish_reason") is not None:
                        reason = finish_reason(choice["finish_reason"])
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if not isinstance(content, str):
                            raise ValueError("Invalid content")
                        yield TextEvent(content)
                        deadline = (
                            asyncio.get_running_loop().time() + entry.idle_timeout
                        )
        except (ValueError, TypeError, KeyError, AttributeError, ValidationError):
            raise GatewayError(
                502, "Upstream returned an invalid stream", "upstream_error"
            ) from None
        finally:
            if upstream is not None:
                with anyio.CancelScope(shield=True):
                    await upstream.aclose()
