from typing import Literal
from uuid import uuid4

from google import genai
from google.genai import types

from llm_gateway.config import get_settings
from llm_gateway.errors import GatewayError
from llm_gateway.retries import with_retries
from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    CompletionTokensDetails,
    Usage,
)

_shared_client: genai.client.AsyncClient | None = None


def get_gemini_client() -> genai.client.AsyncClient:
    global _shared_client
    if _shared_client is None:
        settings = get_settings()
        if settings.gemini_api_key is None:
            raise GatewayError(
                502, "Gemini provider is not configured", "upstream_error"
            )
        # No await during initialization: concurrent requests on the app loop share it.
        # Disable SDK retries: the gateway owns all retry attempts and backoff.
        _shared_client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        ).aio
    return _shared_client


async def close_gemini_client() -> None:
    global _shared_client
    client, _shared_client = _shared_client, None
    if client is not None:
        await client.aclose()


def map_finish_reason(
    reason: types.FinishReason | None,  # Explicit upstream terminal reason.
) -> Literal["stop", "length", "content_filter"]:
    if reason == types.FinishReason.STOP:
        return "stop"
    if reason == types.FinishReason.MAX_TOKENS:
        return "length"
    if reason in {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.SPII,
        types.FinishReason.IMAGE_SAFETY,
        types.FinishReason.IMAGE_PROHIBITED_CONTENT,
        types.FinishReason.IMAGE_RECITATION,
    }:
        return "content_filter"
    raise GatewayError(
        502, "Upstream returned an unsupported completion", "upstream_error"
    )


class GeminiProvider:
    def __init__(
        self,
        model_id: str,  # Verified upstream model ID.
        thinking_level: types.ThinkingLevel
        | None = None,  # Fixed registry policy; None uses model default.
        client: genai.client.AsyncClient | None = None,  # Injectable async SDK client.
        timeout: float | None = None,  # Optional per-attempt deadline override.
        default_max_tokens: int | None = None,  # Registry default output budget.
        max_tokens_cap: int | None = None,  # Registry maximum output budget.
    ) -> None:
        self.model_id = model_id
        self.thinking_level = thinking_level
        self.client = client
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.max_tokens_cap = max_tokens_cap

    async def complete(
        self,
        request: ChatCompletionRequest,  # Validated gateway request.
    ) -> ChatCompletionResponse:
        limit = (
            request.max_tokens
            if request.max_tokens is not None
            else self.default_max_tokens
        )
        if limit is not None and self.max_tokens_cap is not None:
            limit = min(limit, self.max_tokens_cap)
        request = request.model_copy(update={"max_tokens": limit})
        settings = get_settings()
        timeout = (
            self.timeout
            if self.timeout is not None
            else settings.gemini_timeout_seconds
        )
        client = self.client if self.client is not None else get_gemini_client()
        return await self._complete(request, client, timeout)

    async def _complete(
        self,
        request: ChatCompletionRequest,  # Gateway model name stays in the response.
        client: genai.client.AsyncClient,  # Async-only SDK interface.
        timeout: float,  # Deadline for each individual attempt.
    ) -> ChatCompletionResponse:
        systems = [
            message.content for message in request.messages if message.role == "system"
        ]
        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part(text=message.content)],
            )
            for message in request.messages
            if message.role != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction="\n".join(systems) if systems else None,
            max_output_tokens=request.max_tokens,
            temperature=request.temperature,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
            if self.thinking_level is not None
            else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        response = await with_retries(
            lambda: client.models.generate_content(
                model=self.model_id, contents=contents, config=config
            ),
            timeout,
        )
        if not response.candidates:
            raise GatewayError(502, "Upstream returned no completion", "upstream_error")
        candidate = response.candidates[0]
        finish_reason = map_finish_reason(candidate.finish_reason)
        parts = candidate.content.parts or [] if candidate.content else []
        content = "".join(part.text for part in parts if part.text and not part.thought)
        usage = response.usage_metadata
        reasoning_tokens = usage.thoughts_token_count or 0 if usage else 0
        visible_tokens = usage.candidates_token_count or 0 if usage else 0
        return ChatCompletionResponse(
            id=response.response_id or f"chatcmpl-{uuid4()}",
            model=request.model,
            choices=[
                Choice(
                    message=AssistantMessage(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=usage.prompt_token_count or 0 if usage else 0,
                completion_tokens=visible_tokens + reasoning_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
                total_tokens=usage.total_token_count or 0 if usage else 0,
            ),
        )
