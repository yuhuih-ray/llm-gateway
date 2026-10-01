from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends
from google.genai import types

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.errors import GatewayError
from llm_gateway.gemini import GeminiProvider
from llm_gateway.providers import FakeProvider, Provider
from llm_gateway.schemas import ChatCompletionRequest


@dataclass(frozen=True)
class ModelEntry:
    provider: str
    upstream_model: str | None
    thinking_level: types.ThinkingLevel | None
    default_max_tokens: int | None
    max_tokens_cap: int | None
    first_event_timeout: float
    idle_timeout: float


# Thinking tokens count toward this limit and are billed as output tokens,
# so these limits are the main guard against runaway cost.
# Changing thinking settings changes quality, latency, and cost.
# These IDs accepted real generateContent requests, not just model-list queries.
# Measured 2026-09-30: 3 prompts per model, both modes, default 1024 tokens.
# First timeout = ceil(3 * max(non-stream total, stream first text)):
# Flash 3.986s -> 12s; Pro 9.303s -> 28s.
# Idle = ceil(3 * worst stream event gap): Flash .162s / Pro .180s -> 1s.
# Small sample at default limits, not a latency guarantee for larger budgets.
# Fake has no upstream measurements; retain generous local-test deadlines.
MODELS = {
    "fake": ModelEntry("fake", None, None, None, None, 30, 30),
    "gemini-flash": ModelEntry(
        "gemini", "gemini-3.8-flash", types.ThinkingLevel.LOW, 1024, 8192, 12, 1
    ),
    # Preview may retire on short notice; the live test detects availability changes.
    "gemini-pro": ModelEntry(
        "gemini", "gemini-3.1-pro-preview", None, 1024, 8192, 28, 1
    ),
}


def get_provider(
    request: ChatCompletionRequest,  # Resolve the client's gateway model name.
    auth: Annotated[AuthContext, Depends(authenticate)],  # Authenticate before lookup.
) -> Provider:
    entry = MODELS.get(request.model)
    if entry is None:
        raise GatewayError(
            404, "Model not found", "invalid_request_error", "model_not_found"
        )
    if entry.provider == "fake":
        return FakeProvider()
    assert entry.upstream_model is not None
    return GeminiProvider(
        entry.upstream_model,
        thinking_level=entry.thinking_level,
        timeout=entry.first_event_timeout,
    )


def apply_token_limits(
    request: ChatCompletionRequest,  # Return a copy with the gateway's final budget.
) -> ChatCompletionRequest:
    entry = MODELS.get(request.model)
    if entry is None:
        raise GatewayError(
            404, "Model not found", "invalid_request_error", "model_not_found"
        )
    limit = (
        request.max_tokens
        if request.max_tokens is not None
        else entry.default_max_tokens
    )
    if limit is not None and entry.max_tokens_cap is not None:
        limit = min(limit, entry.max_tokens_cap)
    return request.model_copy(update={"max_tokens": limit})
