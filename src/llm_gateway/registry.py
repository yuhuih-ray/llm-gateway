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
    thinking_budget: int | None = None


# Thinking tokens count toward this limit and are billed as output tokens,
# so these limits are the main guard against runaway cost.
# Changing thinking settings changes quality, latency, and cost.
# These IDs accepted real generateContent requests, not just model-list queries.
# Every (upstream model, thinking level/budget) pair is a distinct configuration;
# training data and serving must use the same pair. A thinking variant can later
# be added as a separate gateway model name.
# Size deadlines for the 8192-token cap, approximately 1.5 * cap / measured
# throughput, rather than short sample maxima: 60s Flash tiers, 120s Pro.
# A too-short idle timeout breaks responses that cannot be retried after commit.
MODELS = {
    "fake": ModelEntry("fake", None, None, None, None, 30, 30),
    "gemini-flash-lite": ModelEntry(
        "gemini", "gemini-3.1-flash-lite", None, 2048, 8192, 60, 30, thinking_budget=0
    ),
    # Budget 0 was accepted but still produced reasoning; LOW is the lowest level.
    "gemini-flash": ModelEntry(
        "gemini", "gemini-3.8-flash", types.ThinkingLevel.LOW, 2048, 8192, 60, 30
    ),
    # Preview may retire on short notice; the live test detects availability changes.
    "gemini-pro": ModelEntry(
        "gemini", "gemini-3.1-pro-preview", types.ThinkingLevel.LOW, 2048, 8192, 120, 30
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
        thinking_budget=entry.thinking_budget,
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
