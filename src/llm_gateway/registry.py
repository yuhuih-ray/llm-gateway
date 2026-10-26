from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated

from fastapi import Depends
from google.genai import types

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.errors import GatewayError
from llm_gateway.gemini import GeminiProvider
from llm_gateway.providers import FakeProvider, Provider
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.vllm import VllmProvider


@dataclass(frozen=True)
class ModelEntry:
    provider: str
    upstream_model: str | None
    thinking_level: types.ThinkingLevel | None
    default_max_tokens: int | None
    max_tokens_cap: int | None
    first_event_timeout: float
    idle_timeout: float
    max_input_chars: int = 100_000
    thinking_budget: int | None = None
    input_price: Decimal = Decimal("0")
    output_price: Decimal = Decimal("0")
    long_input_price: Decimal | None = None
    long_output_price: Decimal | None = None
    fallbacks: list[str] = field(default_factory=list)


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
# Standard USD per 1M text tokens, verified 2026-10-04:
# https://ai.google.dev/gemini-api/docs/pricing
# Flash promotional .75/3.75 expires 2026-12-31; reverify before 2027.
MODELS = {
    # Marginal cost is approximately zero; Phase 4 routing must consider local
    # capacity as well as price. Server max-model-len is 8192 tokens.
    "local-qwen": ModelEntry(
        "vllm",
        "qwen2.5-7b-awq",
        None,
        1024,
        4096,
        60,
        30,
        max_input_chars=20_000,
        fallbacks=["gemini-flash-lite"],
    ),
    "fake": ModelEntry("fake", None, None, None, None, 30, 30),
    "gemini-flash-lite": ModelEntry(
        "gemini",
        "gemini-3.1-flash-lite",
        None,
        2048,
        8192,
        60,
        30,
        thinking_budget=0,
        input_price=Decimal("0.25"),
        output_price=Decimal("1.50"),
    ),
    # Budget 0 was accepted but still produced reasoning; LOW is the lowest level.
    "gemini-flash": ModelEntry(
        "gemini",
        "gemini-3.8-flash",
        types.ThinkingLevel.LOW,
        2048,
        8192,
        60,
        30,
        input_price=Decimal("0.75"),
        output_price=Decimal("3.75"),
    ),
    # Preview may retire on short notice; the live test detects availability changes.
    "gemini-pro": ModelEntry(
        "gemini",
        "gemini-3.1-pro-preview",
        types.ThinkingLevel.LOW,
        2048,
        8192,
        120,
        30,
        input_price=Decimal("2.00"),
        output_price=Decimal("12.00"),
        long_input_price=Decimal("4.00"),
        long_output_price=Decimal("18.00"),
    ),
}


def get_provider(
    request: ChatCompletionRequest,  # Resolve the client's gateway model name.
    auth: Annotated[AuthContext, Depends(authenticate)],  # Authenticate before lookup.
) -> Provider | None:
    if request.model == "auto":
        # The endpoint resolves the artifact and hard filters before provider creation.
        return None
    return create_provider(request.model)


def create_provider(
    model: str,
) -> Provider:  # Validated gateway target, including fallback.
    entry = MODELS.get(model)
    if entry is None:
        raise GatewayError(
            404, "Model not found", "invalid_request_error", "model_not_found"
        )
    if entry.provider == "fake":
        return FakeProvider()
    assert entry.upstream_model is not None
    if entry.provider == "vllm":
        return VllmProvider(entry.upstream_model)
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
    # Characters, not estimated tokens: English averages ~4 chars/token, Chinese
    # ~1-1.5. 100,000 characters stays below Pro's 200K-token long-context tier
    # across languages; actual usage_logs prompt_tokens can calibrate this later.
    if (
        sum(len(message.content) for message in request.messages)
        > entry.max_input_chars
    ):
        raise GatewayError(
            400,
            "Input exceeds model character limit",
            "invalid_request_error",
            "context_length_exceeded",
        )
    limit = (
        request.max_tokens
        if request.max_tokens is not None
        else entry.default_max_tokens
    )
    if limit is not None and entry.max_tokens_cap is not None:
        limit = min(limit, entry.max_tokens_cap)
    return request.model_copy(update={"max_tokens": limit})
