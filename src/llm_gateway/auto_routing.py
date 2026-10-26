"""Lazy artifact resolution and hard input filters for model=auto."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from llm_gateway.config import get_settings
from llm_gateway.errors import GatewayError
from llm_gateway.registry import MODELS, apply_token_limits
from llm_gateway.router import Router
from llm_gateway.schemas import ChatCompletionRequest

QUALITIES = {"economy", "balanced", "high"}


@lru_cache(maxsize=1)
def load_router(
    path: Path,
) -> Router:  # One validated artifact per process/configured path.
    router = Router(path)
    if not isinstance(router.run_name, str) or not router.run_name:
        raise ValueError("Artifact run name is missing")
    if not QUALITIES.issubset(router.levels) or any(
        model not in MODELS for model in router.models
    ):
        raise ValueError("Artifact models or quality levels are invalid")
    return router


def auto_targets(
    request: ChatCompletionRequest,  # Full context is checked; only the last user text is scored.
    quality: str,
) -> tuple[list[str], dict[str, Any]]:
    if quality not in QUALITIES:
        raise GatewayError(
            400,
            "X-Gateway-Quality must be economy, balanced, or high",
            "invalid_request_error",
            "invalid_quality",
        )
    question = next(
        (
            message.content
            for message in reversed(request.messages)
            if message.role == "user"
        ),
        None,
    )
    if question is None:
        raise GatewayError(
            400,
            "Automatic routing requires a user message",
            "invalid_request_error",
            "missing_user_message",
        )
    try:
        router = load_router(get_settings().router_artifact)
    except Exception as exc:
        raise GatewayError(
            503,
            "Automatic routing unavailable: router artifact is missing or invalid",
            "router_unavailable",
            "router_unavailable",
        ) from exc
    eligible = []
    for model in router.models:
        try:
            apply_token_limits(request.model_copy(update={"model": model}))
        except GatewayError as exc:
            if exc.code != "context_length_exceeded":
                raise
        else:
            eligible.append(model)
    if not eligible:
        raise GatewayError(
            400,
            "Input exceeds every automatic routing model's limit",
            "invalid_request_error",
            "context_length_exceeded",
        )
    try:
        decision = router.route(question, quality, allowed_models=eligible)
        if decision.model not in eligible:
            raise ValueError("Router selected an ineligible model")
    except Exception as exc:
        raise GatewayError(
            503,
            "Automatic routing unavailable: router artifact is missing or invalid",
            "router_unavailable",
            "router_unavailable",
        ) from exc
    # No cheaper model is tried after a routing decision. All more expensive,
    # eligible targets are available, independently of explicit-model fallbacks.
    targets = eligible[eligible.index(decision.model) :]
    return targets, dict(
        artifact_run=router.run_name,
        quality=quality,
        tau=decision.tau,
        probabilities=decision.probabilities,
        routed_model=decision.model,
    )
