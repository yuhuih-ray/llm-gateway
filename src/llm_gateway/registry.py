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


# Changing thinking settings changes quality, latency, and cost.
# These IDs accepted real generateContent requests, not just model-list queries.
MODELS = {
    "fake": ModelEntry("fake", None, None),
    "gemini-flash": ModelEntry("gemini", "gemini-3.8-flash", types.ThinkingLevel.LOW),
    # Preview may retire on short notice; the live test detects availability changes.
    "gemini-pro": ModelEntry("gemini", "gemini-3.1-pro-preview", None),
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
    return GeminiProvider(entry.upstream_model, thinking_level=entry.thinking_level)
