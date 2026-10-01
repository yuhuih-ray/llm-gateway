import os

import pytest

from llm_gateway.gemini import GeminiProvider
from llm_gateway.registry import MODELS
from llm_gateway.schemas import ChatCompletionRequest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.live
@pytest.mark.anyio
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set"
)
@pytest.mark.parametrize("model", ["gemini-flash", "gemini-pro"])
async def test_live_gemini(model):  # Test both pinned registry configurations.
    entry = MODELS[model]
    assert entry.upstream_model is not None
    response = await GeminiProvider(
        entry.upstream_model, thinking_level=entry.thinking_level
    ).complete(
        ChatCompletionRequest(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 + 2? Reply with only the number.",
                }
            ],
            max_tokens=512,
        )
    )
    assert response.model == model
    assert response.choices[0].message.content
    assert response.usage.total_tokens > 0
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.completion_tokens_details.reasoning_tokens >= 0
