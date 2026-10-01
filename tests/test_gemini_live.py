import os

import pytest

from llm_gateway.gemini import GeminiProvider, close_gemini_client
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


@pytest.fixture(autouse=True)
async def close_live_client(anyio_backend):  # Close before the test event loop ends.
    yield
    await close_gemini_client()


@pytest.mark.live
@pytest.mark.anyio
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set"
)
async def test_live_gemini_stream():
    import json
    from uuid import uuid4

    import httpx2

    from llm_gateway.auth import AuthContext, authenticate
    from llm_gateway.main import app

    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-flash",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a short paragraph of about 100 words explaining HTTP streaming.",
                        }
                    ],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": 512,
                },
            )
        assert response.status_code == 200
        frames = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert frames[-1] == "[DONE]"
        chunks = [json.loads(frame) for frame in frames[:-1]]
        content = [
            chunk
            for chunk in chunks
            if chunk["choices"] and chunk["choices"][0]["delta"].get("content")
        ]
        assert len(content) >= 2
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["total_tokens"] > 0
    finally:
        del app.dependency_overrides[authenticate]
