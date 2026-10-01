import asyncio
from time import perf_counter
from uuid import uuid4

import httpx2
import pytest
from fastapi.testclient import TestClient

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.main import app, get_provider
from llm_gateway.providers import FakeProvider
from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
)


@pytest.fixture(autouse=True)
def authenticated():
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    yield
    del app.dependency_overrides[authenticate]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_chat_completion():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "assistant", "content": "How can I help?"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "id": "chatcmpl-fake",
        "model": "fake",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from FakeProvider.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    ChatCompletionResponse.model_validate(response.json())


def test_invalid_message_role():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake",
                "messages": [{"role": "tool", "content": "Hello"}],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "messages", 0, "role"]


def test_missing_model():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "model"]


@pytest.mark.anyio
async def test_fake_provider_is_deterministic():
    provider = FakeProvider()
    request = ChatCompletionRequest(
        model="fake", messages=[{"role": "user", "content": "Hello"}]
    )

    first = await provider.complete(request)
    assert first == await provider.complete(request)
    assert first.model == "fake"


def test_endpoint_delegates_to_provider():
    received = []
    expected = ChatCompletionResponse(
        id="test-completion",
        model="test-model",
        choices=[Choice(message=AssistantMessage(content="Provider result"))],
    )

    class StubProvider:
        async def complete(
            self,
            request: ChatCompletionRequest,  # Capture the endpoint input.
        ) -> ChatCompletionResponse:
            received.append(request)
            return expected

    app.dependency_overrides[get_provider] = lambda: StubProvider()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
    finally:
        del app.dependency_overrides[get_provider]

    assert response.status_code == 200
    assert response.json() == expected.model_dump()
    assert len(received) == 1
    assert received[0].model == "test-model"
    assert received[0].messages[0].content == "Hello"


@pytest.mark.anyio
async def test_endpoint_does_not_block_event_loop():
    class SlowProvider:
        async def complete(
            self,
            request: ChatCompletionRequest,  # Preserve the requested model in the reply.
        ) -> ChatCompletionResponse:
            await asyncio.sleep(0.5)
            return ChatCompletionResponse(
                id="slow-completion",
                model=request.model,
                choices=[Choice(message=AssistantMessage(content="Done"))],
            )

    app.dependency_overrides[get_provider] = lambda: SlowProvider()
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            started = perf_counter()
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "Hello"}],
                        },
                    )
                    for _ in range(10)
                )
            )
            elapsed = perf_counter() - started
    finally:
        del app.dependency_overrides[get_provider]

    assert all(response.status_code == 200 for response in responses)
    assert all(
        response.json()["choices"][0]["message"]["content"] == "Done"
        for response in responses
    )
    # Ten sequential 0.5-second waits take about 5 seconds; concurrent waits stay below 2.
    assert elapsed < 2, f"Concurrent requests took {elapsed:.3f} seconds"
