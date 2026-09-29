from fastapi.testclient import TestClient

from llm_gateway.main import app, get_provider
from llm_gateway.providers import FakeProvider
from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
)


def test_chat_completion():
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "assistant", "content": "How can I help?"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": "chatcmpl-fake",
        "model": "fake-model",
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
                "model": "fake-model",
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


def test_fake_provider_is_deterministic():
    provider = FakeProvider()
    request = ChatCompletionRequest(model="another-model", messages=[])

    first = provider.complete(request)
    assert first == provider.complete(request)
    assert first.model == "another-model"


def test_endpoint_delegates_to_provider():
    received = []
    expected = ChatCompletionResponse(
        id="test-completion",
        model="test-model",
        choices=[Choice(message=AssistantMessage(content="Provider result"))],
    )

    class StubProvider:
        def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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
