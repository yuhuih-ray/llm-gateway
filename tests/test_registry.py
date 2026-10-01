from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.db import get_sessionmaker
from llm_gateway.main import app


def test_unknown_model_after_auth():
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "unknown",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
    finally:
        del app.dependency_overrides[authenticate]


def test_unauthenticated_unknown_model():
    app.dependency_overrides[get_sessionmaker] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "unknown",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        assert response.status_code == 401
    finally:
        del app.dependency_overrides[get_sessionmaker]


@pytest.mark.parametrize(
    "extra",
    [{"messages": []}, {"max_tokens": 0}, {"temperature": -1}, {"temperature": 2.1}],
)
def test_request_validation(extra):  # Invalid request field overrides.
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake",
                    "messages": [{"role": "user", "content": "Hi"}],
                    **extra,
                },
            )
        assert response.status_code == 422
    finally:
        del app.dependency_overrides[authenticate]


@pytest.mark.parametrize("model", ["gemini-flash", "gemini-pro"])
@pytest.mark.parametrize("supplied,expected", [(None, 1024), (10000, 8192), (512, 512)])
@pytest.mark.anyio
async def test_token_limits(
    model, supplied, expected
):  # Gateway model and requested budget.
    from llm_gateway.registry import get_provider
    from llm_gateway.schemas import ChatCompletionRequest

    request = ChatCompletionRequest(
        model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=supplied
    )
    from unittest.mock import AsyncMock, patch

    from llm_gateway.gemini import GeminiProvider

    provider = get_provider(request, AuthContext(uuid4(), uuid4()))
    assert isinstance(provider, GeminiProvider)
    provider.client = object()
    with patch.object(provider, "_complete", new_callable=AsyncMock) as complete:
        await provider.complete(request)
    assert complete.call_args.args[0].max_tokens == expected


def test_fake_has_no_token_limit():
    from llm_gateway.registry import get_provider
    from llm_gateway.schemas import ChatCompletionRequest

    for limit in (None, 100000):
        request = ChatCompletionRequest(
            model="fake", messages=[{"role": "user", "content": "Hi"}], max_tokens=limit
        )
        get_provider(request, AuthContext(uuid4(), uuid4()))
        assert request.max_tokens == limit


@pytest.fixture
def anyio_backend():
    return "asyncio"
