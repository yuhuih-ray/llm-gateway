import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.main import app, get_provider
from llm_gateway.providers import FakeProvider, FinalEvent, TextEvent
from llm_gateway.schemas import ChatCompletionRequest, Usage
from llm_gateway.streaming import stream_response


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def authenticated():
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    yield
    app.dependency_overrides.pop(authenticate, None)
    app.dependency_overrides.pop(get_provider, None)


def payload(**extra):  # Request overrides for individual scenarios.
    return {
        "model": "fake",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        **extra,
    }


def events(response):  # Parse each SSE data frame without hiding framing errors.
    frames = response.text.split("\n\n")
    assert frames.pop() == ""
    assert all(frame.startswith("data: ") for frame in frames)
    return [
        frame[6:] if frame[6:] == "[DONE]" else json.loads(frame[6:])
        for frame in frames
    ]


@pytest.mark.parametrize("include_usage", [False, True])
def test_sse_order(authenticated, include_usage):  # Exercise the real fake stream.
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=payload(stream_options={"include_usage": include_usage}),
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    chunks = events(response)
    assert chunks[-1] == "[DONE]"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks[1:4]] == [
        "Hello ",
        "from ",
        "FakeProvider.",
    ]
    assert chunks[4]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert len(chunks) == (7 if include_usage else 6)
    if include_usage:
        assert chunks[5]["choices"] == []
        assert chunks[5]["usage"] == Usage().model_dump()
    else:
        assert all("usage" not in chunk for chunk in chunks[:-1])
    assert len({chunk["id"] for chunk in chunks[:-1]}) == 1
    assert len({chunk["created"] for chunk in chunks[:-1]}) == 1
    assert all(
        chunk["object"] == "chat.completion.chunk" and chunk["model"] == "fake"
        for chunk in chunks[:-1]
    )


@pytest.mark.parametrize("fail_after,attempts,status", [(0, 3, 502), (1, 1, 200)])
def test_commit_point(
    authenticated, monkeypatch, fail_after, attempts, status
):  # Failure before or after first event.
    provider = FakeProvider(fail_after=fail_after)
    provider.stream = Mock(wraps=provider.stream)
    app.dependency_overrides[get_provider] = lambda: provider
    monkeypatch.setattr("llm_gateway.retries.random.uniform", lambda *_: 0)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload())
    assert response.status_code == status
    assert provider.stream.call_count == attempts
    if fail_after == 0:
        assert response.headers["content-type"] == "application/json"
        assert response.json()["error"]["type"] == "upstream_error"
    else:
        chunks = events(response)
        assert chunks[-1] == {
            "error": {"message": "Upstream stream failed", "type": "upstream_error"}
        }
        assert "[DONE]" not in chunks
        assert sum("error" in chunk for chunk in chunks) == 1


@pytest.mark.parametrize("first", [False, True])
def test_stream_timeout(
    authenticated, monkeypatch, first
):  # Before commitment or while idle.
    from dataclasses import replace

    from llm_gateway.registry import MODELS

    monkeypatch.setitem(
        MODELS,
        "fake",
        replace(MODELS["fake"], first_event_timeout=0.01, idle_timeout=0.01),
    )
    monkeypatch.setattr("llm_gateway.retries.random.uniform", lambda *_: 0)
    closed = []

    async def stream(request):  # Artificial stall without blocking the loop.
        try:
            if not first:
                yield TextEvent("first")
            await asyncio.sleep(1)
            yield FinalEvent("stop")
        finally:
            closed.append(True)

    provider = SimpleNamespace(stream=Mock(side_effect=stream))
    app.dependency_overrides[get_provider] = lambda: provider
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload())
    assert response.status_code == (504 if first else 200)
    assert len(closed) == 1
    assert provider.stream.call_count == 1
    if not first:
        assert events(response)[-1]["error"]["type"] == "upstream_error"
        assert "[DONE]" not in response.text


@pytest.mark.anyio
async def test_cancel_closes_upstream(caplog):
    waiting = asyncio.Event()
    closed = AsyncMock()

    async def stream(request):  # Stays open until the consumer disconnects.
        try:
            yield TextEvent("first")
            waiting.set()
            await asyncio.Event().wait()
        finally:
            await closed()

    response = await stream_response(
        SimpleNamespace(stream=stream), ChatCompletionRequest(**payload())
    )

    async def consume():
        async for _ in response.body_iterator:
            pass

    with caplog.at_level("INFO"):
        task = asyncio.create_task(consume())
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    closed.assert_awaited_once()
    assert "Client disconnected chunks_sent=2" in caplog.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("supplied,expected", [(None, 2048), (9000, 8192), (64, 64)])
def test_gateway_limits_both_paths(
    authenticated, stream, supplied, expected
):  # Providers see only the final budget.
    class RecordingProvider(FakeProvider):
        async def complete(self, request):  # Capture final non-stream request.
            assert request.max_tokens == expected
            return await super().complete(request)

        async def stream(self, request):  # Capture final streaming request.
            assert request.max_tokens == expected
            async for event in super().stream(request):
                yield event

    app.dependency_overrides[get_provider] = lambda: RecordingProvider()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=payload(model="gemini-flash", stream=stream, max_tokens=supplied),
        )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_disconnect_during_headers_closes_upstream(caplog):
    closed = AsyncMock()

    async def stream(request):  # Iterator is opened before response headers.
        try:
            yield TextEvent("first")
        finally:
            await closed()

    response = await stream_response(
        SimpleNamespace(stream=stream), ChatCompletionRequest(**payload())
    )

    async def send(message):  # Simulate cancellation during response headers.
        raise asyncio.CancelledError()

    with caplog.at_level("INFO"):
        with pytest.raises(asyncio.CancelledError):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send
            )
    closed.assert_awaited_once()
    assert "Client disconnected chunks_sent=0" in caplog.text
