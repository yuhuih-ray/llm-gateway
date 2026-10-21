import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.config import Settings, get_settings
from llm_gateway.errors import GatewayError
from llm_gateway.main import app, get_provider
from llm_gateway.providers import FinalEvent, TextEvent
from llm_gateway.registry import MODELS, apply_token_limits
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.streaming import stream_response
from llm_gateway.vllm import VllmProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


def request(**changes):
    return apply_token_limits(
        ChatCompletionRequest(
            model="local-qwen",
            messages=[
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "hi"},
            ],
            **changes,
        )
    )


def answer(reason="stop"):
    return {
        "id": "upstream-id",
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def event(value):
    return f"data: {json.dumps(value)}\n\n".encode()


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["stop", "length", "content_filter"])
async def test_complete_mapping(reason):
    def handle(upstream):
        assert upstream.url.path == "/v1/chat/completions"
        payload = json.loads(upstream.content)
        assert payload["model"] == "qwen2.5-7b-awq"
        assert payload["messages"] == request().model_dump()["messages"]
        assert payload["max_tokens"] == 1024
        assert payload["temperature"] == 0
        assert not payload["stream"]
        return httpx.Response(200, json=answer(reason))

    async with httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        response = await VllmProvider("qwen2.5-7b-awq", client).complete(
            request(temperature=0)
        )
    assert response.id == "upstream-id"
    assert response.model == "local-qwen"
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].finish_reason == reason
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 12


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["stop", "length", "content_filter"])
async def test_stream_waits_for_usage(reason):
    def handle(upstream):
        body = json.loads(upstream.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        content = b": keepalive\n\n" + event(
            {"choices": [{"delta": {"role": "assistant"}}]}
        )
        content += event({"choices": [{"delta": {"content": "hello"}}]})
        content += event({"choices": [{"delta": {}, "finish_reason": reason}]})
        content += (
            event({"choices": [], "usage": answer()["usage"]}) + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        events = [
            item
            async for item in VllmProvider("qwen2.5-7b-awq", client).stream(request())
        ]
    assert events[0] == TextEvent("hello")
    assert len(events) == 2
    assert isinstance(events[1], FinalEvent)
    assert events[1].finish_reason == reason
    assert events[1].usage.total_tokens == 12


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure,expected,calls",
    [
        ("connect", 502, 3),
        ("read", 504, 1),
        (500, 502, 3),
        (501, 502, 3),
        (400, 400, 1),
        (401, 502, 1),
    ],
)
async def test_error_mapping_and_retry(failure, expected, calls, monkeypatch):
    attempts = 0
    monkeypatch.setattr("llm_gateway.retries.random.uniform", lambda *args: 0)

    def handle(upstream):
        nonlocal attempts
        attempts += 1
        if failure == "connect":
            assert upstream.extensions["timeout"]["connect"] == 2
            raise httpx.ConnectError("Connection refused")
        if failure == "read":
            raise httpx.ReadTimeout("read timed out")
        return httpx.Response(failure, json={"error": "private upstream error"})

    async with httpx.AsyncClient(
        base_url="http://test/v1/",
        timeout=httpx.Timeout(2),
        transport=httpx.MockTransport(handle),
    ) as client:
        async with asyncio.timeout(2):
            with pytest.raises(GatewayError) as exc:
                await VllmProvider("qwen2.5-7b-awq", client).complete(request())
    assert exc.value.status == expected
    assert attempts == calls
    assert "private" not in exc.value.message


@pytest.mark.anyio
async def test_stream_retries_only_before_text(monkeypatch):
    monkeypatch.setattr("llm_gateway.retries.random.uniform", lambda *args: 0)
    calls = 0

    def handle(upstream):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        # EOF without DONE after visible content must not restart generation.
        return httpx.Response(
            200, content=event({"choices": [{"delta": {"content": "hello"}}]})
        )

    async with httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        response = await stream_response(
            VllmProvider("qwen2.5-7b-awq", client), request()
        )
        content = "".join([part async for part in response.body_iterator])
    assert "hello" in content
    assert '"error"' in content
    assert "[DONE]" not in content
    assert calls == 2


@pytest.mark.anyio
async def test_lazy_client_settings_and_cleanup(monkeypatch):
    from llm_gateway import vllm

    settings = Settings(_env_file=None, vllm_base_url=None, vllm_api_key=None)
    monkeypatch.setattr(vllm, "get_settings", lambda: settings)
    await vllm.close_vllm_client()
    with pytest.raises(GatewayError):
        vllm.get_vllm_client()
    settings.vllm_base_url = "http://test/v1"
    from pydantic import SecretStr

    settings.vllm_api_key = SecretStr("test-only-key")
    client = vllm.get_vllm_client()
    assert vllm.get_vllm_client() is client
    assert client.headers["authorization"] == "Bearer test-only-key"
    assert client.timeout.connect == 2
    assert client.timeout.read is None
    async with app.router.lifespan_context(app):
        pass
    assert client.is_closed
    assert vllm._shared_client is None


@pytest.mark.anyio
async def test_semaphore_wait_reject_and_release(monkeypatch, isolated_usage_queue):
    monkeypatch.setattr(get_settings(), "vllm_max_concurrency", 2)
    entered = 0
    release = asyncio.Event()

    async def handle(upstream):
        nonlocal entered
        entered += 1
        await release.wait()
        return httpx.Response(200, json=answer())

    auth = AuthContext(uuid4(), uuid4())
    app.dependency_overrides[authenticate] = lambda: auth
    try:
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as upstream:
            app.dependency_overrides[get_provider] = lambda: VllmProvider(
                "qwen2.5-7b-awq", upstream
            )
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://gateway"
                ) as client,
            ):
                body = request().model_dump()
                tasks = [
                    asyncio.create_task(client.post("/v1/chat/completions", json=body))
                    for _ in range(2)
                ]
                try:
                    async with asyncio.timeout(1):
                        while entered < 2:
                            await asyncio.sleep(0.001)
                    started = asyncio.get_running_loop().time()
                    response = await client.post("/v1/chat/completions", json=body)
                    assert 1.9 <= asyncio.get_running_loop().time() - started < 3
                    assert response.status_code == 503
                    assert response.headers["retry-after"] == "1"
                    assert response.json()["error"]["type"] == "upstream_overloaded"
                    assert not isolated_usage_queue.recorded
                    assert entered == 2
                finally:
                    release.set()
                    results = await asyncio.gather(*tasks)
                assert all(result.status_code == 200 for result in results)
                assert app.state.provider_semaphores["vllm"]._value == 2
                assert len(isolated_usage_queue.recorded) == 2
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
@pytest.mark.parametrize("disconnect_on", ["headers", "text"])
async def test_disconnected_stream_releases_slot(monkeypatch, disconnect_on):
    monkeypatch.setattr(get_settings(), "vllm_max_concurrency", 1)
    closed = asyncio.Event()
    sent = asyncio.Event()

    class Upstream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event({"choices": [{"delta": {"content": "hello"}}]})
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def send(message):
        if (
            disconnect_on == "headers" and message["type"] == "http.response.start"
        ) or (disconnect_on == "text" and b"hello" in message.get("body", b"")):
            sent.set()
            await asyncio.Event().wait()

    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        async with httpx.AsyncClient(
            base_url="http://test/v1/",
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, stream=Upstream())
            ),
        ) as upstream:
            app.dependency_overrides[get_provider] = lambda: VllmProvider(
                "qwen2.5-7b-awq", upstream
            )
            async with app.router.lifespan_context(app):
                scope = {
                    "type": "http",
                    "asgi": {"spec_version": "2.4"},
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                    "scheme": "http",
                    "server": ("test", 80),
                    "client": ("127.0.0.1", 1),
                }
                receive = AsyncMock(
                    return_value={
                        "type": "http.request",
                        "body": request(stream=True).model_dump_json().encode(),
                        "more_body": False,
                    }
                )
                task = asyncio.create_task(app(scope, receive, send))
                async with asyncio.timeout(1):
                    await sent.wait()
                assert app.state.provider_semaphores["vllm"].locked()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert closed.is_set()
                assert not app.state.provider_semaphores["vllm"].locked()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_registry_deadline_does_not_retry_reads(monkeypatch):
    calls = 0
    monkeypatch.setitem(
        MODELS, "local-qwen", replace(MODELS["local-qwen"], first_event_timeout=0.01)
    )

    async def handle(upstream):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    async with httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(GatewayError) as exc:
            await VllmProvider("qwen2.5-7b-awq", client).complete(request())
    assert exc.value.status == 504
    assert calls == 1


@pytest.mark.anyio
async def test_idle_timeout_closes_upstream_without_retry(monkeypatch):
    monkeypatch.setitem(
        MODELS, "local-qwen", replace(MODELS["local-qwen"], idle_timeout=0.01)
    )
    closed = asyncio.Event()
    calls = 0

    class Upstream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event({"choices": [{"delta": {"content": "hello"}}]})
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    def handle(upstream):
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=Upstream())

    async with httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        response = await stream_response(
            VllmProvider("qwen2.5-7b-awq", client), request()
        )
        content = "".join([chunk async for chunk in response.body_iterator])
    assert '"error"' in content
    assert "[DONE]" not in content
    assert calls == 1
    assert closed.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_provider_failure_releases_capacity(monkeypatch, stream):
    monkeypatch.setattr(get_settings(), "vllm_max_concurrency", 1)
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        async with httpx.AsyncClient(
            base_url="http://test/v1/",
            transport=httpx.MockTransport(lambda req: httpx.Response(401)),
        ) as upstream:
            app.dependency_overrides[get_provider] = lambda: VllmProvider(
                "qwen2.5-7b-awq", upstream
            )
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://gateway"
                ) as client,
            ):
                for _ in range(2):
                    response = await client.post(
                        "/v1/chat/completions", json=request(stream=stream).model_dump()
                    )
                    assert response.status_code == 502
                    assert app.state.provider_semaphores["vllm"]._value == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_registry_limits_and_gemini_has_no_semaphore():
    from starlette.datastructures import State

    from llm_gateway.concurrency import acquire_provider

    assert request(max_tokens=9999).max_tokens == 4096
    body = request().model_copy(update={"messages": request().messages * 4000})
    with pytest.raises(GatewayError) as exc:
        apply_token_limits(body)
    assert exc.value.code == "context_length_exceeded"
    state = State({"provider_semaphores": {}})
    assert await acquire_provider(state, "gemini") is None
    assert state.provider_semaphores == {}
