import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.circuit import CircuitBreaker
from llm_gateway.config import get_settings
from llm_gateway.errors import GatewayError
from llm_gateway.main import app, get_provider
from llm_gateway.providers import FakeProvider
from llm_gateway.ratelimit import Decision
from llm_gateway.registry import MODELS
from llm_gateway.vllm import VllmProvider


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def routing_auth(monkeypatch):
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    monkeypatch.setattr("llm_gateway.retries.random.uniform", lambda *args: 0)
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def body(stream=False):
    return {
        "model": "local-qwen",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }


def answer(reason="stop"):
    return {
        "choices": [{"message": {"content": "local"}, "finish_reason": reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
async def test_connection_fallback_attribution(
    monkeypatch, isolated_usage_queue, stream, opt_in
):
    calls = 0

    def refused(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("private connection details")

    fallback = FakeProvider()
    fallback.complete = AsyncMock(wraps=fallback.complete)
    factory = Mock(return_value=fallback)
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(refused), base_url="http://upstream/"
    ) as upstream:
        app.dependency_overrides[get_provider] = lambda: VllmProvider("qwen", upstream)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            response = await client.post(
                "/v1/chat/completions",
                json=body(stream),
                headers={"X-Gateway-Fallback": "on"} if opt_in else {},
            )
            assert (
                app.state.provider_semaphores["vllm"]._value
                == get_settings().vllm_max_concurrency
            )
    assert calls == 3
    record = isolated_usage_queue.recorded[-1]
    assert record["model_requested"] == "local-qwen"
    if not opt_in:
        assert response.status_code == 502
        factory.assert_not_called()
        assert record["model_selected"] == "local-qwen"
        return
    assert response.status_code == 200
    factory.assert_called_once_with("gemini-flash-lite")
    assert response.headers["X-Gateway-Model"] == "gemini-flash-lite"
    assert response.headers["X-Gateway-Fallback-From"] == "local-qwen"
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert all(chunk["model"] == "gemini-flash-lite" for chunk in chunks)
        assert "[DONE]" in response.text
    else:
        assert response.json()["model"] == "gemini-flash-lite"
        assert fallback.complete.call_args.args[0].max_tokens == 2048
    assert record["model_selected"] == "gemini-flash-lite"
    assert record["upstream_model"] == MODELS["gemini-flash-lite"].upstream_model
    assert record["router_info"] == {
        "fallback_from": "local-qwen",
        "fallback_reason": "connection_error",
    }
    assert record["status"] == "success"
    assert record["error_type"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status,should_fallback,breaker_failure",
    [
        (429, True, False),
        (500, True, True),
        (503, True, True),
        (400, False, False),
        (401, False, False),
        (403, False, False),
        (422, False, False),
    ],
)
async def test_status_classification(
    monkeypatch, status, should_fallback, breaker_failure
):
    factory = Mock(return_value=FakeProvider())
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(status)),
        base_url="http://upstream/",
    ) as upstream:
        app.dependency_overrides[get_provider] = lambda: VllmProvider("qwen", upstream)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            response = await client.post(
                "/v1/chat/completions",
                json=body(),
                headers={"X-Gateway-Fallback": "on"},
            )
            assert app.state.circuits["vllm"].failures == int(breaker_failure)
    assert factory.called == should_fallback
    assert (response.status_code == 200) == should_fallback


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_timeout_allows_fallback(monkeypatch, stream):
    def timeout(request):
        raise httpx.ReadTimeout("private")

    factory = Mock(return_value=FakeProvider())
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(timeout), base_url="http://upstream/"
    ) as upstream:
        app.dependency_overrides[get_provider] = lambda: VllmProvider("qwen", upstream)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            response = await client.post(
                "/v1/chat/completions",
                json=body(stream),
                headers={"X-Gateway-Fallback": "on"},
            )
            assert app.state.circuits["vllm"].failures == 1
    assert response.status_code == 200
    factory.assert_called_once()


@pytest.mark.anyio
async def test_midstream_failure_never_falls_back(monkeypatch):
    app.dependency_overrides[get_provider] = lambda: FakeProvider(fail_after=1)
    factory = Mock()
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body(True),
            headers={"X-Gateway-Fallback": "on"},
        )
        assert app.state.circuits["vllm"].failures == 0
    assert response.status_code == 200
    assert '"error"' in response.text
    assert "[DONE]" not in response.text
    assert response.headers["X-Gateway-Model"] == "local-qwen"
    assert "X-Gateway-Fallback-From" not in response.headers
    factory.assert_not_called()


@pytest.mark.anyio
async def test_content_filter_never_falls_back(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json=answer("content_filter"))
        ),
        base_url="http://upstream/",
    ) as upstream:
        app.dependency_overrides[get_provider] = lambda: VllmProvider("qwen", upstream)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            response = await client.post(
                "/v1/chat/completions",
                json=body(),
                headers={"X-Gateway-Fallback": "on"},
            )
            assert app.state.circuits["vllm"].failures == 0
    assert response.json()["choices"][0]["finish_reason"] == "content_filter"
    factory.assert_not_called()


@pytest.mark.anyio
async def test_fallback_rechecks_budget_and_provider_bucket(monkeypatch):
    app.dependency_overrides[authenticate] = lambda: AuthContext(
        uuid4(), uuid4(), Decimal("1")
    )
    # Spending can exhaust the budget while the local attempt is in flight.
    budget = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr("llm_gateway.main.budget_exceeded", budget)
    limiter = AsyncMock(return_value=Decision(True, 100, 0))
    monkeypatch.setattr("llm_gateway.main.rate_limiter.check", limiter)
    factory = Mock()
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    app.dependency_overrides[get_provider] = lambda: FakeProvider(fail_after=0)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body(True),
            headers={"X-Gateway-Fallback": "on"},
        )
        assert (
            app.state.provider_semaphores["vllm"]._value
            == get_settings().vllm_max_concurrency
        )
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "insufficient_quota"
    assert budget.await_count == 2
    assert [b.key for b in limiter.call_args_list[1].args[1]] == [
        "ratelimit:provider:gemini"
    ]
    factory.assert_not_called()


def test_breaker_transitions_single_probe_and_stale_results(monkeypatch, caplog):
    clock = [100.0]
    monkeypatch.setattr(
        "llm_gateway.circuit.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    breaker = CircuitBreaker("vllm")
    stale = breaker.acquire()
    for _ in range(4):
        breaker.finish(breaker.acquire(), False)
    breaker.finish(breaker.acquire(), True)
    assert breaker.failures == 0
    for _ in range(5):
        breaker.finish(breaker.acquire(), False)
    assert breaker.state == "open"
    breaker.finish(stale, True)
    assert breaker.state == "open"
    with pytest.raises(GatewayError) as error:
        breaker.acquire()
    assert error.value.headers == {"Retry-After": "30"}
    clock[0] += 30
    probe = breaker.acquire()
    with pytest.raises(GatewayError):
        breaker.acquire()
    breaker.finish(probe, False)
    assert breaker.state == "open"
    clock[0] += 30
    probe = breaker.acquire()
    breaker.finish(probe, True)
    assert breaker.state == "closed"
    assert breaker.failures == 0
    assert "from=half-open to=closed" in caplog.text


@pytest.mark.anyio
async def test_open_circuit_skips_upstream_and_can_fallback(monkeypatch):
    calls = 0

    def failing(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    factory = Mock(return_value=FakeProvider())
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(failing), base_url="http://upstream/"
    ) as upstream:
        app.dependency_overrides[get_provider] = lambda: VllmProvider("qwen", upstream)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            for _ in range(5):
                assert (
                    await client.post("/v1/chat/completions", json=body())
                ).status_code == 502
            assert calls == 15
            response = await client.post("/v1/chat/completions", json=body())
            assert response.status_code == 503
            assert response.json()["error"]["type"] == "upstream_unavailable"
            assert 1 <= int(response.headers["Retry-After"]) <= 30
            response = await client.post(
                "/v1/chat/completions",
                json=body(),
                headers={"X-Gateway-Fallback": "on"},
            )
            assert response.status_code == 200
            assert calls == 15
            assert "gemini" in app.state.circuits


@pytest.mark.anyio
async def test_slot_overload_is_neutral_and_falls_back(
    monkeypatch, isolated_usage_queue
):
    overload = GatewayError(
        503, "busy", "upstream_overloaded", fallback_reason="upstream_overloaded"
    )
    acquire = AsyncMock(side_effect=[overload] * 6 + [None])
    monkeypatch.setattr("llm_gateway.main.acquire_provider", acquire)
    factory = Mock(return_value=FakeProvider())
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    app.dependency_overrides[get_provider] = lambda: FakeProvider()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        for _ in range(5):
            assert (
                await client.post("/v1/chat/completions", json=body())
            ).status_code == 503
        assert not isolated_usage_queue.recorded
        response = await client.post(
            "/v1/chat/completions", json=body(), headers={"X-Gateway-Fallback": "on"}
        )
        assert app.state.circuits["vllm"].failures == 0
        assert app.state.circuits["vllm"].state == "closed"
    assert response.status_code == 200
    assert acquire.call_args.args[1] == "gemini"


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_deadline_spans_fallback(monkeypatch, isolated_usage_queue, stream):
    monkeypatch.setattr(get_settings(), "request_deadline_seconds", 0.08)
    monkeypatch.setitem(
        MODELS, "local-qwen", replace(MODELS["local-qwen"], first_event_timeout=0.03)
    )

    async def hanging(request):
        await asyncio.Event().wait()

    slow = FakeProvider()
    slow.completion_delay = 1
    factory = Mock(return_value=slow)
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(hanging), base_url="http://upstream/"
    ) as upstream:
        app.dependency_overrides[get_provider] = lambda: VllmProvider("qwen", upstream)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            start = asyncio.get_running_loop().time()
            response = await client.post(
                "/v1/chat/completions",
                json=body(stream),
                headers={"X-Gateway-Fallback": "on"},
            )
            assert 0.06 <= asyncio.get_running_loop().time() - start < 0.2
            assert (
                app.state.provider_semaphores["vllm"]._value
                == get_settings().vllm_max_concurrency
            )
    assert response.status_code == 504
    assert response.json()["error"]["message"] == "Request deadline exceeded"
    factory.assert_called_once()
    assert isolated_usage_queue.recorded[-1]["error_type"] == "upstream_timeout"


@pytest.mark.anyio
@pytest.mark.parametrize("where", ["auth", "slot", "backoff"])
async def test_deadline_bounds_every_wait(monkeypatch, where):
    monkeypatch.setattr(get_settings(), "request_deadline_seconds", 0.03)
    monkeypatch.setattr("llm_gateway.retries.random.uniform", lambda *args: 1)

    async def hanging_auth():
        await asyncio.Event().wait()

    if where == "auth":
        app.dependency_overrides[authenticate] = hanging_auth
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(503)),
        base_url="http://upstream/",
    ) as upstream:
        app.dependency_overrides[get_provider] = lambda: VllmProvider("qwen", upstream)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client,
        ):
            if where == "slot":
                app.state.provider_semaphores["vllm"] = asyncio.Semaphore(0)
            start = asyncio.get_running_loop().time()
            response = await client.post(
                "/v1/chat/completions",
                json=body(),
                headers={"X-Gateway-Fallback": "on"},
            )
            assert asyncio.get_running_loop().time() - start < 0.15
    assert response.status_code == 504
    assert response.json()["error"]["message"] == "Request deadline exceeded"


@pytest.mark.anyio
async def test_deadline_stops_at_first_event(monkeypatch):
    monkeypatch.setattr(get_settings(), "request_deadline_seconds", 0.03)
    app.dependency_overrides[get_provider] = lambda: FakeProvider(delay=0.02)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        start = asyncio.get_running_loop().time()
        response = await client.post("/v1/chat/completions", json=body(True))
        assert asyncio.get_running_loop().time() - start > 0.05
    assert response.status_code == 200
    assert "[DONE]" in response.text
