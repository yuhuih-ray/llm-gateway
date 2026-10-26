import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest

from llm_gateway import auto_routing
from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.config import get_settings
from llm_gateway.errors import GatewayError
from llm_gateway.main import app
from llm_gateway.providers import FakeProvider
from llm_gateway.registry import MODELS
from llm_gateway.router import RoutingDecision

NAMES = ["local-qwen", "gemini-flash-lite", "gemini-flash"]
REAL_LOADER = auto_routing.load_router


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_routing(monkeypatch):
    REAL_LOADER.cache_clear()
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    factory = Mock(side_effect=lambda model: FakeProvider(delay=0))
    monkeypatch.setattr("llm_gateway.main.create_provider", factory)
    monkeypatch.setattr("llm_gateway.registry.create_provider", factory)
    try:
        yield factory
    finally:
        REAL_LOADER.cache_clear()
        app.dependency_overrides.clear()


@pytest.fixture
def stub_router(monkeypatch):
    router = Mock(models=tuple(NAMES), run_name="stub-run")

    def route(question, level, allowed_models):
        selected = dict(economy=NAMES[0], balanced=NAMES[1], high=NAMES[2])[level]
        if selected not in allowed_models:
            selected = allowed_models[0]
        return RoutingDecision(
            selected,
            {NAMES[0]: 0.8, NAMES[1]: 0.9},
            dict(economy=0.7, balanced=0.85, high=0.95)[level],
        )

    router.route.side_effect = route
    monkeypatch.setattr(auto_routing, "load_router", Mock(return_value=router))
    return router


def body(stream=False):
    return dict(
        model="auto",
        stream=stream,
        messages=[
            {"role": "system", "content": "System context"},
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old reply"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Last "},
                    {"type": "text", "text": "question?"},
                ],
            },
            {"role": "assistant", "content": "Assistant suffix"},
        ],
    )


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "quality,selected", list(zip(["economy", "balanced", "high"], NAMES))
)
async def test_quality_routes_and_records_metadata(
    stub_router, isolated_routing, isolated_usage_queue, quality, selected, stream
):
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body(stream),
            headers={"X-Gateway-Quality": quality},
        )
    assert response.status_code == 200
    stub_router.route.assert_called_once_with(
        "Last question?", quality, allowed_models=NAMES
    )
    isolated_routing.assert_called_once_with(selected)
    assert response.headers["X-Gateway-Model"] == selected
    assert "X-Gateway-Fallback-From" not in response.headers
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert all(chunk["model"] == selected for chunk in chunks)
    else:
        assert response.json()["model"] == selected
    record = isolated_usage_queue.recorded[-1]
    assert record["model_requested"] == "auto"
    assert record["model_selected"] == selected
    assert record["upstream_model"] == MODELS[selected].upstream_model
    assert record["router_info"] == dict(
        artifact_run="stub-run",
        quality=quality,
        tau=dict(economy=0.7, balanced=0.85, high=0.95)[quality],
        probabilities={NAMES[0]: 0.8, NAMES[1]: 0.9},
        routed_model=selected,
    )


@pytest.mark.anyio
async def test_default_quality_setting(monkeypatch, stub_router):
    monkeypatch.setattr(get_settings(), "router_default_quality", "high")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post("/v1/chat/completions", json=body())
    assert response.headers["X-Gateway-Model"] == NAMES[2]


@pytest.mark.anyio
async def test_full_context_filter_before_routing(stub_router, isolated_usage_queue):
    payload = body()
    payload["messages"][0]["content"] = "x" * 20_001
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-Gateway-Quality": "economy"},
        )
    assert response.status_code == 200
    stub_router.route.assert_called_once_with(
        "Last question?", "economy", allowed_models=NAMES[1:]
    )
    assert response.json()["model"] == NAMES[1]
    assert isolated_usage_queue.recorded[-1]["router_info"]["routed_model"] == NAMES[1]
    assert "fallback_reason" not in isolated_usage_queue.recorded[-1]["router_info"]


@pytest.mark.anyio
async def test_all_models_filtered(stub_router, isolated_usage_queue):
    payload = body()
    payload["messages"][0]["content"] = "x" * 100_001
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    stub_router.route.assert_not_called()
    assert not isolated_usage_queue.recorded


@pytest.mark.anyio
async def test_full_local_slot_escalates_without_acquire(
    stub_router, isolated_routing, isolated_usage_queue
):
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        full = asyncio.Semaphore(0)
        full.acquire = AsyncMock(
            side_effect=AssertionError("Must not wait for a full local slot")
        )
        app.state.provider_semaphores["vllm"] = full
        async with asyncio.timeout(0.5):
            response = await client.post(
                "/v1/chat/completions",
                json=body(),
                headers={"X-Gateway-Quality": "economy"},
            )
        full.acquire.assert_not_awaited()
    assert response.status_code == 200
    isolated_routing.assert_called_once_with(NAMES[1])
    assert response.headers["X-Gateway-Fallback-From"] == NAMES[0]
    assert len(isolated_usage_queue.recorded) == 1
    info = isolated_usage_queue.recorded[-1]["router_info"]
    assert info["fallback_from"] == info["routed_model"] == NAMES[0]
    assert info["fallback_reason"] == "upstream_overloaded"
    assert info["quality"] == "economy"


class Unreachable(FakeProvider):
    async def complete(self, request):
        raise GatewayError(
            502,
            "Connection refused",
            "upstream_error",
            fallback_reason="connection_error",
            breaker_failure=True,
        )

    async def stream(self, request):
        await self.complete(request)
        yield  # This must be an async generator, never a delivered event.


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_auto_uses_all_more_expensive_targets(
    monkeypatch, stub_router, isolated_routing, isolated_usage_queue, stream
):
    isolated_routing.side_effect = lambda model: (
        Unreachable() if model != NAMES[2] else FakeProvider(delay=0)
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body(stream),
            headers={"X-Gateway-Quality": "economy", "X-Gateway-Fallback": "off"},
        )
    assert response.status_code == 200
    assert [call.args[0] for call in isolated_routing.call_args_list] == NAMES
    assert response.headers["X-Gateway-Model"] == NAMES[2]
    record = isolated_usage_queue.recorded[-1]
    assert record["model_selected"] == NAMES[2]
    assert record["router_info"]["fallback_from"] == NAMES[0]
    assert record["router_info"]["fallback_reason"] == "connection_error"
    assert record["router_info"]["artifact_run"] == "stub-run"


@pytest.mark.anyio
async def test_midstream_failure_never_escalates(stub_router, isolated_routing):
    isolated_routing.side_effect = lambda model: FakeProvider(fail_after=1, delay=0)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body(True),
            headers={"X-Gateway-Quality": "economy"},
        )
    assert response.status_code == 200 and "error" in response.text
    isolated_routing.assert_called_once_with(NAMES[0])


@pytest.mark.anyio
@pytest.mark.parametrize("quality", ["invalid", "", "HIGH"])
async def test_invalid_quality(stub_router, quality, isolated_usage_queue):
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions", json=body(), headers={"X-Gateway-Quality": quality}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_quality"
    auto_routing.load_router.assert_not_called()
    assert not isolated_usage_queue.recorded


def artifact():
    value = json.loads(
        (Path(__file__).parent / "fixtures/router/synthetic.json").read_text()
    )
    old = value["models"]
    value["classifiers"] = {
        new: value["classifiers"][name] for new, name in zip(NAMES[:-1], old[:-1])
    }
    value["models"] = list(NAMES)
    return value


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid", ["missing", "json", "fields", "model", "quality", "settings"]
)
async def test_artifact_errors_only_affect_auto(monkeypatch, tmp_path, invalid):
    path = tmp_path / "router.json"
    value = artifact()
    if invalid == "model":
        value["models"][-1] = "unknown"
    elif invalid == "settings":
        value["classifiers"][NAMES[0]]["settings"]["lowercase"] = "yes"
    elif invalid == "quality":
        del value["levels"]["economy"]
    if invalid != "missing":
        path.write_text(
            "{"
            if invalid == "json"
            else "{}"
            if invalid == "fields"
            else json.dumps(value)
        )
    monkeypatch.setattr(get_settings(), "router_artifact", path)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post("/v1/chat/completions", json=body())
        explicit = await client.post(
            "/v1/chat/completions",
            json={**body(), "model": "fake"},
            headers={"X-Gateway-Quality": "invalid"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "router_unavailable"
    assert "missing or invalid" in response.json()["error"]["message"]
    assert explicit.status_code == 200 and explicit.json()["model"] == "fake"


@pytest.mark.anyio
async def test_artifact_loaded_lazily_once(monkeypatch, tmp_path):
    path = tmp_path / "router.json"
    path.write_text(json.dumps(artifact()))
    monkeypatch.setattr(get_settings(), "router_artifact", path)
    constructor = Mock(wraps=auto_routing.Router)
    monkeypatch.setattr(auto_routing, "Router", constructor)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        constructor.assert_not_called()
        first = await client.post("/v1/chat/completions", json=body())
        path.unlink()
        second = await client.post("/v1/chat/completions", json=body())
    assert first.status_code == second.status_code == 200
    constructor.assert_called_once_with(path)


@pytest.mark.anyio
async def test_no_user_message_is_rejected(stub_router):
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=dict(
                model="auto",
                messages=[dict(role="system", content="Only instructions")],
            ),
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_user_message"


@pytest.mark.anyio
async def test_auto_forwards_full_context_and_applies_selected_model_limits(
    stub_router, isolated_routing
):
    provider = FakeProvider()
    provider.complete = AsyncMock(wraps=provider.complete)
    isolated_routing.side_effect = lambda model: provider
    payload = {**body(), "max_tokens": 99999}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-Gateway-Quality": "economy"},
        )
    assert response.status_code == 200
    forwarded = provider.complete.call_args.args[0]
    assert forwarded.model == NAMES[0]
    assert forwarded.max_tokens == MODELS[NAMES[0]].max_tokens_cap
    assert [m.content for m in forwarded.messages] == [
        "System context",
        "Old question",
        "Old reply",
        "Last question?",
        "Assistant suffix",
    ]


@pytest.mark.anyio
async def test_auto_fallback_still_checks_budget(
    monkeypatch, stub_router, isolated_routing
):
    check = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr("llm_gateway.main.budget_exceeded", check)
    isolated_routing.side_effect = lambda model: Unreachable()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body(),
            headers={"X-Gateway-Quality": "economy"},
        )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "insufficient_quota"
    assert check.await_count == 2
    isolated_routing.assert_called_once_with(NAMES[0])
