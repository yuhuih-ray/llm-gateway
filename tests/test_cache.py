import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from llm_gateway import cache
from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.errors import GatewayError
from llm_gateway.main import app
from llm_gateway.providers import FakeProvider, FinalEvent, TextEvent
from llm_gateway.registry import MODELS, apply_token_limits, get_provider
from llm_gateway.schemas import ChatCompletionRequest, Usage
from llm_gateway.streaming import stream_response


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def setup_cache(monkeypatch, isolated_response_cache):
    values = {}

    async def get(key):
        return values.get(key)

    async def set_value(key, value, *, ex):
        assert ex == 86400
        values[key] = value

    isolated_response_cache.get.side_effect = get
    isolated_response_cache.set.side_effect = set_value
    auth = AuthContext(uuid4(), uuid4(), Decimal("1"))
    app.dependency_overrides[authenticate] = lambda: auth
    provider = FakeProvider()
    complete = provider.complete
    stream = provider.stream

    async def priced_complete(request):
        response = await complete(request)
        response.usage = Usage(prompt_tokens=12, completion_tokens=4, total_tokens=16)
        return response

    async def priced_stream(request):
        async for event in stream(request):
            if isinstance(event, FinalEvent):
                event = FinalEvent(
                    "stop",
                    Usage(prompt_tokens=12, completion_tokens=4, total_tokens=16),
                )
            yield event

    provider.complete = AsyncMock(side_effect=priced_complete)
    provider.stream = priced_stream
    app.dependency_overrides[get_provider] = lambda: provider
    charge = AsyncMock()
    monkeypatch.setattr("llm_gateway.main.rate_limiter.charge", charge)
    try:
        with TestClient(app) as client:
            yield client, provider, auth, charge, values
    finally:
        app.dependency_overrides.clear()


def payload(**changes):
    return {
        "model": "fake",
        "messages": [{"role": "user", "content": "private text"}],
        "temperature": 0,
        **changes,
    }


def post(client, **changes):
    return client.post(
        "/v1/chat/completions",
        json=payload(**changes),
        headers={"X-Gateway-Cache": "on"},
    )


def test_hit_accounting(setup_cache, isolated_usage_queue, monkeypatch):
    client, provider, auth, charge, values = setup_cache
    # Use nonzero prices so a zero-cost hit cannot accidentally pass as normal usage.
    monkeypatch.setitem(
        MODELS,
        "fake",
        replace(MODELS["fake"], upstream_model="original", input_price=Decimal("1")),
    )
    first = post(client)
    first_record = isolated_usage_queue.recorded[-1]
    second = post(client)
    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert first.json()["choices"] == second.json()["choices"]
    assert first.json()["id"] != second.json()["id"]
    provider.complete.assert_awaited_once()
    charge.assert_not_awaited()
    record = isolated_usage_queue.recorded[-1]
    assert record["cache_hit"] is True
    assert record["status"] == "success"
    assert Decimal(record["cost"]) == 0
    assert Decimal(first_record["cost"]) > 0
    assert record["upstream_model"] == "original"
    assert (record["prompt_tokens"], record["completion_tokens"]) == (12, 4)
    from llm_gateway.accounting import BATCH_LUA

    client.portal.call(app.state.accounting.flush)
    batches = [
        json.loads(c.args[-2])
        for c in isolated_usage_queue.eval.call_args_list
        if c.args[0] == BATCH_LUA
    ]
    assert (
        sum(charge["amount"] for batch in batches for charge in batch["tokens"]) == 16
    )
    assert sum(
        int(spend["amount"]) for batch in batches for spend in batch["spend"]
    ) == int(Decimal(first_record["cost"]) * 100_000_000)
    assert all(
        "private text" not in key and len(key) == len("cache:v1:") + 64
        for key in values
    )


@pytest.mark.parametrize(
    "change",
    ["tenant", "model", "system", "history", "max_tokens", "upstream", "thinking"],
)
def test_isolation(change, setup_cache, monkeypatch):
    client, provider, auth, _, _ = setup_cache
    assert post(client).headers["x-cache"] == "MISS"
    changes = {}
    if change == "tenant":
        app.dependency_overrides[authenticate] = lambda: AuthContext(
            uuid4(), auth.api_key_id
        )
    elif change == "model":
        changes["model"] = "gemini-flash-lite"
    elif change == "system":
        changes["messages"] = [
            {"role": "system", "content": "changed"},
            *payload()["messages"],
        ]
    elif change == "history":
        changes["messages"] = [
            *payload()["messages"],
            {"role": "assistant", "content": "history"},
            *payload()["messages"],
        ]
    elif change == "max_tokens":
        changes["max_tokens"] = 10
    else:
        monkeypatch.setitem(
            MODELS,
            "fake",
            replace(
                MODELS["fake"],
                **(
                    {"upstream_model": "new"}
                    if change == "upstream"
                    else {"thinking_budget": 32}
                ),
            ),
        )
    assert post(client, **changes).headers["x-cache"] == "MISS"
    assert provider.complete.await_count == 2


@pytest.mark.parametrize("temperature", [None, 0.5])
def test_nonzero_or_omitted(setup_cache, temperature):
    client, provider, _, _, values = setup_cache
    body = payload(temperature=temperature)
    if temperature is None:
        del body["temperature"]
    for _ in range(2):
        assert (
            client.post(
                "/v1/chat/completions", json=body, headers={"X-Gateway-Cache": "on"}
            ).headers["x-cache"]
            == "MISS"
        )
    assert not values
    assert provider.complete.await_count == 2


def test_bypass_existing_and_no_write(setup_cache):
    client, provider, _, _, values = setup_cache
    for _ in range(2):
        assert (
            client.post(
                "/v1/chat/completions",
                json=payload(),
                headers={
                    "X-Gateway-Cache": "on",
                    "Cache-Control": "max-age=0, No-Cache",
                },
            ).headers["x-cache"]
            == "MISS"
        )
    assert not values
    post(client)
    assert (
        client.post(
            "/v1/chat/completions",
            json=payload(),
            headers={"X-Gateway-Cache": "on", "Cache-Control": "no-cache"},
        ).headers["x-cache"]
        == "MISS"
    )
    assert provider.complete.await_count == 4


@pytest.mark.parametrize("first_stream", [True, False])
def test_stream_replay(setup_cache, first_stream, isolated_usage_queue):
    client, provider, _, charge, values = setup_cache
    assert post(client, stream=first_stream).headers["x-cache"] == "MISS"
    assert len(values) == 1
    response = post(client, stream=True, stream_options={"include_usage": True})
    assert response.headers["x-cache"] == "HIT"
    frames = [
        line[6:] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(line) for line in frames[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hello from FakeProvider."
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[3]["choices"] == []
    assert chunks[3]["usage"]["total_tokens"] == 16
    assert isolated_usage_queue.recorded[-1]["cache_hit"] is True
    charge.assert_not_awaited()
    assert post(client).headers["x-cache"] == "HIT"
    assert provider.complete.await_count == (0 if first_stream else 1)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["error", "content_filter"])
def test_unsuccessful_not_stored(setup_cache, stream, failure):
    client, provider, _, _, values = setup_cache
    if stream:

        async def failed(request):
            yield TextEvent("partial")
            if failure == "error":
                raise ConnectionError()
            yield FinalEvent("content_filter", Usage())

        provider.stream = failed
    elif failure == "error":
        provider.complete.side_effect = GatewayError(502, "failed", "upstream_error")
    else:

        async def filtered(request):
            response = await FakeProvider().complete(request)
            response.choices[0].finish_reason = "content_filter"
            return response

        provider.complete.side_effect = filtered
    for _ in range(2):
        assert post(client, stream=stream).headers["x-cache"] == "MISS"
    assert not values


def test_cache_down(setup_cache, isolated_response_cache):
    client, provider, _, _, _ = setup_cache
    isolated_response_cache.get.side_effect = ConnectionError()
    isolated_response_cache.set.side_effect = ConnectionError()
    for _ in range(2):
        response = post(client)
        assert response.status_code == 200
        assert response.headers["x-cache"] == "MISS"
    assert provider.complete.await_count == 2


def test_budget_exhausted_hit(setup_cache, isolated_usage_queue):
    client, provider, _, _, _ = setup_cache
    post(client)
    isolated_usage_queue.get.return_value = b"100000000"
    assert post(client).headers["x-cache"] == "HIT"
    assert post(client, max_tokens=1).status_code == 429
    provider.complete.assert_awaited_once()


@pytest.mark.anyio
async def test_timeouts_and_throttled_warnings(monkeypatch, caplog):
    client = AsyncMock()

    async def hang(*args, **kwargs):
        await asyncio.sleep(10)

    client.get.side_effect = hang
    client.set.side_effect = hang
    clock = [0]
    monkeypatch.setattr(cache, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cache, "last_warning", -float("inf"))
    answer = cache.CachedAnswer(
        response=await FakeProvider().complete(ChatCompletionRequest(**payload())),
        upstream_model=None,
    )
    async with asyncio.timeout(0.5):
        assert await cache.read_cache(client, "key") is None
        await cache.write_cache(client, "key", answer)
    assert len(caplog.records) == 1
    clock[0] = 30
    await cache.read_cache(client, "key")
    assert len(caplog.records) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_at", ["headers", "content", "done", "final"])
async def test_cancelled_delivery_not_stored(cancel_at):
    store = AsyncMock()
    response = await stream_response(
        FakeProvider(), ChatCompletionRequest(**payload()), on_complete=store
    )

    async def send(message):
        body = message.get("body", b"")
        if (
            cancel_at == "headers"
            and message["type"] == "http.response.start"
            or cancel_at == "content"
            and b"content" in body
            or cancel_at == "done"
            and b"[DONE]" in body
            or cancel_at == "final"
            and message["type"] == "http.response.body"
            and not message.get("more_body", False)
        ):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send
        )
    store.assert_not_awaited()


def test_key_uses_effective_limit():
    tenant = uuid4()
    a = apply_token_limits(
        ChatCompletionRequest(**payload(model="gemini-flash-lite", max_tokens=9000))
    )
    b = apply_token_limits(
        ChatCompletionRequest(**payload(model="gemini-flash-lite", max_tokens=8192))
    )
    assert cache.cache_key(tenant, a) == cache.cache_key(tenant, b)


def test_lifespan_owns_client(isolated_response_cache):
    with TestClient(app):
        assert app.state.cache_client is isolated_response_cache
        isolated_response_cache.aclose.assert_not_awaited()
    isolated_response_cache.aclose.assert_awaited_once()


@pytest.mark.parametrize("stream", [False, True])
def test_length_cached(setup_cache, stream):
    client, provider, _, _, values = setup_cache

    async def truncated(request):
        response = await FakeProvider().complete(request)
        response.choices[0].finish_reason = "length"
        return response

    async def truncated_stream(request):
        yield TextEvent("partial")
        yield FinalEvent("length", Usage())

    provider.complete.side_effect = truncated
    provider.stream = truncated_stream
    assert post(client, stream=stream).headers["x-cache"] == "MISS"
    assert values
    response = post(client)
    assert response.headers["x-cache"] == "HIT"
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_corrupt_entry_is_miss(setup_cache, isolated_response_cache):
    client, provider, _, _, _ = setup_cache
    isolated_response_cache.get.side_effect = None
    isolated_response_cache.get.return_value = b"not json"
    response = post(client)
    assert response.status_code == 200
    assert response.headers["x-cache"] == "MISS"
    provider.complete.assert_awaited_once()


@pytest.mark.parametrize("opt_in", [None, "off", "true", ""])
@pytest.mark.parametrize("stream", [False, True])
def test_temperature_zero_requires_opt_in(
    setup_cache, isolated_response_cache, opt_in, stream
):
    client, _, _, _, values = setup_cache
    headers = {} if opt_in is None else {"X-Gateway-Cache": opt_in}
    for _ in range(2):
        response = client.post(
            "/v1/chat/completions", json=payload(stream=stream), headers=headers
        )
        assert response.status_code == 200
        assert response.headers["x-cache"] == "MISS"
    assert not values
    isolated_response_cache.get.assert_not_awaited()
    isolated_response_cache.set.assert_not_awaited()

    # An existing answer still requires explicit consent on every request.
    assert post(client).headers["x-cache"] == "MISS"
    assert values
    isolated_response_cache.get.reset_mock()
    isolated_response_cache.set.reset_mock()
    response = client.post(
        "/v1/chat/completions", json=payload(stream=stream), headers=headers
    )
    assert response.status_code == 200
    assert response.headers["x-cache"] == "MISS"
    isolated_response_cache.get.assert_not_awaited()
    isolated_response_cache.set.assert_not_awaited()
