import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.main import app
from llm_gateway.ratelimit import Bucket, RateLimiter, request_buckets


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("stream", [False, True])
def test_endpoint_fallback_burst(stream, monkeypatch, isolated_usage_queue, caplog):
    limiter = RateLimiter()
    clock = [100.0]
    monkeypatch.setattr("llm_gateway.main.rate_limiter", limiter)
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: clock[0])
    isolated_usage_queue.eval.side_effect = ConnectionError("Redis unreachable")
    tenant = [uuid4()]
    app.dependency_overrides[authenticate] = lambda: AuthContext(tenant[0], uuid4())
    payload = {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}
    try:
        with TestClient(app) as client:
            for _ in range(20):
                assert (
                    client.post("/v1/chat/completions", json=payload).status_code == 200
                )
            response = client.post(
                "/v1/chat/completions", json={**payload, "stream": stream}
            )
            assert response.status_code == 429
            assert response.headers["content-type"].startswith("application/json")
            assert response.headers["Retry-After"] == "1"
            assert response.headers["X-RateLimit-Remaining"] == "0"
            assert response.json() == {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            }
            assert isolated_usage_queue.enqueue_job.await_count == 20
            tenant[0] = uuid4()
            assert client.post("/v1/chat/completions", json=payload).status_code == 200

            def warnings():
                return [r for r in caplog.records if "in-process fallback" in r.message]

            assert len(warnings()) == 1
            clock[0] += 30
            client.post("/v1/chat/completions", json=payload)
            assert len(warnings()) == 2
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_fallback_all_or_nothing_and_refill(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: clock[0])
    limiter = RateLimiter()
    tenant, provider = Bucket("tenant", 20, 2), Bucket("provider", 1, 2)
    assert (await limiter.check(None, [provider])).allowed
    denied = await limiter.check(None, [tenant, provider])
    assert not denied.allowed
    assert denied.retry_after_ms == 500
    assert limiter.buckets["tenant"][0] == 20
    clock[0] += 0.5
    assert (await limiter.check(None, [tenant, provider])).allowed
    assert limiter.buckets["tenant"][0] == 19


@pytest.mark.anyio
async def test_redis_timeout_is_bounded():
    pool = AsyncMock()

    async def hang(*args):
        await asyncio.sleep(10)

    pool.eval.side_effect = hang
    limiter = RateLimiter()
    async with asyncio.timeout(0.5):
        decision = await limiter.check(pool, [Bucket("test", 1, 0.01)])
    assert decision.allowed
    assert not (await limiter.check(None, [Bucket("test", 1, 0.01)])).allowed


def test_provider_global_key_shared_across_tenants():
    a, b = uuid4(), uuid4()
    assert len(request_buckets(a, "fake")) == 2
    first, second = request_buckets(a, "gemini"), request_buckets(b, "gemini")
    assert first[0].key != second[0].key
    assert first[1] == second[1]
    assert first[1].capacity == 60
    assert first[1].refill == 5


def test_redis_rejection_checks_both_buckets_before_provider(isolated_usage_queue):
    from types import SimpleNamespace

    from llm_gateway.main import get_provider

    tenant = uuid4()
    provider = SimpleNamespace(complete=AsyncMock(), stream=AsyncMock())
    isolated_usage_queue.eval.return_value = [0, "0", 1501]
    app.dependency_overrides[authenticate] = lambda: AuthContext(tenant, uuid4())
    app.dependency_overrides[get_provider] = lambda: provider
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-flash",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "2"
        args = isolated_usage_queue.eval.call_args.args
        assert args[1:5] == (
            3,
            f"ratelimit:tenant:{tenant}",
            "ratelimit:provider:gemini",
            f"ratelimit:tokens:{tenant}",
        )
        isolated_usage_queue.eval.assert_awaited_once()
        provider.complete.assert_not_called()
        provider.stream.assert_not_called()
        isolated_usage_queue.enqueue_job.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()


def test_fallback_bucket_cap_and_lru(monkeypatch):
    from llm_gateway.ratelimit import MAX_LOCAL_BUCKETS

    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: 0.0)
    limiter = RateLimiter()
    for index in range(MAX_LOCAL_BUCKETS):
        limiter.fallback([Bucket(str(index), 20, 2)])
    # Accessing the oldest entry makes it newest; the next-oldest must be evicted.
    limiter.fallback([Bucket("0", 20, 2)])
    for index in range(MAX_LOCAL_BUCKETS, MAX_LOCAL_BUCKETS + 10):
        limiter.fallback([Bucket(str(index), 20, 2)])
        assert len(limiter.buckets) == MAX_LOCAL_BUCKETS
    assert "0" in limiter.buckets
    assert all(str(index) not in limiter.buckets for index in range(1, 11))


def test_fallback_prunes_idle_buckets_by_refill_time(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: clock[0])
    limiter = RateLimiter()
    slow, fast = Bucket("slow", 20, 2), Bucket("fast", 2, 2)
    limiter.fallback([slow, fast])
    clock[0] = 1.1
    limiter.fallback([Bucket("other", 20, 2)])
    assert "fast" not in limiter.buckets
    assert "slow" in limiter.buckets
    clock[0] = 10.1
    limiter.fallback([Bucket("other", 20, 2)])
    assert "slow" not in limiter.buckets
    assert limiter.fallback([slow]).remaining == 19
