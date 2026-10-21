import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from llm_gateway import admin, admin_auth
from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.config import Settings
from llm_gateway.db import get_sessionmaker
from llm_gateway.main import app
from llm_gateway.ratelimit import (
    Bucket,
    RateLimiter,
    client_ip,
    login_buckets,
    token_bucket,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_token_debt_floor_retry_and_recovery(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: clock[0])
    limiter = RateLimiter()
    token = Bucket("tokens", 100, 10, check_only=True)
    request = Bucket("requests", 20, 2)
    await limiter.charge(None, token, 99)
    assert (await limiter.check(None, [request, token])).allowed
    assert limiter.buckets[token.key][0] == 1
    await limiter.charge(None, token, 1000)
    assert limiter.buckets[token.key][0] == -100
    denied = await limiter.check(None, [request, token])
    assert not denied.allowed
    assert denied.retry_after_ms == 10001
    assert limiter.buckets[request.key][0] == 19
    clock[0] = 10
    assert not (await limiter.check(None, [token])).allowed  # Zero is not positive.
    clock[0] = 10.001
    assert (await limiter.check(None, [token])).allowed


@pytest.mark.parametrize(
    "trusted,peer,forwarded,expected",
    [
        ([], "192.0.2.1", "198.51.100.5", "192.0.2.1"),
        (["10.0.0.0/8"], "10.0.0.2", "198.51.100.5, 10.0.0.1", "198.51.100.5"),
        (["::1"], "::1", "192.0.2.9, 198.51.100.5", "198.51.100.5"),
        (["10.0.0.0/8"], "10.0.0.2", "bad", "10.0.0.2"),
    ],
)
def test_client_ip(monkeypatch, trusted, peer, forwarded, expected):
    monkeypatch.setattr(
        "llm_gateway.ratelimit.get_settings",
        lambda: Settings(_env_file=None, trusted_proxies=trusted),
    )
    request = Request(
        {
            "type": "http",
            "client": (peer, 1),
            "headers": [(b"x-forwarded-for", forwarded.encode())],
        }
    )
    assert client_ip(request) == expected
    keys = [b.key for b in login_buckets(request, " Private@Example.com ")]
    assert all("private" not in key.lower() and "@" not in key for key in keys)
    assert keys == [b.key for b in login_buckets(request, "private@example.com")]


@pytest.fixture
def login_setup(monkeypatch, isolated_usage_queue):
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: 0.0)
    isolated_usage_queue.eval.side_effect = ConnectionError("down")
    limiter = RateLimiter()
    monkeypatch.setattr(admin, "rate_limiter", limiter)
    user = SimpleNamespace(password_hash="hash", disabled_at=None)
    session = AsyncMock()
    session.scalar.return_value = user
    sessions = Mock()
    sessions.return_value.__aenter__ = AsyncMock(return_value=session)
    sessions.return_value.__aexit__ = AsyncMock(return_value=False)
    verify = Mock(return_value=True)
    monkeypatch.setattr(admin, "verify_password", verify)
    monkeypatch.setattr(
        admin,
        "password_hasher",
        SimpleNamespace(check_needs_rehash=lambda value: False),
    )
    monkeypatch.setattr(admin, "issue_token", lambda *args: "jwt")
    app.dependency_overrides[admin_auth.jwt_secret] = lambda: "test-secret"
    app.dependency_overrides[get_sessionmaker] = lambda: sessions
    try:
        yield limiter, verify, session
    finally:
        app.dependency_overrides.clear()


def test_login_ip_limit_precedes_database_and_argon(login_setup):
    limiter, verify, session = login_setup
    with TestClient(app) as client:
        for _ in range(10):
            assert (
                client.post(
                    "/auth/login", json={"email": "a@example.com", "password": "pw"}
                ).status_code
                == 200
            )
        response = client.post(
            "/auth/login", json={"email": "a@example.com", "password": "pw"}
        )
    assert response.status_code == 429
    assert response.json() == {"detail": "Rate limit exceeded"}
    assert response.headers["retry-after"] == "6"
    assert verify.call_count == session.scalar.await_count == 10
    assert (
        next(value[0] for key, value in limiter.buckets.items() if ":email:" in key)
        == 5
    )


@pytest.mark.parametrize("unknown", [False, True])
def test_email_failed_attempts_across_ips(login_setup, monkeypatch, unknown):
    limiter, verify, session = login_setup
    verify.return_value = False
    if unknown:
        session.scalar.return_value = None
    peer = [0]
    monkeypatch.setattr(
        "llm_gateway.ratelimit.client_ip", lambda request: f"192.0.2.{peer[0]}"
    )
    with TestClient(app) as client:
        for index in range(5):
            peer[0] = index
            assert (
                client.post(
                    "/auth/login", json={"email": " A@example.com ", "password": "bad"}
                ).status_code
                == 401
            )
        peer[0] = 100
        response = client.post(
            "/auth/login", json={"email": "a@example.com", "password": "bad"}
        )
    assert response.status_code == 429
    assert verify.call_count == session.scalar.await_count == 5
    assert all("example.com" not in key for key in limiter.buckets)


@pytest.mark.anyio
@pytest.mark.parametrize("known", [True, False])
async def test_charge_after_final_response_and_unknown_skipped(
    monkeypatch, known, isolated_usage_queue
):
    from datetime import datetime, timezone

    from starlette.datastructures import State

    from llm_gateway.schemas import Usage
    from llm_gateway.usage import UsageMiddleware, UsageTracker

    sent_final = False
    charge_calls = []
    pool = AsyncMock()

    async def slow_charge(*args):
        assert sent_final
        charge_calls.append(args)
        await asyncio.sleep(10)

    pool.pipeline = isolated_usage_queue.pipeline
    pool.eval.side_effect = slow_charge
    from llm_gateway.accounting import AccountingBuffer

    state = State({"usage_pool": pool})
    state.accounting = AccountingBuffer(state)
    tracker = UsageTracker(
        {
            "started": 0.0,
            "request_id": uuid4(),
            "created_at": datetime.now(timezone.utc),
        },
        AuthContext(uuid4(), uuid4()),
        "fake",
        state,
    )
    if known:
        tracker.success(
            Usage(
                prompt_tokens=20,
                completion_tokens=30,
                completion_tokens_details={"reasoning_tokens": 10},
            )
        )

    async def inner(scope, receive, send):
        scope["state"]["usage_tracker"] = tracker
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        nonlocal sent_final
        if message["type"] == "http.response.body":
            sent_final = True

    async with asyncio.timeout(0.5):
        await UsageMiddleware(inner)(
            {"type": "http", "path": "/v1/chat/completions"}, AsyncMock(), send
        )
    assert charge_calls == []
    assert len(state.accounting.records) == 1
    pool.enqueue_job.assert_not_awaited()
    if known:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await state.accounting.flush()
        assert charge_calls[0][-1] == 50
    else:
        await state.accounting.flush()
        assert charge_calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_endpoint_token_debt_is_retained_during_outage(
    monkeypatch, isolated_usage_queue, stream
):
    from llm_gateway.main import get_provider
    from llm_gateway.providers import FakeProvider, FinalEvent, TextEvent
    from llm_gateway.schemas import Usage

    class CostlyProvider(FakeProvider):
        async def complete(self, request):
            response = await super().complete(request)
            response.usage = Usage(prompt_tokens=100_000, completion_tokens=100_000)
            return response

        async def stream(self, request):
            yield TextEvent("hello")
            yield FinalEvent(
                "stop", Usage(prompt_tokens=100_000, completion_tokens=100_000)
            )

    limiter = RateLimiter()
    clock = [0.0]
    monkeypatch.setattr("llm_gateway.main.rate_limiter", limiter)
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: clock[0])
    monkeypatch.setattr("llm_gateway.accounting.SHUTDOWN_TIMEOUT", 0.01)
    isolated_usage_queue.eval.side_effect = ConnectionError("down")
    tenant = uuid4()
    limiter.fallback([token_bucket(tenant)], charge=99_999)
    app.dependency_overrides[authenticate] = lambda: AuthContext(tenant, uuid4())
    app.dependency_overrides[get_provider] = CostlyProvider
    payload = {
        "model": "fake",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }
    try:
        with TestClient(app) as client:
            assert client.post("/v1/chat/completions", json=payload).status_code == 200
            assert limiter.buckets[token_bucket(tenant).key][0] == 1
            assert len(app.state.accounting.records) == 1
            record = app.state.accounting.records[0]
            assert record.prompt_tokens + record.completion_tokens == 200_000
            isolated_usage_queue.enqueue_job.assert_not_awaited()
        assert len(app.state.accounting.records) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_parallel_login_attempts_reserve_before_argon(login_setup, monkeypatch):
    import httpx2

    limiter, verify, session = login_setup
    verify.return_value = False
    release = asyncio.Event()
    started = []

    async def gated_verify(function, *args):
        started.append(1)
        await release.wait()
        return function(*args)

    monkeypatch.setattr(admin, "run_in_threadpool", gated_verify)

    # Direct peers differ; forwarded headers must not be needed to rotate IPs.
    async def attempt(index):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app, client=(f"192.0.2.{index}", 1)),
            base_url="http://test",
        ) as client:
            return await client.post(
                "/auth/login", json={"email": "victim@example.com", "password": "wrong"}
            )

    tasks = [asyncio.create_task(attempt(index)) for index in range(20)]
    try:
        # Keep the admitted password checks in flight until the other 15 reject.
        async with asyncio.timeout(2):
            while sum(task.done() for task in tasks) < 15:
                await asyncio.sleep(0.001)
        assert len(started) == 5
        release.set()
        responses = await asyncio.gather(*tasks)
        assert sum(response.status_code == 401 for response in responses) == 5
        assert sum(response.status_code == 429 for response in responses) == 15
        assert verify.call_count == session.scalar.await_count == 5
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_refund_fallback_caps_balance_and_times_out():
    limiter = RateLimiter()
    bucket = Bucket("email", 5, 1 / 60)
    assert (await limiter.check(None, [bucket])).allowed
    pool = AsyncMock()

    async def hang(*args):
        await asyncio.sleep(10)

    pool.eval.side_effect = hang
    async with asyncio.timeout(0.5):
        await limiter.refund(pool, bucket, 100)
    assert limiter.buckets[bucket.key][0] == 5
