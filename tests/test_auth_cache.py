import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.auth_cache import (
    CachedAuth,
    cache_key,
    fill_auth,
    invalidate_auth,
    read_auth,
)
from llm_gateway.db import get_sessionmaker
from llm_gateway.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def setup_auth(monkeypatch, isolated_usage_queue):
    from llm_gateway.ratelimit import Decision

    monkeypatch.setattr(
        "llm_gateway.main.rate_limiter.check",
        AsyncMock(return_value=Decision(True, 100, 0)),
    )
    key = generate_key()
    tenant = SimpleNamespace(
        id=uuid4(), status="active", monthly_budget_usd=Decimal("12.34")
    )
    api_key = SimpleNamespace(id=uuid4(), key_hash=hash_key(key), revoked_at=None)
    session = AsyncMock()
    session.execute.return_value = Mock(
        one_or_none=Mock(return_value=(api_key, tenant))
    )
    context = AsyncMock()
    context.__aenter__.return_value = session
    app.dependency_overrides[get_sessionmaker] = lambda: lambda: context
    pool = isolated_usage_queue
    values = {}
    pool.get.side_effect = lambda name: values.get(name)

    async def put(name, value, **kwargs):
        assert kwargs == {"ex": 60, "nx": True}
        if name in values:
            return False
        values[name] = value
        return True

    pool.set.side_effect = put
    monkeypatch.setattr("llm_gateway.accounting.SHUTDOWN_TIMEOUT", 0.01)
    try:
        with TestClient(app) as client:
            yield SimpleNamespace(
                client=client,
                key=key,
                tenant=tenant,
                session=session,
                values=values,
                pool=pool,
                api_key=api_key,
            )
    finally:
        app.dependency_overrides.clear()


def post(setup, key=None, headers=None):
    return setup.client.post(
        "/v1/chat/completions",
        headers=headers
        if headers is not None
        else {"Authorization": f"Bearer {key or setup.key}"},
        json={"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_second_request_skips_database(setup_auth):
    setup = setup_auth
    assert post(setup).status_code == 200
    assert post(setup).status_code == 200
    setup.session.execute.assert_awaited_once()
    stored = json.loads(setup.values[cache_key(hash_key(setup.key))])
    assert stored == dict(
        tenant_id=str(setup.tenant.id),
        api_key_id=str(setup.api_key.id),
        tenant_status="active",
        monthly_budget_usd="12.34",
    )
    assert list(setup.values) == ["auth:" + hash_key(setup.key)]
    assert setup.key not in str(setup.values)
    assert parse_key(setup.key) not in str(setup.values)
    setup.pool.set.assert_awaited_once()


def test_wrong_secret_with_correct_prefix_is_never_a_hit(setup_auth):
    setup = setup_auth
    assert post(setup).status_code == 200
    wrong = parse_key(setup.key) + "_" + generate_key().split("_", 2)[2]
    assert post(setup, wrong).status_code == 401
    assert setup.session.execute.await_count == 2
    assert cache_key(hash_key(wrong)) not in setup.values
    setup.pool.set.assert_awaited_once()


def test_unknown_key_is_not_cached(setup_auth):
    setup_auth.session.execute.return_value.one_or_none.return_value = None
    assert post(setup_auth).status_code == 401
    assert setup_auth.values == {}
    setup_auth.pool.set.assert_not_awaited()


@pytest.mark.parametrize("failure", ["read", "write", "malformed"])
def test_cache_failures_use_database(setup_auth, failure):
    setup = setup_auth
    if failure == "read":
        setup.pool.get.side_effect = ConnectionError("down")
    elif failure == "write":
        setup.pool.set.side_effect = ConnectionError("down")
    else:
        setup.values[cache_key(hash_key(setup.key))] = '{"tenant_id":"invalid"}'
    assert post(setup).status_code == 200
    setup.session.execute.assert_awaited_once()


@pytest.mark.parametrize(
    "headers",
    [None, {}, {"Authorization": "Basic wrong"}, {"Authorization": "Bearer malformed"}],
)
def test_failed_auth_limit_before_lookup(setup_auth, monkeypatch, headers):
    setup = setup_auth
    setup.pool.eval.side_effect = ConnectionError("down")
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: 100)
    setup.session.execute.return_value.one_or_none.return_value = None
    for _ in range(20):
        assert post(setup, headers=headers).status_code == 401
    lookups = setup.session.execute.await_count
    response = post(setup, headers=headers)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3"
    assert setup.session.execute.await_count == lookups
    assert lookups == (20 if headers is None else 0)
    assert not setup.values


@pytest.mark.parametrize("status", ["active", "suspended"])
def test_non_401_does_not_spend_failed_auth_tokens(setup_auth, monkeypatch, status):
    setup = setup_auth
    setup.tenant.status = status
    setup.pool.get.side_effect = ConnectionError("down")
    setup.pool.eval.side_effect = ConnectionError("down")
    monkeypatch.setattr("llm_gateway.ratelimit.monotonic", lambda: 100)
    expected = 200 if status == "active" else 403
    for _ in range(21):
        assert post(setup).status_code == expected
    assert setup.session.execute.await_count == 21


@pytest.mark.anyio
async def test_delayed_fill_cannot_overwrite_tombstone():
    values = {}
    pool = AsyncMock()
    pending, release = asyncio.Event(), asyncio.Event()

    async def set_value(key, value, **kwargs):
        pending.set()
        await release.wait()
        if not kwargs.get("nx") or key not in values:
            values[key] = value

    pool.set.side_effect = set_value
    digest = hash_key(generate_key())
    value = CachedAuth(
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        tenant_status="active",
        monthly_budget_usd=None,
    )
    task = asyncio.create_task(fill_auth(pool, digest, value))
    await pending.wait()
    values[cache_key(digest)] = '{"revoked":true}'
    release.set()
    await task
    pool.get.side_effect = lambda key: values.get(key)
    assert await read_auth(pool, digest) == "revoked"


@pytest.mark.anyio
async def test_invalidation_failure_logs_and_returns(caplog):
    pool = Mock()
    pool.pipeline.side_effect = ConnectionError("down")
    await invalidate_auth(pool, [hash_key(generate_key())], "revoke")
    assert "auth_cache_invalidation_failed" in caplog.text
    assert caplog.records[-1].levelname == "ERROR"


@pytest.mark.anyio
async def test_cache_timeout_falls_back():
    pool = AsyncMock()

    async def hang(*args, **kwargs):
        await asyncio.Event().wait()

    pool.get.side_effect = hang
    async with asyncio.timeout(0.5):
        assert await read_auth(pool, hash_key(generate_key())) is None
