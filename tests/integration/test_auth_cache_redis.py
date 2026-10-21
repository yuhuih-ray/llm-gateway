import argparse
import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx2
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.requests import Request
from testcontainers.redis import RedisContainer

from llm_gateway import auth, cli
from llm_gateway.admin_auth import UserContext, authenticate_user
from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.auth_cache import CachedAuth, cache_key, fill_auth
from llm_gateway.db import get_sessionmaker
from llm_gateway.main import app
from llm_gateway.models import ApiKey, Tenant

pytestmark = pytest.mark.anyio


@pytest.fixture
async def system(engine, monkeypatch):
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    keys = [generate_key(), generate_key()]
    async with sessions.begin() as session:
        tenant = Tenant(name="cached-auth", monthly_budget_usd=Decimal("10"))
        session.add(tenant)
        await session.flush()
        rows = [
            ApiKey(
                tenant_id=tenant.id,
                name=str(index),
                key_prefix=parse_key(key),
                key_hash=hash_key(key),
            )
            for index, key in enumerate(keys)
        ]
        session.add_all(rows)
    with RedisContainer("redis:8") as container:
        host, port = (
            container.get_container_host_ip(),
            int(container.get_exposed_port(6379)),
        )
        pool = await create_pool(RedisSettings(host=host, port=port))
        monkeypatch.setattr(
            "llm_gateway.main.create_usage_pool", AsyncMock(return_value=pool)
        )
        monkeypatch.setattr(
            "llm_gateway.auth_cache.create_cli_client",
            lambda: Redis(host=host, port=port),
        )
        monkeypatch.setattr(cli, "get_sessionmaker", lambda: sessions)
        monkeypatch.setattr(cli, "get_engine", lambda: engine)
        lookup = AsyncMock(wraps=auth.lookup_auth)
        monkeypatch.setattr(auth, "lookup_auth", lookup)
        app.dependency_overrides[get_sessionmaker] = lambda: sessions
        app.dependency_overrides[authenticate_user] = lambda: UserContext(
            uuid4(), tenant.id, "admin"
        )
        try:
            async with (
                app.router.lifespan_context(app),
                httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app), base_url="http://test"
                ) as client,
            ):
                yield SimpleNamespace(
                    client=client,
                    pool=pool,
                    tenant=tenant,
                    keys=keys,
                    rows=rows,
                    lookup=lookup,
                    sessions=sessions,
                )
        finally:
            app.dependency_overrides.clear()


async def post(system, key):
    return await system.client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
    )


@pytest.mark.parametrize("via", ["api", "cli"])
async def test_revoke_cached_key_and_late_fill(system, via):
    key = system.keys[0]
    name = cache_key(hash_key(key))
    assert (await post(system, key)).status_code == 200
    snapshot = CachedAuth.model_validate_json(await system.pool.get(name))
    assert (await post(system, key)).status_code == 200
    system.lookup.assert_awaited_once()
    assert 50 < await system.pool.ttl(name) <= 60
    if via == "api":
        response = await system.client.post(f"/admin/keys/{system.rows[0].id}/revoke")
        assert response.status_code == 200
    else:
        await cli.run(argparse.Namespace(command="revoke-key", prefix=parse_key(key)))
    assert json.loads(await system.pool.get(name)) == {"revoked": True}
    await fill_auth(system.pool, hash_key(key), snapshot)
    assert json.loads(await system.pool.get(name)) == {"revoked": True}
    assert (await post(system, key)).status_code == 401
    system.lookup.assert_awaited_once()
    assert 50 < await system.pool.ttl(name) <= 60
    async with system.sessions() as session:
        assert (
            await session.scalar(
                select(ApiKey.revoked_at).where(ApiKey.id == system.rows[0].id)
            )
            is not None
        )


async def test_suspend_reactivate_and_budget_invalidate_all_keys(system):
    for key in system.keys:
        assert (await post(system, key)).status_code == 200
    assert system.lookup.await_count == 2
    await cli.run(
        argparse.Namespace(command="suspend-tenant", tenant=system.tenant.name)
    )
    for key in system.keys:
        assert (await post(system, key)).status_code == 403
        assert json.loads(await system.pool.get(cache_key(hash_key(key)))) == {
            "revoked": True,
            "tenant_status": "suspended",
        }
    assert system.lookup.await_count == 2
    await cli.run(
        argparse.Namespace(command="reactivate-tenant", tenant=system.tenant.name)
    )
    for key in system.keys:
        assert await system.pool.get(cache_key(hash_key(key))) is None
        assert (await post(system, key)).status_code == 200
    assert system.lookup.await_count == 4
    await cli.run(
        argparse.Namespace(command="set-budget", tenant=system.tenant.name, usd="12.34")
    )
    for key in system.keys:
        assert await system.pool.get(cache_key(hash_key(key))) is None
        request = Request({"type": "http", "app": app, "client": ("127.0.0.1", 1234)})
        result = await auth.authenticate(request, system.sessions, f"Bearer {key}")
        assert result.monthly_budget_usd == Decimal("12.34")
    assert system.lookup.await_count == 6


async def test_invalidation_failure_does_not_rollback_revocation(
    system, monkeypatch, caplog
):
    key = system.keys[0]
    assert (await post(system, key)).status_code == 200
    with monkeypatch.context() as patch:
        patch.setattr(
            system.pool,
            "pipeline",
            lambda **kwargs: (_ for _ in ()).throw(ConnectionError("down")),
        )
        response = await system.client.post(f"/admin/keys/{system.rows[0].id}/revoke")
        assert response.status_code == 200
    assert "auth_cache_invalidation_failed" in caplog.text
    async with system.sessions() as session:
        assert (
            await session.scalar(
                select(ApiKey.revoked_at).where(ApiKey.id == system.rows[0].id)
            )
            is not None
        )
    name = cache_key(hash_key(key))
    assert 0 < await system.pool.ttl(name) <= 60
    # Model the cache TTL ending; the next DB lookup must observe revocation.
    await system.pool.expire(name, 0)
    assert (await post(system, key)).status_code == 401


async def test_failed_ip_limit_stops_database_work(system):
    key = generate_key()
    for _ in range(20):
        assert (await post(system, key)).status_code == 401
    assert system.lookup.await_count == 20
    response = await post(system, key)
    assert response.status_code == 429
    assert system.lookup.await_count == 20
    assert await system.pool.keys("auth:*") == []


async def test_concurrent_invalid_attempts_reserve_before_database(system, monkeypatch):
    original = auth.lookup_auth
    release = asyncio.Event()
    started = 0

    async def wait_then_lookup(*args):
        nonlocal started
        started += 1
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(auth, "lookup_auth", wait_then_lookup)
    tasks = [asyncio.create_task(post(system, generate_key())) for _ in range(25)]
    try:
        async with asyncio.timeout(3):
            while started < 20:
                await asyncio.sleep(0.005)
        assert started == 20
    finally:
        release.set()
        responses = await asyncio.gather(*tasks)
    assert (
        sorted(response.status_code for response in responses) == [401] * 20 + [429] * 5
    )
    assert system.lookup.await_count == 20
