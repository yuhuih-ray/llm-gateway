from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx2
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker
from testcontainers.redis import RedisContainer

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.main import app
from llm_gateway.models import ApiKey, Tenant, UsageLog
from llm_gateway.usage import UsageRecord
from llm_gateway.worker import write_usage, write_usage_batch

pytestmark = pytest.mark.anyio


async def seed(engine):  # Foreign keys in the throwaway database.
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        tenant = Tenant(name="usage-test")
        session.add(tenant)
        await session.flush()
        key = ApiKey(
            tenant_id=tenant.id, name="key", key_prefix="gw_test", key_hash="test"
        )
        session.add(key)
        await session.commit()
        return AuthContext(tenant.id, key.id)


@pytest.mark.parametrize("cache_hit", [False, True])
async def test_worker_idempotent(
    engine,
    cache_hit,
):  # Same record twice, original timestamp retained.
    auth = await seed(engine)
    when = datetime(2025, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
    record = UsageRecord(
        request_id=uuid4(),
        tenant_id=auth.tenant_id,
        api_key_id=auth.api_key_id,
        model_requested="fake",
        model_selected="fake",
        upstream_model=None,
        created_at=when,
        status="success",
        cost=Decimal("0") if cache_hit else Decimal("0.00000001"),
        cache_hit=cache_hit,
    )
    ctx = {"sessions": async_sessionmaker(engine), "job_try": 1}
    await write_usage(ctx, record.model_dump(mode="json"))
    await write_usage(ctx, record.model_dump(mode="json"))
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        assert await session.scalar(select(func.count()).select_from(UsageLog)) == 1
        row = await session.scalar(select(UsageLog))
        assert row.created_at == when
        assert row.cost == record.cost
        assert row.cache_hit is cache_hit


async def test_end_to_end_usage(
    engine, monkeypatch
):  # Real Redis + Postgres, no Compose access.
    auth = await seed(engine)
    with RedisContainer("redis:8") as redis:
        settings = RedisSettings(
            host=redis.get_container_host_ip(), port=int(redis.get_exposed_port(6379))
        )
        pool = await create_pool(settings)
        monkeypatch.setattr(
            "llm_gateway.main.create_usage_pool", AsyncMock(return_value=pool)
        )
        app.dependency_overrides[authenticate] = lambda: auth
        try:
            async with app.router.lifespan_context(app):
                async with httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "fake",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                assert response.status_code == 200
                await app.state.accounting.flush()
                worker = Worker(
                    [write_usage, write_usage_batch],
                    redis_pool=pool,
                    burst=True,
                    handle_signals=False,
                    ctx={"sessions": async_sessionmaker(engine)},
                    poll_delay=0.01,
                )
                try:
                    await worker.async_run()
                    assert worker.jobs_complete == 1
                finally:
                    await worker.close()
                async with async_sessionmaker(
                    engine, expire_on_commit=False
                )() as session:
                    row = await session.scalar(
                        select(UsageLog).where(
                            UsageLog.request_id
                            == UUID(response.headers["X-Request-ID"])
                        )
                    )
                    assert row is not None
                    assert row.status == "success"
                    assert row.cost == Decimal("0")
        finally:
            app.dependency_overrides.clear()


async def test_batch_worker_idempotent(engine):
    from sqlalchemy import event

    auth = await seed(engine)
    records = [
        UsageRecord(
            request_id=uuid4(),
            tenant_id=auth.tenant_id,
            api_key_id=auth.api_key_id,
            model_requested="fake",
            model_selected="fake",
            upstream_model=None,
            created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            cost=Decimal("0.00000001"),
        ).model_dump(mode="json")
        for _ in range(3)
    ]
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        if statement.startswith("INSERT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        ctx = {"sessions": async_sessionmaker(engine)}
        await write_usage_batch(ctx, records[:2])
        assert len(statements) == 1
        assert "ON CONFLICT (request_id, created_at) DO NOTHING" in statements[0]
        await write_usage_batch(ctx, records)
        await write_usage(ctx, records[0])
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    async with async_sessionmaker(engine)() as session:
        assert await session.scalar(select(func.count()).select_from(UsageLog)) == 3
        assert await session.scalar(select(func.sum(UsageLog.cost))) == Decimal(
            "0.00000003"
        )
