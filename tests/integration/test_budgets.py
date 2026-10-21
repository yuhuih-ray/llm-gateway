import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.requests import Request
from testcontainers.redis import RedisContainer

from llm_gateway import budgets, cli
from llm_gateway.auth import authenticate, generate_key, hash_key, parse_key
from llm_gateway.models import ApiKey, Tenant, UsageLog

pytestmark = pytest.mark.anyio


@pytest.fixture
async def redis():
    with RedisContainer("redis:8") as container:
        pool = await create_pool(
            RedisSettings(
                host=container.get_container_host_ip(),
                port=int(container.get_exposed_port(6379)),
            )
        )
        try:
            yield pool
        finally:
            await pool.aclose()


async def test_exact_integer_spend_warning_and_new_month(redis, caplog, monkeypatch):
    tenant = uuid4()
    when = datetime(2026, 10, 15, tzinfo=timezone.utc)
    cost = Decimal("0.00000001")
    for _ in range(100):
        await budgets.add_spend(redis, tenant, cost, when, Decimal("0.000001"))
    value = await redis.get(budgets.spend_key(tenant, when))
    assert value == b"100"
    assert Decimal(int(value)) / budgets.SCALE == cost * 100
    assert 0 < await redis.ttl(budgets.spend_key(tenant, when)) <= budgets.TTL
    assert (
        sum("monthly_budget_80_percent" in row.message for row in caplog.records) == 1
    )
    monkeypatch.setattr(budgets, "utcnow", lambda: when)
    assert await budgets.budget_exceeded(redis, tenant, Decimal("0.000001"))
    monkeypatch.setattr(budgets, "utcnow", lambda: budgets.month_end(when))
    assert not await budgets.budget_exceeded(redis, tenant, Decimal("0.000001"))
    await budgets.add_spend(
        redis, tenant, Decimal("1"), budgets.month_end(when), Decimal("1")
    )
    assert (
        sum("monthly_budget_80_percent" in row.message for row in caplog.records) == 2
    )


async def test_reconciliation_and_cli(engine, redis, monkeypatch):
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    async with sessions.begin() as session:
        tenant = Tenant(name="budget", monthly_budget_usd=Decimal("10"))
        session.add(tenant)
        await session.flush()
        full_key = generate_key()
        key = ApiKey(
            tenant_id=tenant.id,
            name="key",
            key_prefix=parse_key(full_key),
            key_hash=hash_key(full_key),
        )
        session.add(key)
        await session.flush()
        for cost, when in [
            (Decimal("1.12345678"), now),
            (Decimal("9"), budgets.month_start(now) - timedelta(seconds=1)),
        ]:
            session.add(
                UsageLog(
                    request_id=uuid4(),
                    tenant_id=tenant.id,
                    api_key_id=key.id,
                    model_requested="fake",
                    model_selected="fake",
                    status="success",
                    cost=cost,
                    created_at=when,
                )
            )
    auth = await authenticate(Request({"type": "http"}), sessions, f"Bearer {full_key}")
    assert auth.monthly_budget_usd == Decimal("10")
    ctx = {"sessions": sessions, "redis": redis}
    counter = budgets.spend_key(tenant.id, now)
    for initial, expected in [
        (None, 112345678),
        (1, 112345678),
        (200000000, 200000000),
        (9007199254740993, 9007199254740993),
    ]:
        await redis.delete(counter)
        if initial is not None:
            await redis.set(counter, initial)
        await budgets.reconcile_budgets(ctx)
        assert int(await redis.get(counter)) == expected
    monkeypatch.setattr(cli, "get_sessionmaker", lambda: sessions)
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    for value, expected in [("12.34", Decimal("12.34")), ("none", None)]:
        await cli.run(
            argparse.Namespace(command="set-budget", tenant="budget", usd=value)
        )
        async with sessions() as session:
            assert (
                await session.scalar(
                    select(Tenant.monthly_budget_usd).where(Tenant.id == tenant.id)
                )
                == expected
            )
