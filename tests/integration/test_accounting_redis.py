from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from starlette.datastructures import State
from testcontainers.redis import RedisContainer

from llm_gateway.accounting import SPEND_LUA, TOKEN_LUA, AccountingBuffer
from llm_gateway.budgets import spend_key
from llm_gateway.usage import UsageRecord

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("lost_reply", [TOKEN_LUA, SPEND_LUA, "enqueue"])
async def test_real_redis_retry_exact_charges(lost_reply, monkeypatch, caplog):
    # Tiny refill makes exact debit assertions insensitive to test execution time.
    monkeypatch.setattr(
        "llm_gateway.accounting.token_bucket",
        lambda tenant: SimpleNamespace(
            key=f"ratelimit:tokens:{tenant}",
            capacity=100_000,
            refill=0.000001,
        ),
    )
    with RedisContainer("redis:8") as redis:
        pool = await create_pool(
            RedisSettings(
                host=redis.get_container_host_ip(),
                port=int(redis.get_exposed_port(6379)),
            )
        )
        real_eval, real_enqueue = pool.eval, pool.enqueue_job
        failed = False

        async def eval_with_lost_reply(script, *args):
            nonlocal failed
            result = await real_eval(script, *args)
            if script == lost_reply and not failed:
                failed = True
                raise ConnectionError("reply lost after commit")
            return result

        async def enqueue_with_lost_reply(*args, **kwargs):
            nonlocal failed
            result = await real_enqueue(*args, **kwargs)
            if lost_reply == "enqueue" and not failed:
                failed = True
                raise ConnectionError("reply lost after enqueue")
            return result

        monkeypatch.setattr(pool, "eval", eval_with_lost_reply)
        monkeypatch.setattr(
            pool, "enqueue_job", AsyncMock(side_effect=enqueue_with_lost_reply)
        )
        pending = AccountingBuffer(State({"usage_pool": pool}))
        tenant = uuid4()
        when = datetime(2026, 10, 1, tzinfo=timezone.utc)
        record = UsageRecord(
            request_id=uuid4(),
            tenant_id=tenant,
            api_key_id=uuid4(),
            model_requested="fake",
            model_selected="fake",
            upstream_model=None,
            created_at=when,
            prompt_tokens=10,
            completion_tokens=20,
            cost=Decimal("0.00000001"),
            monthly_budget_usd=Decimal("0.00000003"),
        )
        for _ in range(3):
            pending.record(record.model_copy(update={"request_id": uuid4()}))
        try:
            with pytest.raises(ConnectionError):
                await pending.flush()
            assert len(pending.records) == 3
            await pending.flush()
            assert not pending.records
            assert await pool.get(spend_key(tenant, when)) == b"3"
            assert float(
                await pool.hget(f"ratelimit:tokens:{tenant}", "tokens")
            ) == pytest.approx(99910, abs=0.001)
            assert await pool.zcard("arq:queue") == 1
            assert await pool.keys("accounting:*") == []
            assert (
                sum("monthly_budget_80_percent" in r.message for r in caplog.records)
                == 1
            )
            pending.record(
                record.model_copy(
                    update={
                        "request_id": uuid4(),
                        "cache_hit": True,
                        "cost": Decimal(0),
                    }
                )
            )
            await pending.flush()
            assert await pool.get(spend_key(tenant, when)) == b"3"
            assert float(
                await pool.hget(f"ratelimit:tokens:{tenant}", "tokens")
            ) == pytest.approx(99910, abs=0.001)
            assert (
                sum("monthly_budget_80_percent" in r.message for r in caplog.records)
                == 1
            )
            # Monetary totals must remain exact even beyond Lua's double precision.
            huge = 2**53 + 1
            await pool.set(spend_key(tenant, when), huge)
            pending.record(record.model_copy(update={"request_id": uuid4()}))
            await pending.flush()
            assert int(await pool.get(spend_key(tenant, when))) == huge + 1
        finally:
            await pool.aclose()


@pytest.mark.parametrize("stream", [False, True])
async def test_endpoint_enforces_token_debt_after_batch_flush(monkeypatch, stream):
    import httpx2

    from llm_gateway.auth import AuthContext, authenticate
    from llm_gateway.config import get_settings
    from llm_gateway.main import app, get_provider
    from llm_gateway.providers import FakeProvider, FinalEvent, TextEvent
    from llm_gateway.schemas import Usage

    class MeteredProvider(FakeProvider):
        async def complete(self, request):
            response = await super().complete(request)
            response.usage = Usage(prompt_tokens=10, completion_tokens=20)
            return response

        async def stream(self, request):
            yield TextEvent("hello")
            yield FinalEvent("stop", Usage(prompt_tokens=10, completion_tokens=20))

    monkeypatch.setattr(get_settings(), "tenant_token_capacity", 1)
    monkeypatch.setattr(get_settings(), "tenant_token_refill", 0.001)
    auth = AuthContext(uuid4(), uuid4())
    with RedisContainer("redis:8") as redis:
        pool = await create_pool(
            RedisSettings(
                host=redis.get_container_host_ip(),
                port=int(redis.get_exposed_port(6379)),
            )
        )
        monkeypatch.setattr(
            "llm_gateway.main.create_usage_pool", AsyncMock(return_value=pool)
        )
        app.dependency_overrides[authenticate] = lambda: auth
        app.dependency_overrides[get_provider] = MeteredProvider
        try:
            async with app.router.lifespan_context(app):
                async with httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    payload = {
                        "model": "fake",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": stream,
                    }
                    assert (
                        await client.post("/v1/chat/completions", json=payload)
                    ).status_code == 200
                    await app.state.accounting.flush()
                    response = await client.post("/v1/chat/completions", json=payload)
                    assert response.status_code == 429
                    assert int(response.headers["retry-after"]) > 900
                    assert await pool.zcard("arq:queue") == 1
        finally:
            app.dependency_overrides.clear()
