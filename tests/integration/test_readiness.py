from unittest.mock import AsyncMock

import httpx2
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import async_sessionmaker
from testcontainers.redis import RedisContainer

from llm_gateway.main import app


@pytest.mark.anyio
async def test_readiness_real_postgres_and_redis(engine, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.main.get_sessionmaker", lambda: async_sessionmaker(engine)
    )
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
        enqueue = AsyncMock()
        monkeypatch.setattr(pool, "enqueue_job", enqueue)
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "checks": {"postgres": "ok", "usage_queue": "ok"},
        }
        enqueue.assert_not_awaited()
