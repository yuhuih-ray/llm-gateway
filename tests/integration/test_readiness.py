import time
from unittest.mock import AsyncMock

import httpx2
import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.constants import default_queue_name
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
                await pool.zadd(
                    default_queue_name,
                    {
                        "old-job": (time.time() - 700) * 1000,
                        "new-job": time.time() * 1000,
                    },
                )
                backlog = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "checks": {"postgres": "ok", "usage_queue": "ok"},
            "usage_queue": {"depth": 0, "oldest_job_age_seconds": 0.0},
        }
        assert backlog.status_code == 200
        assert backlog.json()["usage_queue"]["depth"] == 2
        assert 700 <= backlog.json()["usage_queue"]["oldest_job_age_seconds"] < 710
        enqueue.assert_not_awaited()
