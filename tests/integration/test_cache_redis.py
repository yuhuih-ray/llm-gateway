from unittest.mock import AsyncMock
from uuid import uuid4

import httpx2
import pytest
from redis.asyncio import Redis
from testcontainers.redis import RedisContainer

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.main import app
from llm_gateway.providers import FakeProvider
from llm_gateway.registry import get_provider


@pytest.mark.anyio
async def test_dedicated_redis_cache(monkeypatch):
    with RedisContainer("redis:8") as container:
        cache = Redis(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(6379)),
        )
        monkeypatch.setattr("llm_gateway.main.create_cache_client", lambda: cache)
        auth = AuthContext(uuid4(), uuid4())
        provider = FakeProvider()
        provider.complete = AsyncMock(wraps=provider.complete)
        app.dependency_overrides[authenticate] = lambda: auth
        app.dependency_overrides[get_provider] = lambda: provider
        try:
            async with app.router.lifespan_context(app):
                async with httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app),
                    base_url="http://test",
                    headers={"X-Gateway-Cache": "on"},
                ) as client:
                    body = {
                        "model": "fake",
                        "temperature": 0,
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                    first = await client.post("/v1/chat/completions", json=body)
                    second = await client.post("/v1/chat/completions", json=body)
                    assert first.headers["x-cache"] == "MISS"
                    assert second.headers["x-cache"] == "HIT"
                    assert first.json()["choices"] == second.json()["choices"]
                    provider.complete.assert_awaited_once()
                    keys = await cache.keys("*")
                    assert len(keys) == 1
                    assert keys[0].startswith(b"cache:v1:")
                    assert 86390 <= await cache.ttl(keys[0]) <= 86400
        finally:
            app.dependency_overrides.clear()
