from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def isolated_usage_queue(monkeypatch):  # Unit/API tests must never use Compose Redis.
    from llm_gateway.ratelimit import RateLimiter

    monkeypatch.setattr("llm_gateway.admin.rate_limiter", RateLimiter())
    pool = AsyncMock()
    pool.get.return_value = None
    pipeline = MagicMock()
    pipeline.__aenter__.return_value = pipeline
    pipeline.execute = AsyncMock(return_value=[0, True])
    pool.pipeline = MagicMock(return_value=pipeline)
    pool.eval.return_value = [1, "100", 0]
    monkeypatch.setattr(
        "llm_gateway.main.create_usage_pool", AsyncMock(return_value=pool)
    )
    return pool


@pytest.fixture(autouse=True)
def isolated_response_cache(monkeypatch):  # Never access local cache Redis in tests.
    client = AsyncMock()
    client.get.return_value = None
    monkeypatch.setattr("llm_gateway.main.create_cache_client", lambda: client)
    return client
