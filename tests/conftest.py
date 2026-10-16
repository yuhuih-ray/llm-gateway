from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def isolated_usage_queue(monkeypatch):  # Unit/API tests must never use Compose Redis.
    from llm_gateway.ratelimit import RateLimiter

    monkeypatch.setattr("llm_gateway.admin.rate_limiter", RateLimiter())
    pool = AsyncMock()
    pool.eval.return_value = [1, "100", 0]
    monkeypatch.setattr(
        "llm_gateway.main.create_usage_pool", AsyncMock(return_value=pool)
    )
    return pool
