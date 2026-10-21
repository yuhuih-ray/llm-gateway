from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def isolated_usage_queue(monkeypatch):  # Unit/API tests must never use Compose Redis.
    from llm_gateway.ratelimit import RateLimiter

    monkeypatch.setattr("llm_gateway.admin.rate_limiter", RateLimiter())
    monkeypatch.setattr("llm_gateway.auth.failed_auth_limiter", RateLimiter())
    pool = AsyncMock()
    pool.get.return_value = None
    pool.__aenter__.return_value = pool
    monkeypatch.setattr("llm_gateway.auth_cache.create_cli_client", lambda: pool)
    pipeline = MagicMock()
    pipeline.__aenter__.return_value = pipeline
    pipeline.execute = AsyncMock(return_value=[0, True])
    pool.pipeline = MagicMock(return_value=pipeline)
    from llm_gateway.accounting import BATCH_LUA, AccountingBuffer

    pool.eval.return_value = [1, "100", 0]
    pool.eval.side_effect = lambda script, *args: (
        "[]" if script == BATCH_LUA else pool.eval.return_value
    )
    pool.recorded = []
    original_record = AccountingBuffer.record

    def capture(buffer, record):
        pool.recorded.append(record.model_dump(mode="json"))
        original_record(buffer, record)

    monkeypatch.setattr(AccountingBuffer, "record", capture)
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
