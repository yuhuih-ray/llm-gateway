from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def isolated_usage_queue(monkeypatch):  # Unit/API tests must never use Compose Redis.
    pool = AsyncMock()
    monkeypatch.setattr(
        "llm_gateway.main.create_usage_pool", AsyncMock(return_value=pool)
    )
    return pool
