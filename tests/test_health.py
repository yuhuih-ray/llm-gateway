import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from llm_gateway.main import app


@pytest.mark.parametrize("path", ["/health", "/health/live"])
def test_liveness_ignores_dependencies(path, monkeypatch, isolated_usage_queue):
    factory = Mock(side_effect=ConnectionError("private database hostname"))
    monkeypatch.setattr("llm_gateway.main.get_sessionmaker", factory)
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    factory.assert_not_called()
    isolated_usage_queue.ping.assert_not_awaited()
    isolated_usage_queue.enqueue_job.assert_not_awaited()


@pytest.mark.parametrize("hang", [False, True])
def test_readiness_database_unavailable(
    hang, monkeypatch, isolated_usage_queue, caplog
):
    session = AsyncMock()
    session.__aenter__.return_value = session

    async def execute(query):  # Simulate an unavailable or hung DB without networking.
        if hang:
            await asyncio.sleep(10)
        raise ConnectionError("private-host:5432 secret-connection-string")

    session.execute.side_effect = execute
    monkeypatch.setattr("llm_gateway.main.get_sessionmaker", lambda: lambda: session)
    with TestClient(app) as client:
        started = time.monotonic()
        response = client.get("/health/ready")
        elapsed = time.monotonic() - started
    assert elapsed < 2
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"postgres": "unavailable", "usage_queue": "ok"},
    }
    assert "private-host" not in response.text
    assert "secret-connection-string" not in response.text
    assert "Readiness postgres unavailable" in caplog.text
    session.__aexit__.assert_awaited_once()
    isolated_usage_queue.enqueue_job.assert_not_awaited()


@pytest.mark.parametrize("queue", ["missing", "error", "hang", "false"])
def test_readiness_tolerates_queue_failure(queue, monkeypatch, isolated_usage_queue):
    session = AsyncMock()
    session.__aenter__.return_value = session
    monkeypatch.setattr("llm_gateway.main.get_sessionmaker", lambda: lambda: session)

    async def ping():
        if queue == "hang":
            await asyncio.sleep(10)
        if queue == "false":
            return False
        raise ConnectionError("private Redis exception")

    isolated_usage_queue.ping.side_effect = ping
    with TestClient(app) as client:
        if queue == "missing":
            app.state.usage_pool = None
        started = time.monotonic()
        response = client.get("/health/ready")
        assert time.monotonic() - started < 2
        app.state.usage_pool = isolated_usage_queue
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"postgres": "ok", "usage_queue": "unavailable"},
    }
    assert str(session.execute.call_args.args[0]) == "SELECT 1"
    session.__aexit__.assert_awaited_once()
    isolated_usage_queue.enqueue_job.assert_not_awaited()
