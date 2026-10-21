import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from llm_gateway import timing
from llm_gateway.auth import generate_key, hash_key
from llm_gateway.config import Settings
from llm_gateway.db import get_sessionmaker
from llm_gateway.main import app


@pytest.fixture
def timing_client(monkeypatch):
    key = generate_key()
    tenant_id = uuid4()
    session = AsyncMock()
    session.execute.return_value = Mock(
        one_or_none=Mock(
            return_value=(
                SimpleNamespace(id=uuid4(), key_hash=hash_key(key), revoked_at=None),
                SimpleNamespace(id=tenant_id, status="active", monthly_budget_usd=None),
            )
        )
    )
    context = AsyncMock()
    context.__aenter__.return_value = session
    app.dependency_overrides[get_sessionmaker] = lambda: lambda: context
    clock = Mock(side_effect=itertools.count(step=0.001))
    monkeypatch.setattr(timing, "time", SimpleNamespace(perf_counter=clock))
    try:
        with TestClient(app) as client:
            yield client, key, str(tenant_id), clock
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("stream", [False, True])
def test_fixed_stages_measured_in_milliseconds(monkeypatch, timing_client, stream):
    monkeypatch.setattr(
        timing, "get_settings", lambda: Settings(_env_file=None, server_timing=True)
    )
    client, key, tenant, _ = timing_client
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key},
        json={
            "model": "fake",
            "messages": [{"role": "user", "content": "private prompt"}],
            "stream": stream,
        },
    )
    assert response.status_code == 200
    header = response.headers["Server-Timing"]
    values = dict(item.split(";dur=") for item in header.split(", "))
    assert set(values) == {*timing.STAGES, "total"}
    for stage in timing.STAGES:
        assert float(values[stage]) == pytest.approx(1.0)
    assert float(values["total"]) >= 5
    assert all(secret not in header for secret in (key, tenant, "private prompt"))


def test_disabled_by_default(monkeypatch, timing_client):
    settings = Settings(_env_file=None)
    assert settings.server_timing is False
    monkeypatch.setattr(timing, "get_settings", lambda: settings)
    client, key, _, clock = timing_client
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + key},
        json={
            "model": "fake",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    assert "Server-Timing" not in response.headers
    clock.assert_not_called()


def test_failed_auth_has_timing_without_error_details(monkeypatch, timing_client):
    monkeypatch.setattr(
        timing, "get_settings", lambda: Settings(_env_file=None, server_timing=True)
    )
    client, _, _, _ = timing_client
    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 401
    assert "auth;dur=1.000" in response.headers["Server-Timing"]
    assert "provider;dur=0.000" in response.headers["Server-Timing"]


def test_repeated_cache_measurements_accumulate(monkeypatch):
    clock = Mock(side_effect=[1, 1.01, 2, 2.02])
    monkeypatch.setattr(timing, "time", SimpleNamespace(perf_counter=clock))
    request = Request(
        {"type": "http", "state": {"server_timing": dict.fromkeys(timing.STAGES, 0.0)}}
    )
    with timing.measure(request, "cache"):
        pass
    with timing.measure(request, "cache"):
        pass
    assert request.state.server_timing["cache"] == pytest.approx(30)
