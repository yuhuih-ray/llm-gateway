"""Safety and summary parsing checks; these never execute a load test."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "loadtest_manage",
    Path(__file__).resolve().parents[1] / "scripts/loadtest/manage.py",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


manage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage)


@pytest.mark.parametrize("endpoint", ["database", "redis", "cache"])
def test_refuses_dev_services(monkeypatch, endpoint):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://gateway_loadtest@127.0.0.1:15432/gateway_loadtest",
    )
    monkeypatch.setenv("REDIS_URL", manage.REDIS_URL)
    monkeypatch.setenv("CACHE_REDIS_URL", manage.CACHE_URL)
    assert manage.validate_environment()
    variable, value = {
        "database": (
            "DATABASE_URL",
            "postgresql+asyncpg://dev@localhost:5432/llm_gateway",
        ),
        "redis": ("REDIS_URL", "redis://localhost:6379/0"),
        "cache": ("CACHE_REDIS_URL", "redis://localhost:6380/0"),
    }[endpoint]
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="Refusing"):
        manage.validate_environment()


def test_summary_counter_formats():
    for metric in ({"count": 200}, {"values": {"count": 200}}):
        assert (
            manage.metric_count({"metrics": {"requests_sent": metric}}, "requests_sent")
            == 200
        )


def test_private_configuration_and_key_files(monkeypatch, tmp_path):
    monkeypatch.setattr(manage, "STATE", tmp_path)
    manage.initialize()
    path = tmp_path / "env"
    first = path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert "15432/gateway_loadtest" in first
    manage.initialize()
    assert path.read_text() == first
    keys = tmp_path / "keys.json"
    manage.write_json(keys, ["test-only"])
    assert keys.stat().st_mode & 0o777 == 0o600
    assert json.loads(keys.read_text()) == ["test-only"]


@pytest.mark.anyio
@pytest.mark.parametrize("written,drained", [(80, True), (79, False)])
async def test_drain_compares_successful_requests(
    monkeypatch, tmp_path, written, drained
):
    import argparse
    from unittest.mock import AsyncMock, Mock

    monkeypatch.setattr(manage, "STATE", tmp_path)
    (tmp_path / "baseline.json").write_text(
        json.dumps({"usage_rows": 10, "started_at": "test"})
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "metrics": {
                    "requests_sent": {"count": 100},
                    "requests_successful": {"count": 80},
                }
            }
        )
    )
    pool = AsyncMock()
    pool.__aenter__.return_value = pool
    monkeypatch.setattr(manage, "Redis", Mock(from_url=Mock(return_value=pool)))
    monkeypatch.setattr(
        manage, "queue_state", AsyncMock(return_value={"queued": 0, "in_progress": 0})
    )
    monkeypatch.setattr(manage, "usage_count", AsyncMock(return_value=10 + written))
    monkeypatch.setattr(manage.time, "perf_counter", Mock(side_effect=[0, 31, 31]))
    monkeypatch.setenv("FAKE_DELAY_MS", "0")
    monkeypatch.setenv("SERVER_TIMING", "false")
    output = tmp_path / "meta.json"
    await manage.finish(
        "unused",
        argparse.Namespace(
            summary=str(summary),
            output=str(output),
            scenario="s1",
            smoke=True,
            exit_code=0,
        ),
    )
    assert json.loads(output.read_text())["drained"] is drained
