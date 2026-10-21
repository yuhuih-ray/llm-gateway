"""Safety and summary parsing checks; these never execute a load test."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "loadtest_manage",
    Path(__file__).resolve().parents[1] / "scripts/loadtest/manage.py",
)
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
