import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import State

from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.main import app, get_provider
from llm_gateway.providers import FakeProvider, TextEvent
from llm_gateway.registry import MODELS
from llm_gateway.schemas import ChatCompletionRequest, Usage
from llm_gateway.streaming import stream_response
from llm_gateway.usage import UsageTracker, calculate_cost, enqueue_usage


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_cost():
    assert calculate_cost(MODELS["gemini-flash-lite"], 100, 200) == Decimal(
        "0.00032500"
    )
    assert calculate_cost(MODELS["gemini-flash"], 1, 1) == Decimal("0.00000450")
    assert calculate_cost(MODELS["gemini-pro"], 200000, 100) == Decimal("0.40120000")
    assert calculate_cost(MODELS["gemini-pro"], 200001, 100) == Decimal("0.80180400")
    assert calculate_cost(MODELS["fake"], 100, 100) == Decimal("0.00000000")
    assert calculate_cost(MODELS["fake"], None, 100) is None
    assert calculate_cost(MODELS["gemini-pro"], 100, None) is None
    assert calculate_cost(MODELS["gemini-flash-lite"], 1, 0).as_tuple().exponent == -8


def tracker(pool):  # Isolated request accounting state.
    import time

    return UsageTracker(
        dict(
            request_id=uuid4(),
            created_at=datetime.now(timezone.utc),
            started=time.monotonic(),
        ),
        AuthContext(uuid4(), uuid4()),
        "fake",
        State({"usage_pool": pool}),
    )


@pytest.mark.parametrize(
    "stream,fail_after,status",
    [(False, None, "success"), (True, None, "success"), (True, 1, "error")],
)
def test_record_response(
    stream, fail_after, status, isolated_usage_queue
):  # Capture enqueued payload.
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    app.dependency_overrides[get_provider] = lambda: FakeProvider(fail_after=fail_after)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream,
                },
            )
        assert response.status_code == 200
        record = isolated_usage_queue.enqueue_job.call_args.args[1]
        assert record["request_id"] == response.headers["X-Request-ID"]
        assert record["status"] == status
        assert record["ttft_ms"] is not None if stream else record["ttft_ms"] is None
        assert isolated_usage_queue.enqueue_job.call_count == 1
        assert (
            isolated_usage_queue.enqueue_job.call_args.kwargs["_job_id"]
            == record["request_id"]
        )
        if status == "error":
            assert record["error_type"] == "upstream_error"
            assert record["cost"] is None
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("stream", [False, True])
def test_enqueue_failure_preserves_response(
    stream, isolated_usage_queue, caplog
):  # Queue failure cannot fail HTTP.
    isolated_usage_queue.enqueue_job.side_effect = ConnectionError()
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream,
                },
            )
        assert response.status_code == 200
        logs = [
            r.message for r in caplog.records if "usage_record_dropped" in r.message
        ]
        assert len(logs) == 1
        assert (
            json.loads(logs[0].split(" ", 1)[1])["request_id"]
            == response.headers["X-Request-ID"]
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_cancel_record():
    pool = AsyncMock()
    account = tracker(pool)
    waiting = asyncio.Event()

    async def stream(request):  # Wait indefinitely after the first text.
        yield TextEvent("hello")
        waiting.set()
        await asyncio.Event().wait()

    response = await stream_response(
        SimpleNamespace(stream=stream),
        ChatCompletionRequest(
            model="fake", messages=[{"role": "user", "content": "hi"}]
        ),
        account,
    )

    async def consume():
        async for _ in response.body_iterator:
            pass

    task = asyncio.create_task(consume())
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.enqueue_job.call_args.args[1]["status"] == "cancelled"
    assert pool.enqueue_job.call_args.args[1]["prompt_tokens"] is None


@pytest.mark.anyio
async def test_enqueue_deadline(caplog):
    pool = AsyncMock()

    async def hang(*args, **kwargs):  # A stuck queue must remain bounded.
        await asyncio.sleep(10)

    pool.enqueue_job.side_effect = hang
    await asyncio.wait_for(
        enqueue_usage(State({"usage_pool": pool}), tracker(pool).record), 1
    )
    assert "usage_record_dropped" in caplog.text


def test_reasoning_is_not_charged_twice():
    account = tracker(None)
    account.record.model_selected = "gemini-flash-lite"
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=200,
        completion_tokens_details={"reasoning_tokens": 150},
    )
    account.success(usage)
    assert account.record.reasoning_tokens == 150
    assert account.record.cost == Decimal("0.00032500")


@pytest.mark.parametrize("case", ["auth", "model", "validation"])
def test_rejected_request_not_recorded(
    case, isolated_usage_queue
):  # Never reached a provider.
    from llm_gateway.db import get_sessionmaker

    app.dependency_overrides[get_sessionmaker] = lambda: None
    if case != "auth":
        app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "unknown" if case == "model" else "fake",
                    "messages": []
                    if case == "validation"
                    else [{"role": "user", "content": "hi"}],
                },
            )
        assert (
            response.status_code == {"auth": 401, "model": 404, "validation": 422}[case]
        )
        assert response.headers["X-Request-ID"]
        isolated_usage_queue.enqueue_job.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_worker_retries_db_errors(caplog):
    from arq import Retry
    from sqlalchemy.exc import OperationalError

    from llm_gateway.worker import write_usage

    session = AsyncMock()
    session.begin = lambda: AsyncMock()
    session.execute.side_effect = OperationalError("insert", {}, Exception("offline"))
    context = AsyncMock()
    context.__aenter__.return_value = session
    payload = tracker(None).record.model_dump(mode="json")
    with pytest.raises(Retry):
        await write_usage({"sessions": lambda: context, "job_try": 1}, payload)
    assert "usage_record_failed" not in caplog.text
    with pytest.raises(OperationalError):
        await write_usage({"sessions": lambda: context, "job_try": 5}, payload)
    logs = [r.message for r in caplog.records if "usage_record_failed" in r.message]
    assert len(logs) == 1
    assert json.loads(logs[0].split(" ", 1)[1]) == payload


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_timeout_record(stream, isolated_usage_queue):
    from llm_gateway.errors import GatewayError

    class TimeoutProvider:
        async def complete(self, request):
            raise GatewayError(504, "Upstream timed out", "upstream_error")

        async def stream(self, request):
            raise GatewayError(504, "Upstream timed out", "upstream_error")
            yield  # Keep the same async iterator interface as real providers.

    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    app.dependency_overrides[get_provider] = TimeoutProvider
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream,
                },
            )
        assert response.status_code == 504
        record = isolated_usage_queue.enqueue_job.call_args.args[1]
        assert record["request_id"] == response.headers["X-Request-ID"]
        assert record["status"] == "error"
        assert record["error_type"] == "upstream_timeout"
        assert record["prompt_tokens"] is None
        assert record["cost"] is None
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_worker_integrity_failure_is_not_retried(caplog):
    from sqlalchemy.exc import IntegrityError

    from llm_gateway.worker import write_usage

    session = AsyncMock()
    session.begin = lambda: AsyncMock()
    session.execute.side_effect = IntegrityError("insert", {}, Exception("constraint"))
    context = AsyncMock()
    context.__aenter__.return_value = session
    payload = tracker(None).record.model_dump(mode="json")
    with pytest.raises(IntegrityError):
        await write_usage({"sessions": lambda: context, "job_try": 1}, payload)
    logs = [r.message for r in caplog.records if "usage_record_failed" in r.message]
    assert len(logs) == 1
    assert json.loads(logs[0].split(" ", 1)[1]) == payload


def test_queue_recovers_after_startup_failure(monkeypatch):
    from llm_gateway.usage import create_usage_pool

    clock = [100.0]
    pool = AsyncMock()
    create = AsyncMock(side_effect=[ConnectionError(), ConnectionError(), pool])
    monkeypatch.setattr("llm_gateway.main.create_usage_pool", create_usage_pool)
    monkeypatch.setattr("llm_gateway.usage.create_pool", create)
    monkeypatch.setattr("llm_gateway.main.monotonic", lambda: clock[0])
    monkeypatch.setattr("llm_gateway.usage.monotonic", lambda: clock[0])
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    payload = {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}
    try:
        with TestClient(app) as client:
            assert app.state.usage_pool is None
            assert client.post("/v1/chat/completions", json=payload).status_code == 200
            assert create.await_count == 1
            clock[0] = 110
            for _ in range(2):
                assert (
                    client.post("/v1/chat/completions", json=payload).status_code == 200
                )
            assert create.await_count == 2  # One recovery attempt within this window.
            clock[0] = 120
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200
            assert app.state.usage_pool is pool
            assert (
                pool.enqueue_job.call_args.args[1]["request_id"]
                == response.headers["X-Request-ID"]
            )
            client.post("/v1/chat/completions", json=payload)
            assert create.await_count == 3
            assert pool.enqueue_job.await_count == 2
            assert all(call.args[0].conn_retries == 0 for call in create.call_args_list)
        pool.aclose.assert_awaited_once()
    finally:
        app.dependency_overrides.clear()
