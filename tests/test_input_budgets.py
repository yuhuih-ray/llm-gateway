import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from llm_gateway import budgets
from llm_gateway.auth import AuthContext, authenticate
from llm_gateway.config import Settings
from llm_gateway.input_limits import BodySizeMiddleware
from llm_gateway.main import app
from llm_gateway.registry import MODELS


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("declared", [True, False])
async def test_body_limit_before_app(monkeypatch, declared):
    monkeypatch.setattr(
        "llm_gateway.input_limits.get_settings",
        lambda: Settings(_env_file=None, max_request_bytes=10),
    )
    inner = AsyncMock()
    receive = AsyncMock(
        side_effect=[
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"123456", "more_body": True},
        ]
    )
    sent = []

    async def send(message):
        sent.append(message)

    await BodySizeMiddleware(inner)(
        {
            "type": "http",
            "method": "POST",
            "path": "/auth/login",
            "headers": [(b"content-length", b"12")] if declared else [],
        },
        receive,
        send,
    )
    inner.assert_not_called()
    assert receive.await_count == (0 if declared else 2)
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"] == {
        "message": "Request body too large",
        "type": "invalid_request_error",
        "code": "request_too_large",
    }


@pytest.mark.parametrize("content", ["x" * 6, "中文字符测试"])
def test_all_message_characters_checked_before_limits(
    content, monkeypatch, isolated_usage_queue
):
    monkeypatch.setitem(MODELS, "fake", replace(MODELS["fake"], max_input_chars=10))
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    payload = {
        "model": "fake",
        "messages": [
            {"role": role, "content": content}
            for role in ["system", "assistant", "user"]
        ],
    }
    try:
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"
        isolated_usage_queue.eval.assert_not_called()
        isolated_usage_queue.enqueue_job.assert_not_called()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("budget,status", [(Decimal("1"), 429), (None, 200)])
def test_budget_endpoint(budget, status, isolated_usage_queue):
    isolated_usage_queue.get.return_value = b"100000000"
    app.dependency_overrides[authenticate] = lambda: AuthContext(
        uuid4(), uuid4(), budget
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == status
        if status == 429:
            assert "retry-after" not in response.headers
            assert response.json() == {
                "error": {
                    "message": "Monthly budget exceeded",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            }
            isolated_usage_queue.enqueue_job.assert_not_called()
            # Request rate limits are consumed before checking the budget.
            isolated_usage_queue.eval.assert_awaited_once()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_budget_unavailable_warns_at_most_every_30_seconds(monkeypatch, caplog):
    clock = [0.0]
    monkeypatch.setattr(budgets, "monotonic", lambda: clock[0])
    monkeypatch.setattr(budgets, "last_warning", -float("inf"))
    for _ in range(3):
        assert not await budgets.budget_exceeded(None, uuid4(), Decimal("1"))
    assert len(caplog.records) == 1
    clock[0] = 30
    assert not await budgets.budget_exceeded(None, uuid4(), Decimal("1"))
    assert len(caplog.records) == 2


@pytest.mark.anyio
async def test_budget_timeout_allows_and_threshold_failure_does_not_bypass():
    pool = AsyncMock()

    async def hang(*args):
        await asyncio.sleep(10)

    pool.get.side_effect = hang
    async with asyncio.timeout(0.5):
        assert not await budgets.budget_exceeded(pool, uuid4(), Decimal("1"))
    pool.get.side_effect = None
    pool.get.return_value = b"100000000"
    pool.set.side_effect = ConnectionError()
    assert await budgets.budget_exceeded(pool, uuid4(), Decimal("1"))


def test_new_month_key():
    tenant = uuid4()
    assert budgets.spend_key(
        tenant, datetime(2026, 12, 31, tzinfo=timezone.utc)
    ) != budgets.spend_key(tenant, datetime(2027, 1, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    "value", ["-1", "NaN", "Infinity", "1.001", "10000000000", "abc"]
)
def test_cli_rejects_invalid_budgets(value):
    from llm_gateway.cli import parse_budget

    with pytest.raises(ValueError):
        parse_budget(value)


@pytest.mark.parametrize(
    "path", ["/health/live", "/auth/login", "/admin/keys", "/v1/chat/completions"]
)
def test_body_middleware_wraps_every_endpoint(path, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.input_limits.get_settings",
        lambda: Settings(_env_file=None, max_request_bytes=10),
    )
    with TestClient(app) as client:
        response = client.post(path, content=b"x" * 11)
    assert response.status_code == 413
    if path == "/v1/chat/completions":
        assert "x-request-id" in response.headers


@pytest.mark.anyio
async def test_body_at_limit_replayed_without_changes(monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.input_limits.get_settings",
        lambda: Settings(_env_file=None, max_request_bytes=10),
    )

    async def inner(scope, receive, send):
        assert (await receive())["body"] == b"1234567890"

    await BodySizeMiddleware(inner)(
        {"type": "http", "headers": []},
        AsyncMock(
            side_effect=[
                {"type": "http.request", "body": b"12345", "more_body": True},
                {"type": "http.request", "body": b"67890", "more_body": False},
            ]
        ),
        AsyncMock(),
    )


@pytest.mark.anyio
async def test_cost_charged_after_response_and_bounded(monkeypatch):
    from starlette.datastructures import State

    from llm_gateway.accounting import SPEND_LUA, AccountingBuffer
    from llm_gateway.usage import UsageMiddleware, UsageTracker

    final_sent = False
    pool = AsyncMock()
    state = State({"usage_pool": pool})
    state.accounting = AccountingBuffer(state)
    entered = asyncio.Event()

    async def slow_eval(*args):
        assert final_sent
        entered.set()
        await asyncio.Event().wait()

    pool.eval.side_effect = slow_eval
    tracker = UsageTracker(
        {"started": 0, "created_at": datetime.now(timezone.utc), "request_id": uuid4()},
        AuthContext(uuid4(), uuid4()),
        "fake",
        state,
    )
    tracker.record.cost = Decimal("0.12345678")

    async def inner(scope, receive, send):
        scope["state"]["usage_tracker"] = tracker
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
        nonlocal final_sent
        if message["type"] == "http.response.body":
            final_sent = True

    async with asyncio.timeout(0.05):
        await UsageMiddleware(inner)(
            {"type": "http", "path": "/v1/chat/completions"}, AsyncMock(), send
        )
    pool.eval.assert_not_awaited()
    pool.enqueue_job.assert_not_awaited()
    assert state.accounting.records[0].cost == Decimal("0.12345678")
    flush = asyncio.create_task(state.accounting.flush())
    await entered.wait()
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    assert pool.eval.call_args.args[0] == SPEND_LUA
    assert pool.eval.call_args.args[-2] == 12345678
