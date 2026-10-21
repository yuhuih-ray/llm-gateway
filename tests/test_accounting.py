import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from starlette.datastructures import State

from llm_gateway.accounting import BATCH_LUA, AccountingBuffer
from llm_gateway.usage import UsageRecord


@pytest.fixture
def anyio_backend():
    return "asyncio"


def usage(**changes):  # Unique requests; explicit counts and exact cost.
    values = dict(
        request_id=uuid4(),
        tenant_id=uuid4(),
        api_key_id=uuid4(),
        model_requested="fake",
        model_selected="fake",
        upstream_model=None,
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        status="success",
        prompt_tokens=10,
        completion_tokens=20,
        cost=Decimal("0.12345678"),
    )
    return UsageRecord(**(values | changes))


def buffer(pool):
    return AccountingBuffer(State({"usage_pool": pool}))


def test_full_buffer_logs_new_record(caplog, isolated_usage_queue):
    pending = buffer(isolated_usage_queue)
    pending.capacity = 2
    records = [usage() for _ in range(3)]
    for record in records:
        pending.record(record)
    assert list(pending.records) == records[:2]
    assert pending.dropped == 1
    message = next(
        r.message for r in caplog.records if "usage_record_dropped" in r.message
    )
    assert json.loads(message.split(" ", 1)[1]) == records[2].model_dump(mode="json")
    isolated_usage_queue.eval.assert_not_awaited()


@pytest.mark.anyio
async def test_batches_aggregate_tenants_months_and_cache_hits(isolated_usage_queue):
    pool = isolated_usage_queue
    pending = buffer(pool)
    tenant = uuid4()
    records = [usage(tenant_id=tenant) for _ in range(501)]
    other = usage()
    cached = usage(tenant_id=tenant, cache_hit=True, cost=Decimal(0))
    unknown = usage(
        tenant_id=tenant, prompt_tokens=None, completion_tokens=None, cost=None
    )
    prior = usage(
        tenant_id=tenant, created_at=datetime(2026, 9, 30, tzinfo=timezone.utc)
    )
    for record in [*records, other, cached, unknown, prior]:
        pending.record(record)
    await pending.flush()
    assert len(pending.records) == 5
    await pending.flush()
    assert not pending.records
    jobs = pool.enqueue_job.call_args_list
    assert [len(call.args[1]) for call in jobs] == [500, 5]
    assert all(call.args[0] == "write_usage_batch" for call in jobs)
    assert len({call.kwargs["_job_id"] for call in jobs}) == 2
    batches = [json.loads(c.args[-2]) for c in pool.eval.call_args_list]
    charges = [charge for batch in batches for charge in batch["tokens"]]
    assert sorted(c["amount"] for c in charges) == [30, 60, 15000]
    spends = [spend for batch in batches for spend in batch["spend"]]
    assert sum(int(c["amount"]) for c in spends) == 503 * 12345678
    assert len(spends) == 4
    prior_spend = [
        spend
        for call in pool.eval.call_args_list
        for spend in json.loads(call.args[-2])["spend"]
        if call.args[spend["key"] + 1].endswith("2026-09")
    ]
    assert sum(int(c["amount"]) for c in prior_spend) == 12345678


@pytest.mark.anyio
async def test_retry_retains_inflight_cap_and_stable_job_id(
    isolated_usage_queue, caplog
):
    pending = buffer(isolated_usage_queue)
    pending.capacity = 1
    record = usage()
    pending.record(record)
    isolated_usage_queue.enqueue_job.side_effect = [ConnectionError(), object()]
    with pytest.raises(ConnectionError):
        await pending.flush()
    pending.record(usage())
    assert pending.dropped == 1
    assert list(pending.records) == [record]
    await pending.flush()
    assert not pending.records
    assert (
        len(
            {
                c.kwargs["_job_id"]
                for c in isolated_usage_queue.enqueue_job.call_args_list
            }
        )
        == 1
    )


@pytest.mark.anyio
async def test_flusher_timer_threshold_and_shutdown(isolated_usage_queue, monkeypatch):
    pool = isolated_usage_queue
    pending = buffer(pool)
    pending.record(usage())
    pending.start()
    async with asyncio.timeout(1):
        while pending.records:
            await asyncio.sleep(0.005)
    assert pool.enqueue_job.await_count == 1
    # A large interval proves 500 records wake the sleeping task immediately.
    monkeypatch.setattr("llm_gateway.accounting.INTERVAL", 60)
    await asyncio.sleep(0.12)
    record = usage()
    for _ in range(500):
        pending.record(record.model_copy(update={"request_id": uuid4()}))
    async with asyncio.timeout(1):
        while pending.records:
            await asyncio.sleep(0.005)
    assert pool.enqueue_job.await_count == 2
    pending.record(usage())
    await pending.close()
    assert pool.enqueue_job.await_count == 3
    assert not pending.records
    assert pending.task.done()


@pytest.mark.anyio
async def test_recovery_and_warning_throttle(isolated_usage_queue, caplog):
    pool = isolated_usage_queue
    pending = buffer(pool)
    pool.eval.side_effect = [ConnectionError(), ConnectionError(), "[]"]
    pending.record(usage())
    pending.start()
    async with asyncio.timeout(2):
        while pending.records:
            await asyncio.sleep(0.01)
    await pending.close()
    assert pool.enqueue_job.await_count == 1
    assert pending.dropped == 0
    assert sum("Accounting Redis unavailable" in r.message for r in caplog.records) == 1


@pytest.mark.anyio
async def test_cleanup_retry_does_not_reenqueue_or_charge(isolated_usage_queue):
    pool = isolated_usage_queue
    pending = buffer(pool)
    pool.delete.side_effect = [ConnectionError(), 1]
    pending.record(usage())
    with pytest.raises(ConnectionError):
        await pending.flush()
    assert len(pending.records) == 1
    await pending.flush()
    pool.enqueue_job.assert_awaited_once()
    assert pool.eval.await_count == 1
    assert not pending.records


@pytest.mark.anyio
@pytest.mark.parametrize("integrity", [False, True])
async def test_batch_worker_failure_rules(integrity, caplog):
    from arq import Retry
    from sqlalchemy.exc import IntegrityError, OperationalError

    from llm_gateway.worker import write_usage_batch

    session = AsyncMock()
    session.begin = lambda: AsyncMock()
    error = IntegrityError if integrity else OperationalError
    session.execute.side_effect = error("insert", {}, Exception("failed"))
    context = AsyncMock()
    context.__aenter__.return_value = session
    payloads = [usage().model_dump(mode="json") for _ in range(2)]
    if not integrity:
        with pytest.raises(Retry):
            await write_usage_batch(
                {"sessions": lambda: context, "job_try": 1}, payloads
            )
        assert "usage_record_failed" not in caplog.text
    with pytest.raises(error):
        await write_usage_batch(
            {"sessions": lambda: context, "job_try": 1 if integrity else 5}, payloads
        )
    logs = [
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if "usage_record_failed" in r.message
    ]
    assert logs == payloads


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_http_response_does_not_wait_for_hung_accounting(
    stream,
    monkeypatch,
    isolated_usage_queue,
):
    import httpx2

    from llm_gateway.auth import AuthContext, authenticate
    from llm_gateway.main import app

    entered = asyncio.Event()
    pool = isolated_usage_queue

    async def eval_script(script, *args):
        if script == BATCH_LUA:
            entered.set()
            await asyncio.Event().wait()
        return [1, "100", 0]

    pool.eval.side_effect = eval_script
    monkeypatch.setattr("llm_gateway.accounting.SHUTDOWN_TIMEOUT", 0.01)
    app.dependency_overrides[authenticate] = lambda: AuthContext(uuid4(), uuid4())
    try:
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                payload = {
                    "model": "fake",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream,
                }
                first = await client.post("/v1/chat/completions", json=payload)
                assert first.status_code == 200
                app.state.accounting.wake.set()
                async with asyncio.timeout(1):
                    await entered.wait()
                # The background flusher remains stuck, but another complete HTTP
                # exchange (including ASGI cleanup) cannot wait for accounting.
                async with asyncio.timeout(0.2):
                    response = await client.post("/v1/chat/completions", json=payload)
                assert response.status_code == 200
                assert len(app.state.accounting.records) == 2
                pool.enqueue_job.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_shutdown_timeout_cancels_flusher(isolated_usage_queue, caplog):
    pending = buffer(isolated_usage_queue)
    isolated_usage_queue.eval.side_effect = ConnectionError("Redis unavailable")
    pending.record(usage())
    pending.start()
    started = asyncio.get_running_loop().time()
    try:
        async with asyncio.timeout(6):
            await pending.close()
        elapsed = asyncio.get_running_loop().time() - started
        assert 4.9 <= elapsed < 6
        assert pending.task.done()
        assert pending.task.cancelled()
        assert isolated_usage_queue.eval.await_count > 1
        isolated_usage_queue.enqueue_job.assert_not_awaited()
        assert len(pending.records) == 1
        assert (
            sum(
                r.message == "accounting_shutdown_unflushed count=1"
                for r in caplog.records
            )
            == 1
        )
    finally:
        # Keep a regression from leaking a retrying task into subsequent tests.
        pending.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending.task


@pytest.mark.anyio
async def test_fifty_tenants_use_three_redis_calls(isolated_usage_queue, caplog):
    pool = isolated_usage_queue
    pending = buffer(pool)
    tenants = [uuid4() for _ in range(50)]
    for _ in range(10):
        for tenant in tenants:
            pending.record(usage(tenant_id=tenant, monthly_budget_usd=Decimal("1")))
    pool.eval.side_effect = None
    pool.eval.return_value = json.dumps(list(range(1, 51)))
    await pending.flush()
    assert [call[0] for call in pool.method_calls] == ["eval", "enqueue_job", "delete"]
    call = pool.eval.call_args
    assert call.args[0] == BATCH_LUA
    batch = json.loads(call.args[-2])
    assert len(batch["tokens"]) == len(batch["spend"]) == 50
    assert all(charge["amount"] == 300 for charge in batch["tokens"])
    assert all(spend["amount"] == "123456780" for spend in batch["spend"])
    assert all(spend["threshold"] == "80000000" for spend in batch["spend"])
    assert pool.delete.call_args.args == (f"accounting:{pending.batch_id}",)
    assert len(pool.enqueue_job.call_args.args[1]) == 500
    assert sum("monthly_budget_80_percent" in r.message for r in caplog.records) == 50
    assert not pending.records
