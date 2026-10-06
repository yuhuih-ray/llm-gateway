import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg
import pytest
from sqlalchemy import insert, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from llm_gateway import partitions
from llm_gateway.admin import usage_report
from llm_gateway.admin_auth import UserContext
from llm_gateway.models import ApiKey, Tenant, UsageLog
from llm_gateway.partitions import add_months, month_start

pytestmark = pytest.mark.anyio
OLD = "1d76484b0ba6"


@pytest.fixture
async def partition_db(database_url):  # Separate throwaway DB per destructive DDL test.
    name = "partition_test_" + uuid4().hex
    base = make_url(database_url)
    admin = await asyncpg.connect(
        base.set(drivername="postgresql").render_as_string(hide_password=False)
    )
    await admin.execute(f'CREATE DATABASE "{name}"')
    url = base.set(database=name)
    engine = create_async_engine(url)

    def migrate(*args, success=True):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *args],
            env={
                **os.environ,
                "DATABASE_URL": url.render_as_string(hide_password=False),
            },
            capture_output=True,
            text=True,
        )
        if success:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0
        return result

    try:
        migrate("upgrade", "head")
        yield SimpleNamespace(engine=engine, migrate=migrate, ctx={"engine": engine})
    finally:
        await engine.dispose()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


async def values(engine):  # Real referenced rows, independent of application caches.
    async with engine.begin() as connection:
        tenant = await connection.scalar(
            insert(Tenant).values(name="partition-tenant").returning(Tenant.id)
        )
        key = await connection.scalar(
            insert(ApiKey)
            .values(tenant_id=tenant, name="test", key_prefix="test", key_hash="test")
            .returning(ApiKey.id)
        )
    return dict(
        tenant_id=tenant,
        api_key_id=key,
        request_id=uuid4(),
        model_requested="fake",
        status="success",
    )


async def partition_names(engine):
    async with engine.connect() as connection:
        return {row.name for row in await partitions.children(connection)}


async def assert_month_coverage(engine, start):
    async with engine.connect() as connection:
        ranges = [
            partitions.partition_range(row.bound)
            for row in await partitions.children(connection)
            if row.bound != "DEFAULT"
        ]
    for offset in range(3):
        lower, upper = add_months(start, offset), add_months(start, offset + 1)
        assert any(
            bounds is not None
            and (bounds[0] is None or bounds[0] <= lower)
            and (bounds[1] is None or bounds[1] >= upper)
            for bounds in ranges
        )


async def test_ensure_current_month_covered_by_legacy(partition_db, monkeypatch):
    db = partition_db
    current = month_start(datetime.now(timezone.utc))
    monkeypatch.setattr(partitions, "utc_now", lambda: current)
    before = await partition_names(db.engine)
    await partitions.ensure_partitions(db.ctx)
    await partitions.ensure_partitions(db.ctx)
    assert await partition_names(db.engine) == before
    assert f"usage_logs_{current:%Y_%m}" not in before
    assert "usage_logs_legacy" in before
    assert {f"usage_logs_{add_months(current, n):%Y_%m}" for n in (1, 2)} <= before
    await assert_month_coverage(db.engine, current)


async def test_ensure_respects_other_existing_ranges(partition_db, monkeypatch):
    db = partition_db
    current = add_months(month_start(datetime.now(timezone.utc)), 6)
    monkeypatch.setattr(partitions, "utc_now", lambda: current)
    # An existing two-month range covers two requested months regardless of its name.
    async with db.engine.begin() as connection:
        await connection.execute(
            text(
                f"CREATE TABLE usage_logs_{current:%Y_%m} PARTITION OF usage_logs FOR VALUES FROM ('{current.isoformat()}') TO ('{add_months(current, 2).isoformat()}')"
            )
        )
    await partitions.ensure_partitions(db.ctx)
    assert f"usage_logs_{add_months(current, 1):%Y_%m}" not in await partition_names(
        db.engine
    )
    await assert_month_coverage(db.engine, current)


async def test_no_copy_migration_identity_indexes_and_downgrade(partition_db):
    db = partition_db
    db.migrate("downgrade", OLD)
    payload = await values(db.engine)
    now = datetime.now(timezone.utc)
    async with db.engine.begin() as connection:
        old_id = await connection.scalar(
            insert(UsageLog).values(**payload, created_at=now).returning(UsageLog.id)
        )
        oid, node = (
            await connection.execute(
                text(
                    "SELECT oid, relfilenode FROM pg_class WHERE oid='usage_logs'::regclass"
                )
            )
        ).one()
        index_oid = await connection.scalar(
            text("SELECT 'ix_usage_logs_tenant_id_created_at'::regclass::oid")
        )
    db.migrate("upgrade", "head")
    async with db.engine.begin() as connection:
        assert (
            await connection.execute(
                text(
                    "SELECT oid, relfilenode FROM pg_class WHERE oid='usage_logs_legacy'::regclass"
                )
            )
        ).one() == (oid, node)
        assert (
            await connection.scalar(
                text(
                    "SELECT 'ix_usage_logs_legacy_tenant_id_created_at'::regclass::oid"
                )
            )
            == index_oid
        )
        row = (
            await connection.execute(
                text("SELECT id, tableoid::regclass::text AS location FROM usage_logs")
            )
        ).one()
        assert row == (old_id, "usage_logs_legacy")
        assert (
            await connection.scalar(
                text(
                    "SELECT count(*) FROM pg_index WHERE indrelid='usage_logs'::regclass AND NOT indisvalid"
                )
            )
            == 0
        )
        # Current month is covered by legacy until the next-month cutover.
        current_id = await connection.scalar(
            insert(UsageLog)
            .values(**{**payload, "request_id": uuid4()}, created_at=now)
            .returning(UsageLog.id)
        )
        assert current_id > old_id
    db.migrate("downgrade", OLD)
    async with db.engine.begin() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM usage_logs")) == 2
        restored_id = await connection.scalar(
            insert(UsageLog)
            .values(**{**payload, "request_id": uuid4()}, created_at=now)
            .returning(UsageLog.id)
        )
        assert restored_id > current_id
    db.migrate("upgrade", "head")
    db.migrate("check")


async def test_monthly_routing_retry_and_safe_downgrade(partition_db):
    db = partition_db
    payload = await values(db.engine)
    current = add_months(month_start(datetime.now(timezone.utc)), 1)
    async with db.engine.begin() as connection:
        result = await connection.execute(
            insert(UsageLog)
            .values(**payload, created_at=current)
            .returning(UsageLog.id)
        )
        assert result.scalar_one() > 0
        assert (
            await connection.scalar(
                text("SELECT tableoid::regclass::text FROM usage_logs")
            )
            == f"usage_logs_{current:%Y_%m}"
        )
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        retried = await connection.execute(
            pg_insert(UsageLog)
            .values(**payload, created_at=current)
            .on_conflict_do_nothing(index_elements=["request_id", "created_at"])
            .returning(UsageLog.id)
        )
        assert retried.all() == []
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError, match="request_id_created_at"):
        async with db.engine.begin() as connection:
            await connection.execute(
                insert(UsageLog).values(**payload, created_at=current)
            )
    failure = db.migrate("downgrade", OLD, success=False)
    assert "Cannot downgrade without data loss" in failure.stderr
    async with db.engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM usage_logs")) == 1


async def test_ensure_idempotent_and_default_alert(partition_db, monkeypatch, caplog):
    db = partition_db
    # Advance beyond pre-created partitions; legacy is still attached, but disjoint.
    now = add_months(month_start(datetime.now(timezone.utc)), 6)
    monkeypatch.setattr(partitions, "utc_now", lambda: now)
    await partitions.ensure_partitions(db.ctx)
    first = await partition_names(db.engine)
    await partitions.ensure_partitions(db.ctx)
    assert await partition_names(db.engine) == first
    await assert_month_coverage(db.engine, now)
    payload = await values(db.engine)
    async with db.engine.begin() as connection:
        await connection.execute(
            insert(UsageLog).values(**payload, created_at=add_months(now, 10))
        )
        assert (
            await connection.scalar(
                text("SELECT tableoid::regclass::text FROM usage_logs")
            )
            == "usage_logs_default"
        )
    await partitions.ensure_partitions(db.ctx)
    assert "usage_default_partition_not_empty count=1" in caplog.text


async def test_default_blocks_creation_without_data_loss(
    partition_db, monkeypatch, caplog
):
    db = partition_db
    now = add_months(month_start(datetime.now(timezone.utc)), 6)
    monkeypatch.setattr(partitions, "utc_now", lambda: now)
    payload = await values(db.engine)
    async with db.engine.begin() as connection:
        await connection.execute(insert(UsageLog).values(**payload, created_at=now))
    await partitions.ensure_partitions(db.ctx)
    assert "usage_partition_creation_blocked" in caplog.text
    async with db.engine.connect() as connection:
        assert (
            await connection.scalar(text("SELECT count(*) FROM usage_logs_default"))
            == 1
        )


async def test_retention_uses_bounds_and_keeps_default(
    partition_db, monkeypatch, caplog
):
    db = partition_db
    base = month_start(datetime.now(timezone.utc))
    # cutoff = base+3; legacy upper=base+1; monthly ends base+2, +3, +4.
    monkeypatch.setattr(partitions, "utc_now", lambda: add_months(base, 15))
    monkeypatch.setattr(
        partitions, "get_settings", lambda: SimpleNamespace(usage_retention_months=12)
    )
    with caplog.at_level("INFO"):
        await partitions.drop_expired_partitions(db.ctx)
    names = await partition_names(db.engine)
    assert "usage_logs_legacy" not in names
    assert f"usage_logs_{add_months(base, 1):%Y_%m}" not in names
    assert (
        f"usage_logs_{add_months(base, 2):%Y_%m}" in names
    )  # Upper equals cutoff: retain.
    assert f"usage_logs_{add_months(base, 3):%Y_%m}" in names
    assert "usage_logs_default" in names
    assert "usage_partition_dropped" in caplog.text
    await partitions.drop_expired_partitions(db.ctx)
    assert await partition_names(db.engine) == names
    failure = db.migrate("downgrade", OLD, success=False)
    assert "retention already removed usage_logs_legacy" in failure.stderr


async def test_report_partition_pruning(partition_db):
    db = partition_db
    base = month_start(datetime.now(timezone.utc))
    end = add_months(base, 2)
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.execute.return_value.all = Mock(return_value=[])
    await usage_report(
        UserContext(uuid4(), uuid4(), "viewer"),
        Mock(return_value=session),
        end - timedelta(days=30),
        end,
    )
    statement = session.execute.call_args.args[0]
    sql = str(
        statement.compile(
            dialect=db.engine.dialect, compile_kwargs={"literal_binds": True}
        )
    )
    async with db.engine.connect() as connection:
        plan = (
            await connection.execute(text("EXPLAIN (FORMAT JSON) " + sql))
        ).scalar_one()
    rendered = json.dumps(plan)
    assert f"usage_logs_{add_months(base, 1):%Y_%m}" in rendered
    assert f"usage_logs_{add_months(base, 2):%Y_%m}" not in rendered
    assert f"usage_logs_{add_months(base, 3):%Y_%m}" not in rendered
    assert "usage_logs_default" not in rendered


async def test_worker_startup_ensures_partitions(partition_db, monkeypatch):
    from llm_gateway import worker

    db = partition_db
    now = add_months(month_start(datetime.now(timezone.utc)), 6)
    monkeypatch.setattr(partitions, "utc_now", lambda: now)
    monkeypatch.setattr(worker, "create_async_engine", lambda url: db.engine)
    ctx = {}
    await worker.on_startup(ctx)
    assert {
        f"usage_logs_{add_months(now, n):%Y_%m}" for n in range(3)
    } <= await partition_names(db.engine)
    await worker.on_shutdown(ctx)
