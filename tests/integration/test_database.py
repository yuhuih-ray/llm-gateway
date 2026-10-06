from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import delete, insert, inspect
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from llm_gateway.models import ApiKey, Base, Tenant, UsageLog

pytestmark = pytest.mark.anyio


async def seed(connection):  # Connection in the test's transaction.
    tenant_id = await connection.scalar(
        insert(Tenant).values(name="tenant").returning(Tenant.id)
    )
    key_id = await connection.scalar(
        insert(ApiKey)
        .values(
            tenant_id=tenant_id, name="key", key_prefix="prefix", key_hash="test-hash"
        )
        .returning(ApiKey.id)
    )
    return dict(
        tenant_id=tenant_id,
        api_key_id=key_id,
        request_id=uuid4(),
        model_requested="fake",
        status="success",
    )


async def test_migration_round_trip(engine, migrate):  # Isolated container resources.
    migrate("downgrade", "base")
    async with engine.connect() as connection:
        tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
        assert not {"tenants", "api_keys", "usage_logs", "users"} & set(tables)
    migrate("upgrade", "head")
    async with engine.connect() as connection:
        tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
        assert {"tenants", "api_keys", "usage_logs", "users"} <= set(tables)


async def test_schema_matches_models(engine):  # Migrated container engine.
    async with engine.connect() as connection:
        differences = await connection.run_sync(
            lambda conn: compare_metadata(
                MigrationContext.configure(conn, opts={"compare_server_default": True}),
                Base.metadata,
            )
        )
    assert differences == []


async def test_tenant_server_defaults(engine):  # Migrated container engine.
    async with engine.begin() as connection:
        row = (
            await connection.execute(
                insert(Tenant)
                .values(name="defaults")
                .returning(Tenant.id, Tenant.status, Tenant.created_at)
            )
        ).one()
        assert row.id is not None
        assert row.status == "active"
        assert row.created_at is not None


async def test_tenant_delete_restricted(engine):  # Migrated container engine.
    with pytest.raises(IntegrityError, match="fk_api_keys_tenant_id_tenants"):
        async with engine.begin() as connection:
            values = await seed(connection)
            await connection.execute(
                delete(Tenant).where(Tenant.id == values["tenant_id"])
            )


async def test_usage_log_missing_tenant(engine):  # Migrated container engine.
    with pytest.raises(IntegrityError, match="fk_usage_logs_tenant_id_tenants"):
        async with engine.begin() as connection:
            values = await seed(connection)
            values["tenant_id"] = uuid4()
            await connection.execute(insert(UsageLog).values(**values))


async def test_duplicate_request_id(engine):  # Migrated container engine.
    with pytest.raises(IntegrityError, match="uq_usage_logs_request_id"):
        async with engine.begin() as connection:
            values = await seed(connection)
            await connection.execute(insert(UsageLog).values(**values))
            await connection.execute(insert(UsageLog).values(**values))


async def test_duplicate_request_id_do_nothing(engine):  # Migrated container engine.
    async with engine.begin() as connection:
        values = await seed(connection)
        await connection.execute(insert(UsageLog).values(**values))
        result = await connection.execute(
            pg_insert(UsageLog)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["request_id"])
            .returning(UsageLog.id)
        )
        assert result.all() == []


async def test_invalid_tenant_status(engine):  # Migrated container engine.
    with pytest.raises(IntegrityError, match="ck_tenants_status"):
        async with engine.begin() as connection:
            await connection.execute(
                insert(Tenant).values(name="invalid", status="deleted")
            )


async def test_usage_report_index_is_valid(engine):
    from sqlalchemy import text

    async with engine.connect() as connection:
        valid = await connection.scalar(
            text("""
            SELECT indisvalid FROM pg_index
            WHERE indexrelid = 'ix_usage_logs_tenant_id_created_at'::regclass
        """)
        )
    assert valid is True


async def test_usage_report_can_use_tenant_time_index(engine):
    from datetime import datetime, timedelta, timezone
    from unittest.mock import AsyncMock, Mock

    from sqlalchemy import text

    from llm_gateway.admin import usage_report
    from llm_gateway.admin_auth import UserContext

    async with engine.begin() as connection:
        values = await seed(connection)
        await connection.execute(insert(UsageLog).values(**values))
        # Capture the endpoint's actual statement rather than maintain a query copy.
        session = AsyncMock()
        session.__aenter__.return_value = session
        session.execute.return_value.all = Mock(return_value=[])
        end = datetime.now(timezone.utc)
        await usage_report(
            UserContext(uuid4(), values["tenant_id"], "viewer"),
            Mock(return_value=session),
            end - timedelta(days=30),
            end,
        )
        statement = session.execute.call_args.args[0]
        sql = str(
            statement.compile(
                dialect=engine.dialect, compile_kwargs={"literal_binds": True}
            )
        )
        # Tiny fixtures normally favor a seq scan; this checks index eligibility.
        # LOCAL keeps the planner setting confined to this test transaction.
        await connection.execute(text("SET LOCAL enable_seqscan = off"))
        plan = (await connection.execute(text("EXPLAIN " + sql))).scalars().all()
        assert "ix_usage_logs_tenant_id_created_at" in "\n".join(plan)
