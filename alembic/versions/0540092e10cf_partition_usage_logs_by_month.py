"""partition usage logs by month

Stop all ARQ workers before upgrade/downgrade and restart only after completion.
The no-copy cutover retains all pre-next-month rows in the legacy partition.
Concurrent preparation commits before attachment; if preparation fails, keep
workers stopped, inspect/drop INVALID indexes, and restore or finish preparation
before retrying. This revision requires an online PostgreSQL connection.

Revision ID: 0540092e10cf
Revises: 1d76484b0ba6
Create Date: 2026-10-05 22:28:40.116415

"""

from datetime import datetime, timezone

import sqlalchemy as sa

from alembic import op

revision = "0540092e10cf"
down_revision = "1d76484b0ba6"
branch_labels = None
depends_on = None


def month(start, offset):
    year, value = divmod(start.year * 12 + start.month - 1 + offset, 12)
    return start.replace(year=year, month=value + 1)


def upgrade():
    connection = op.get_bind()
    cutover = month(
        datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ),
        1,
    )
    # Validate before any rename/commit; future-dated rows require operator attention.
    if connection.scalar(
        sa.text("SELECT EXISTS(SELECT 1 FROM usage_logs WHERE created_at >= :cutover)"),
        {"cutover": cutover},
    ):
        raise RuntimeError(
            "Cannot partition: usage_logs has rows at/after next-month cutover"
        )
    op.execute("ALTER TABLE usage_logs RENAME TO usage_logs_legacy")
    names = (
        connection.execute(
            sa.text(
                "SELECT conname FROM pg_constraint WHERE conrelid='usage_logs_legacy'::regclass"
            )
        )
        .scalars()
        .all()
    )
    quote = connection.dialect.identifier_preparer.quote
    for name in names:
        renamed = name.replace("usage_logs", "usage_logs_legacy")
        if renamed != name:
            op.execute(
                f"ALTER TABLE usage_logs_legacy RENAME CONSTRAINT {quote(name)} TO {quote(renamed)}"
            )
    op.execute(
        "ALTER INDEX ix_usage_logs_tenant_id_created_at RENAME TO ix_usage_logs_legacy_tenant_id_created_at"
    )
    op.execute("ALTER SEQUENCE usage_logs_id_seq RENAME TO usage_logs_legacy_id_seq")
    op.execute(
        f"ALTER TABLE usage_logs_legacy ADD CONSTRAINT ck_usage_logs_legacy_cutover CHECK (created_at < '{cutover.isoformat()}'::timestamptz) NOT VALID"
    )
    # Commit the short exclusive-lock ADD before validation's weaker lock/scan.
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TABLE usage_logs_legacy VALIDATE CONSTRAINT ck_usage_logs_legacy_cutover"
        )
        op.create_index(
            "ux_usage_logs_legacy_request_time",
            "usage_logs_legacy",
            ["request_id", "created_at"],
            unique=True,
            postgresql_concurrently=True,
        )
        op.create_index(
            "ux_usage_logs_legacy_id_time",
            "usage_logs_legacy",
            ["id", "created_at"],
            unique=True,
            postgresql_concurrently=True,
        )
    maximum = connection.scalar(
        sa.text(
            "SELECT greatest(coalesce(max(id),0), (SELECT last_value FROM usage_logs_legacy_id_seq)) FROM usage_logs_legacy"
        )
    )
    # Attached partitions cannot own an identity; the parent generates every ID.
    # Dropping the identity drops its old sequence, not any existing row data.
    op.execute("ALTER TABLE usage_logs_legacy ALTER COLUMN id DROP IDENTITY")
    # CHECK names are table-local and must match for attachment (unlike indexes).
    op.execute(
        "ALTER TABLE usage_logs_legacy RENAME CONSTRAINT ck_usage_logs_legacy_status TO ck_usage_logs_status"
    )
    # PostgreSQL has no global partition index: both UNIQUE and PK include time.
    # Retries still dedupe because created_at is fixed in the gateway job payload.
    op.execute("ALTER TABLE usage_logs_legacy DROP CONSTRAINT pk_usage_logs_legacy")
    op.execute(
        "ALTER TABLE usage_logs_legacy DROP CONSTRAINT uq_usage_logs_legacy_request_id"
    )
    op.execute(
        "ALTER TABLE usage_logs_legacy ADD CONSTRAINT pk_usage_logs_legacy PRIMARY KEY USING INDEX ux_usage_logs_legacy_id_time"
    )
    op.execute(
        "ALTER TABLE usage_logs_legacy ADD CONSTRAINT uq_usage_logs_legacy_request_id UNIQUE USING INDEX ux_usage_logs_legacy_request_time"
    )
    op.execute("""
        CREATE TABLE usage_logs (
            id bigint GENERATED ALWAYS AS IDENTITY,
            request_id uuid NOT NULL,
            tenant_id uuid NOT NULL,
            api_key_id uuid NOT NULL,
            model_requested text NOT NULL,
            model_selected text,
            prompt_tokens integer,
            completion_tokens integer,
            cost numeric(14,8),
            latency_ms integer,
            status text NOT NULL,
            error_type text,
            router_info jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            upstream_model text,
            reasoning_tokens integer,
            ttft_ms integer,
            CONSTRAINT pk_usage_logs PRIMARY KEY (id, created_at),
            CONSTRAINT uq_usage_logs_request_id UNIQUE (request_id, created_at),
            CONSTRAINT fk_usage_logs_tenant_id_tenants FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT,
            CONSTRAINT fk_usage_logs_api_key_id_api_keys FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE RESTRICT,
            CONSTRAINT ck_usage_logs_status CHECK (status IN ('success','error','cancelled'))
        ) PARTITION BY RANGE (created_at)
    """)
    op.create_index(
        "ix_usage_logs_tenant_id_created_at", "usage_logs", ["tenant_id", "created_at"]
    )
    op.execute(
        f"ALTER TABLE usage_logs ATTACH PARTITION usage_logs_legacy FOR VALUES FROM (MINVALUE) TO ('{cutover.isoformat()}')"
    )
    # ATTACH TABLE reuses equivalent valid indexes; explicit attachment also
    # validates the parent when PostgreSQL already attached the matching index.
    for parent, child in [
        ("pk_usage_logs", "pk_usage_logs_legacy"),
        ("uq_usage_logs_request_id", "uq_usage_logs_legacy_request_id"),
        (
            "ix_usage_logs_tenant_id_created_at",
            "ix_usage_logs_legacy_tenant_id_created_at",
        ),
    ]:
        op.execute(f"ALTER INDEX {parent} ATTACH PARTITION {child}")
    for offset in range(3):
        lower, upper = month(cutover, offset), month(cutover, offset + 1)
        op.execute(
            f"CREATE TABLE usage_logs_{lower:%Y_%m} PARTITION OF usage_logs FOR VALUES FROM ('{lower.isoformat()}') TO ('{upper.isoformat()}')"
        )
    op.execute("CREATE TABLE usage_logs_default PARTITION OF usage_logs DEFAULT")
    connection.execute(
        sa.text("SELECT setval('usage_logs_id_seq', :value, true)"),
        {"value": max(1, maximum)},
    )


def downgrade():
    connection = op.get_bind()
    if connection.scalar(sa.text("SELECT to_regclass('usage_logs_legacy')")) is None:
        raise RuntimeError(
            "Cannot downgrade: retention already removed usage_logs_legacy"
        )
    # Lock writers during the preflight and restoration; workers must be stopped.
    op.execute("LOCK TABLE usage_logs IN ACCESS EXCLUSIVE MODE")
    if connection.scalar(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM usage_logs WHERE tableoid <> 'usage_logs_legacy'::regclass)"
        )
    ):
        raise RuntimeError(
            "Cannot downgrade without data loss: monthly/default partitions contain rows; archive or relocate them first"
        )
    for columns in ["id", "request_id"]:
        if connection.scalar(
            sa.text(
                f"SELECT EXISTS(SELECT 1 FROM usage_logs_legacy GROUP BY {columns} HAVING count(*)>1)"
            )
        ):
            raise RuntimeError(
                f"Cannot downgrade: duplicate {columns} values cannot satisfy the old schema"
            )
    maximum = connection.scalar(
        sa.text(
            "SELECT greatest(coalesce(max(id),0), (SELECT last_value FROM usage_logs_id_seq)) FROM usage_logs"
        )
    )
    op.execute("ALTER TABLE usage_logs DETACH PARTITION usage_logs_legacy")
    op.execute("DROP TABLE usage_logs")
    # Drop inherited/old constraints except NOT NULL, then restore the old schema.
    names = (
        connection.execute(
            sa.text(
                "SELECT conname FROM pg_constraint WHERE conrelid='usage_logs_legacy'::regclass AND contype <> 'n'"
            )
        )
        .scalars()
        .all()
    )
    quote = connection.dialect.identifier_preparer.quote
    for name in names:
        op.execute(f"ALTER TABLE usage_logs_legacy DROP CONSTRAINT {quote(name)}")
    op.execute("ALTER TABLE usage_logs_legacy RENAME TO usage_logs")
    op.execute(
        "ALTER INDEX ix_usage_logs_legacy_tenant_id_created_at RENAME TO ix_usage_logs_tenant_id_created_at"
    )
    op.execute("ALTER TABLE usage_logs ALTER COLUMN id DROP IDENTITY IF EXISTS")
    op.execute(
        "ALTER TABLE usage_logs ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY"
    )
    op.execute(
        "ALTER TABLE usage_logs ADD CONSTRAINT pk_usage_logs PRIMARY KEY(id), ADD CONSTRAINT uq_usage_logs_request_id UNIQUE(request_id), ADD CONSTRAINT fk_usage_logs_tenant_id_tenants FOREIGN KEY(tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT, ADD CONSTRAINT fk_usage_logs_api_key_id_api_keys FOREIGN KEY(api_key_id) REFERENCES api_keys(id) ON DELETE RESTRICT, ADD CONSTRAINT ck_usage_logs_status CHECK(status IN ('success','error','cancelled'))"
    )
    connection.execute(
        sa.text("SELECT setval('usage_logs_id_seq', :value, true)"),
        {"value": max(1, maximum)},
    )
