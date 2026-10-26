import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.models import ApiKey, Tenant, UsageLog

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.router_monitor.drift import current_rows


@pytest.mark.anyio
async def test_drift_query_filters_artifact_model_and_half_open_window(engine):
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    since = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
    until = since + timedelta(hours=1)
    async with sessions.begin() as session:
        tenant = Tenant(name="drift-test")
        session.add(tenant)
        await session.flush()
        key = ApiKey(
            tenant_id=tenant.id, name="test", key_prefix="drift", key_hash="a" * 64
        )
        session.add(key)
        await session.flush()
        for marker, model, run, created_at in (
            ("start", "auto", "current", since),
            ("inside", "auto", "current", since + timedelta(minutes=30)),
            ("end", "auto", "current", until),
            ("before", "auto", "current", since - timedelta(seconds=1)),
            ("other-run", "auto", "old", since),
            ("explicit", "local-qwen", "current", since),
        ):
            session.add(
                UsageLog(
                    request_id=uuid4(),
                    tenant_id=tenant.id,
                    api_key_id=key.id,
                    model_requested=model,
                    status="success",
                    created_at=created_at,
                    router_info=dict(artifact_run=run, marker=marker, oov_ratio=0.1),
                )
            )
    rows = await current_rows(sessions, "current", since, until)
    assert {row["marker"] for row in rows} == {"start", "inside"}
