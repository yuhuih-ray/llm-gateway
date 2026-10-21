import logging
from datetime import timezone
from typing import Any

from arq import Retry, cron
from arq.connections import RedisSettings
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import (
    DBAPIError,
    DisconnectionError,
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_gateway.budgets import reconcile_budgets
from llm_gateway.config import get_settings
from llm_gateway.models import UsageLog
from llm_gateway.partitions import drop_expired_partitions, ensure_partitions
from llm_gateway.usage import UsageRecord

logger = logging.getLogger(__name__)


async def on_startup(ctx: dict[str, Any]) -> None:  # ARQ worker-owned engine.
    engine = create_async_engine(get_settings().database_url)
    ctx["engine"] = engine
    ctx["sessions"] = async_sessionmaker(engine)
    try:
        await ensure_partitions(ctx)
        await reconcile_budgets(ctx)
    except BaseException:
        await engine.dispose()
        raise


async def on_shutdown(ctx: dict[str, Any]) -> None:  # Dispose worker connections.
    await ctx["engine"].dispose()


async def write_usage(
    ctx: dict[str, Any],  # ARQ context includes job_try and worker session factory.
    payload: dict[str, Any],  # Gateway-computed cost; never recompute in the worker.
) -> None:
    await write_usage_batch(ctx, [payload])


async def write_usage_batch(
    ctx: dict[str, Any],  # Worker-owned session factory and retry number.
    payloads: list[dict[str, Any]],  # One bounded gateway batch.
) -> None:
    records = [UsageRecord.model_validate(payload) for payload in payloads]
    if not records:
        return
    try:
        async with ctx["sessions"]() as session:
            async with session.begin():
                await session.execute(
                    insert(UsageLog)
                    .values([record.model_dump() for record in records])
                    .on_conflict_do_nothing(index_elements=["request_id", "created_at"])
                )
    except (SQLAlchemyError, OSError) as exc:
        transient = isinstance(
            exc, (OperationalError, DisconnectionError, OSError)
        ) or (isinstance(exc, DBAPIError) and exc.connection_invalidated)
        if (
            transient
            and not isinstance(exc, IntegrityError)
            and ctx.get("job_try", 1) < 5
        ):
            raise Retry(defer=min(30, 2 ** (ctx.get("job_try", 1) - 1))) from None
        for record in records:
            logger.error("usage_record_failed %s", record.model_dump_json())
        raise


class WorkerSettings:
    functions = [write_usage, write_usage_batch]
    cron_jobs = [
        cron(reconcile_budgets, minute=0),
        cron(ensure_partitions, hour=0, minute=5),
        cron(drop_expired_partitions, hour=0, minute=15),
    ]
    on_startup = on_startup
    on_shutdown = on_shutdown
    timezone = timezone.utc
    max_tries = 5
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
