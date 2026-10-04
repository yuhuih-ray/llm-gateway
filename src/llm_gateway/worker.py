import logging
from typing import Any

from arq import Retry
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

from llm_gateway.config import get_settings
from llm_gateway.models import UsageLog
from llm_gateway.usage import UsageRecord

logger = logging.getLogger(__name__)


async def on_startup(ctx: dict[str, Any]) -> None:  # ARQ worker-owned engine.
    engine = create_async_engine(get_settings().database_url)
    ctx["engine"] = engine
    ctx["sessions"] = async_sessionmaker(engine)


async def on_shutdown(ctx: dict[str, Any]) -> None:  # Dispose worker connections.
    await ctx["engine"].dispose()


async def write_usage(
    ctx: dict[str, Any],  # ARQ context includes job_try and worker session factory.
    payload: dict[str, Any],  # Gateway-computed cost; never recompute in the worker.
) -> None:
    record = UsageRecord.model_validate(payload)
    try:
        async with ctx["sessions"]() as session:
            async with session.begin():
                await session.execute(
                    insert(UsageLog)
                    .values(**record.model_dump())
                    .on_conflict_do_nothing(index_elements=["request_id"])
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
        logger.error("usage_record_failed %s", record.model_dump_json())
        raise


class WorkerSettings:
    functions = [write_usage]
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_tries = 5
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
