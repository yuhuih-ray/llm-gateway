"""UTC month partition maintenance; no row movement or default-partition deletion."""

import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from llm_gateway.config import get_settings

logger = logging.getLogger(__name__)
CHILD = re.compile(r"usage_logs_(?:legacy|default|\d{4}_\d{2})\Z")
MONTHLY = re.compile(r"usage_logs_\d{4}_\d{2}\Z")
LOCK = 784625103


def month_start(now: datetime) -> datetime:  # All partition bounds use UTC.
    return now.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )


def add_months(start: datetime, count: int) -> datetime:
    year, month = divmod(start.year * 12 + start.month - 1 + count, 12)
    return start.replace(year=year, month=month + 1)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def upper_bound(bound: str) -> datetime | None:  # pg_get_expr of RANGE bounds.
    match = re.search(r"TO \('([^']+)'\)", bound)
    return datetime.fromisoformat(match[1]).astimezone(timezone.utc) if match else None


def partition_range(bound: str) -> tuple[datetime | None, datetime | None] | None:
    # DEFAULT is an overflow safety net, not a substitute for time-range coverage.
    if bound == "DEFAULT":
        return None
    match = re.fullmatch(
        r"FOR VALUES FROM \((MINVALUE|'[^']+')\) TO \((MAXVALUE|'[^']+')\)", bound
    )
    if match is None:
        raise ValueError(f"Unrecognized usage partition bound: {bound}")
    bounds = [
        None
        if value in {"MINVALUE", "MAXVALUE"}
        else datetime.fromisoformat(value.strip("'")).astimezone(timezone.utc)
        for value in match.groups()
    ]
    return bounds[0], bounds[1]


async def children(connection):  # Catalog data, not guessed table names.
    return (
        await connection.execute(
            text("""
        SELECT c.relname AS name, pg_get_expr(c.relpartbound,c.oid) AS bound
        FROM pg_inherits i JOIN pg_class c ON c.oid=i.inhrelid
        WHERE i.inhparent='public.usage_logs'::regclass
        ORDER BY c.relname
    """)
        )
    ).all()


async def ensure_partitions(ctx: dict[str, Any]) -> None:  # Worker-owned engine.
    start = month_start(utc_now())
    async with ctx["engine"].begin() as connection:
        await connection.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": LOCK}
        )
        rows = await children(connection)
        ranges = [
            (row.name, partition_range(row.bound))
            for row in rows
            if row.bound != "DEFAULT"
        ]
        count = await connection.scalar(
            text("SELECT count(*) FROM public.usage_logs_default")
        )
        if count:
            logger.error("usage_default_partition_not_empty count=%s", count)
        for offset in range(3):
            lower, upper = add_months(start, offset), add_months(start, offset + 1)
            name = f"usage_logs_{lower:%Y_%m}"
            # Legacy covers everything before the cutover month. Any existing
            # partition whose range contains the whole month satisfies coverage.
            covered = False
            overlaps = []
            for existing_name, bounds in ranges:
                assert bounds is not None
                begin, end = bounds
                if (begin is None or begin <= lower) and (end is None or end >= upper):
                    logger.debug(
                        "usage_partition_covered month=%s partition=%s",
                        name,
                        existing_name,
                    )
                    covered = True
                    break
                if (begin is None or begin < upper) and (end is None or end > lower):
                    overlaps.append(existing_name)
            if covered:
                continue
            if overlaps:
                # Do not attempt overlapping DDL on an unexpected partial-month range.
                logger.error(
                    "usage_partition_partial_coverage month=%s partitions=%s",
                    name,
                    overlaps,
                )
                raise RuntimeError(
                    f"Cannot create {name}: existing ranges partially overlap the month"
                )
            # Do not move or delete unexpected rows to make partition creation succeed.
            blocked = (
                await connection.scalar(
                    text("""
                SELECT EXISTS (SELECT 1 FROM public.usage_logs_default
                WHERE created_at >= :lower AND created_at < :upper)
            """),
                    {"lower": lower, "upper": upper},
                )
                if count
                else False
            )
            if blocked:
                logger.error("usage_partition_creation_blocked partition=%s", name)
                continue
            await connection.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS public.{name} PARTITION OF public.usage_logs "
                    f"FOR VALUES FROM ('{lower.isoformat()}') TO ('{upper.isoformat()}')"
                )
            )
            ranges.append((name, (lower, upper)))
            logger.info("usage_partition_created partition=%s", name)


async def drop_expired_partitions(ctx: dict[str, Any]) -> None:  # Never drops DEFAULT.
    cutoff = add_months(month_start(utc_now()), -get_settings().usage_retention_months)
    dropped = 0
    async with ctx["engine"].begin() as connection:
        await connection.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": LOCK}
        )
        for row in await children(connection):
            if row.name != "usage_logs_legacy" and not MONTHLY.fullmatch(row.name):
                continue
            upper = upper_bound(row.bound)
            # Strictly before cutoff; conservatively retain the boundary month.
            if upper is None or upper >= cutoff:
                continue
            await connection.execute(
                text(
                    f"ALTER TABLE public.usage_logs DETACH PARTITION public.{row.name}"
                )
            )
            await connection.execute(text(f"DROP TABLE public.{row.name}"))
            logger.info(
                "usage_partition_dropped partition=%s upper=%s cutoff=%s",
                row.name,
                upper,
                cutoff,
            )
            dropped += 1
    if not dropped:
        logger.debug("usage_retention_noop cutoff=%s", cutoff)
