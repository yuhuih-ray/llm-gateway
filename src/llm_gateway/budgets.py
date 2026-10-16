import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from time import monotonic
from uuid import UUID

from sqlalchemy import func, select

from llm_gateway.models import Tenant, UsageLog

logger = logging.getLogger(__name__)
SCALE = Decimal(100_000_000)
TTL = 40 * 24 * 60 * 60
last_warning = -float("inf")

# Compare nonnegative integer strings, not Lua doubles: money must remain exact
# even above 2**53. The script atomically preserves queued, not-yet-written spend.
RECONCILE_LUA = """
local current = redis.call('GET', KEYS[1]) or '0'
local target = ARGV[1]
if #target > #current or (#target == #current and target > current) then
    current = target
end
redis.call('SET', KEYS[1], current, 'EX', ARGV[2])
return current
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def month_start(when: datetime) -> datetime:
    return when.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )


def month_end(when: datetime) -> datetime:
    start = month_start(when)
    return start.replace(
        year=start.year + (start.month == 12), month=start.month % 12 + 1
    )


def spend_key(tenant_id: UUID, when: datetime) -> str:
    return f"budget:spent:{tenant_id}:{when.astimezone(timezone.utc):%Y-%m}"


def unavailable() -> None:
    global last_warning
    now = monotonic()
    if now - last_warning >= 30:
        last_warning = now
        logger.warning(
            "Budget Redis unavailable; allowing requests; usage_logs is billing truth"
        )


async def warn_threshold(
    pool, tenant_id: UUID, budget: Decimal | None, spent: int, when: datetime
) -> None:
    if budget is not None and Decimal(spent) >= budget * SCALE * Decimal("0.8"):
        flag = spend_key(tenant_id, when).replace("budget:spent:", "budget:warned:")
        if await pool.set(flag, "1", nx=True, ex=TTL):
            logger.warning(
                "monthly_budget_80_percent tenant_id=%s month=%s",
                tenant_id,
                when.strftime("%Y-%m"),
            )


async def read_spend(pool, tenant_id: UUID, when: datetime) -> int:
    async with asyncio.timeout(0.1):
        if pool is None:
            raise ConnectionError("Redis pool missing")
        return int(await pool.get(spend_key(tenant_id, when)) or 0)


async def budget_exceeded(pool, tenant_id: UUID, budget: Decimal | None) -> bool:
    if budget is None:
        return False
    # Redis is non-critical for serving; billing truth is in usage_logs.
    # In-flight overshoot <= in-flight requests * maximum cost per request,
    # bounded by the character input limit and the model's output cap.
    exceeded = False
    try:
        async with asyncio.timeout(0.1):
            when = utcnow()
            spent = await read_spend(pool, tenant_id, when)
            exceeded = Decimal(spent) >= budget * SCALE
            await warn_threshold(pool, tenant_id, budget, spent, when)
    except Exception:
        unavailable()
    return exceeded


async def add_spend(
    pool,  # Lifespan-owned Redis; no DB session is held.
    tenant_id: UUID,
    cost: Decimal | None,  # Gateway-computed, quantized to eight decimal places.
    when: datetime,  # Request start time selects the billing month.
    budget: Decimal | None,
) -> None:
    if cost is None:
        return
    try:
        async with asyncio.timeout(0.1):
            if pool is None:
                raise ConnectionError("Redis pool missing")
            # INCRBY, never INCRBYFLOAT: units are exactly 1e-8 USD.
            async with pool.pipeline(transaction=True) as pipeline:
                pipeline.incrby(spend_key(tenant_id, when), int(cost * SCALE))
                pipeline.expire(spend_key(tenant_id, when), TTL)
                spent, _ = await pipeline.execute()
            await warn_threshold(pool, tenant_id, budget, int(spent), when)
    except Exception:
        unavailable()  # Reconciliation repairs missed increments from usage_logs.


async def db_spend(sessions, tenant_id: UUID, when: datetime) -> Decimal:
    async with sessions() as session:
        return await session.scalar(
            select(func.coalesce(func.sum(UsageLog.cost), 0)).where(
                UsageLog.tenant_id == tenant_id,
                UsageLog.created_at >= month_start(when),
                UsageLog.created_at < month_end(when),
            )
        )


async def reconcile_budgets(ctx: dict) -> None:
    when = utcnow()
    async with ctx["sessions"]() as session:
        tenants = (
            await session.execute(
                select(Tenant.id, Tenant.monthly_budget_usd).where(
                    Tenant.monthly_budget_usd.is_not(None)
                )
            )
        ).all()
    for tenant_id, budget in tenants:
        total = await db_spend(ctx["sessions"], tenant_id, when)
        # Max, not overwrite: queued usage makes DB lag Redis. DB restores Redis
        # after data loss, without erasing spend that has not reached the DB yet.
        current = int(
            await ctx["redis"].eval(
                RECONCILE_LUA,
                1,
                spend_key(tenant_id, when),
                str(int(total * SCALE)),
                TTL,
            )
        )
        await warn_threshold(ctx["redis"], tenant_id, budget, current, when)
    logger.info("Budget reconciliation completed tenants=%s", len(tenants))
