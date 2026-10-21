"""Bounded, process-local accounting; no request waits on accounting I/O."""

import asyncio
import json
import logging
from collections import defaultdict, deque
from contextlib import suppress
from decimal import ROUND_CEILING, Decimal
from itertools import islice
from time import monotonic
from uuid import UUID, uuid4

from arq import create_pool
from arq.connections import RedisSettings
from starlette.datastructures import State

from llm_gateway.budgets import SCALE, TTL, spend_key
from llm_gateway.config import get_settings
from llm_gateway.ratelimit import token_bucket
from llm_gateway.usage import UsageRecord

logger = logging.getLogger(__name__)
BATCH_SIZE = 500
INTERVAL = 0.1
SHUTDOWN_TIMEOUT = 5.0

# One batch receipt makes retries safe even if Redis applied the script but its
# reply was lost. It has no expiry while pending and is deleted on delivery.
# A process crash can leave small orphan receipts; it cannot replay that buffer.
BATCH_LUA = """
local previous = redis.call('GET', KEYS[1])
if previous then return previous end
local batch = cjson.decode(ARGV[1])
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
for _, charge in ipairs(batch.tokens) do
    local key = KEYS[charge.key]
    local capacity, refill, amount = charge.capacity, charge.refill, charge.amount
    local stored = redis.call('HMGET', key, 'tokens', 'updated_at')
    local tokens = math.min(capacity, (tonumber(stored[1]) or capacity)
        + math.max(0, now - (tonumber(stored[2]) or now)) * refill)
    tokens = math.max(-capacity, tokens - amount)
    redis.call('HSET', key, 'tokens', tokens, 'updated_at', now)
    redis.call('PEXPIRE', key, math.max(1, math.ceil(2000 * capacity / refill)))
end
local warnings = {}
for index, spend in ipairs(batch.spend) do
    local key = KEYS[spend.key]
    redis.call('INCRBY', key, spend.amount)
    redis.call('EXPIRE', key, ARGV[2])
    -- GET and integer-string comparisons preserve money above 2**53.
    local total = redis.call('GET', key)
    local threshold = spend.threshold
    if threshold ~= cjson.null and
        (#total > #threshold or (#total == #threshold and total >= threshold)) then
        if redis.call('SET', KEYS[spend.warning_key], '1', 'NX', 'EX', ARGV[2]) then
            table.insert(warnings, index)
        end
    end
end
-- Replay warning decisions after a lost reply without reapplying any counters.
local result = #warnings == 0 and '[]' or cjson.encode(warnings)
redis.call('SET', KEYS[1], result)
return result
"""


class AccountingBuffer:
    def __init__(self, state: State) -> None:  # Lifespan-owned Redis client state.
        self.state = state
        self.capacity = get_settings().accounting_buffer_size
        self.records: deque[UsageRecord] = deque()
        self.dropped = 0
        self.wake = asyncio.Event()
        self.lock = asyncio.Lock()
        self.task: asyncio.Task[None] | None = None
        self.closing = False
        self.last_warning = -float("inf")
        self.batch: list[UsageRecord] = []
        self.batch_id = ""
        self.delivered = False
        self.counters_applied = False

    def record(self, record: UsageRecord) -> None:  # Append on the ASGI event loop.
        if len(self.records) >= self.capacity:
            self.dropped += 1
            logger.error("usage_record_dropped %s", record.model_dump_json())
            return
        self.records.append(record)
        if len(self.records) >= BATCH_SIZE:
            self.wake.set()

    def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="accounting-flusher")

    async def flush(self) -> None:
        """Deliver one batch; retain its records and identity on any failure."""
        async with self.lock:
            if not self.records:
                return
            if not self.batch:
                self.batch = list(islice(self.records, BATCH_SIZE))
                self.batch_id = str(uuid4())
                self.delivered = False
                self.counters_applied = False
            pool = self.state.usage_pool
            if pool is None:
                settings = RedisSettings.from_dsn(get_settings().redis_url)
                settings.conn_retries = 0
                pool = await create_pool(settings)
                self.state.usage_pool = pool
            tokens: dict[UUID, int] = defaultdict(int)
            spend: dict[str, int] = defaultdict(int)
            hints: dict[str, UsageRecord] = {}
            for record in self.batch:
                if (
                    not record.cache_hit
                    and record.prompt_tokens is not None
                    and record.completion_tokens is not None
                ):
                    tokens[record.tenant_id] += (
                        record.prompt_tokens + record.completion_tokens
                    )
                if record.cost is not None:
                    key = spend_key(record.tenant_id, record.created_at)
                    spend[key] += int(record.cost * SCALE)
                    hints[key] = record
            receipt = f"accounting:{self.batch_id}"
            # Apply counters before publishing rows so budget reconciliation
            # cannot observe this batch's DB rows before its Redis increments.
            if not self.counters_applied:
                keys = [receipt]
                charges = []
                spends = []
                for tenant, amount in tokens.items():
                    bucket = token_bucket(tenant)
                    keys.append(bucket.key)
                    charges.append(
                        dict(
                            key=len(keys),
                            capacity=bucket.capacity,
                            refill=bucket.refill,
                            amount=amount,
                        )
                    )
                for key, amount in spend.items():
                    hint = hints[key]
                    keys.extend([key, key.replace("budget:spent:", "budget:warned:")])
                    budget = hint.monthly_budget_usd
                    threshold = (
                        None
                        if budget is None
                        else str(
                            int(
                                (budget * SCALE * Decimal("0.8")).to_integral_value(
                                    rounding=ROUND_CEILING
                                )
                            )
                        )
                    )
                    spends.append(
                        dict(
                            key=len(keys) - 1,
                            warning_key=len(keys),
                            amount=str(amount),
                            threshold=threshold,
                        )
                    )
                warnings = await pool.eval(
                    BATCH_LUA,
                    len(keys),
                    *keys,
                    json.dumps(dict(tokens=charges, spend=spends)),
                    TTL,
                )
                self.counters_applied = True
                ordered_hints = list(hints.values())
                for index in json.loads(warnings):
                    hint = ordered_hints[index - 1]
                    logger.warning(
                        "monthly_budget_80_percent tenant_id=%s month=%s",
                        hint.tenant_id,
                        spend_key(hint.tenant_id, hint.created_at).rsplit(":", 1)[1],
                    )
            if not self.delivered:
                await pool.enqueue_job(
                    "write_usage_batch",
                    [record.model_dump(mode="json") for record in self.batch],
                    _job_id=self.batch_id,
                )
                self.delivered = True
            await pool.delete(receipt)
            for _ in self.batch:
                self.records.popleft()
            self.batch = []

    async def _run(self) -> None:
        backoff = INTERVAL
        while not self.closing or self.records:
            if not self.closing and len(self.records) < BATCH_SIZE:
                try:
                    await asyncio.wait_for(self.wake.wait(), INTERVAL)
                except TimeoutError:
                    pass
            self.wake.clear()
            try:
                # This timeout is only for background recovery, never a drop.
                async with asyncio.timeout(5):
                    await self.flush()
                backoff = INTERVAL
            except asyncio.CancelledError:
                raise
            except Exception:
                now = monotonic()
                if now - self.last_warning >= 30:
                    self.last_warning = now
                    logger.warning("Accounting Redis unavailable; retaining batch")
                await asyncio.sleep(backoff)
                backoff = min(2, backoff * 2)

    async def close(self) -> None:
        self.closing = True
        self.wake.set()
        if self.task is not None:
            try:
                async with asyncio.timeout(SHUTDOWN_TIMEOUT):
                    # Own cancellation explicitly after the drain deadline.
                    await asyncio.shield(self.task)
            except TimeoutError:
                self.task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.task
        if self.records:
            logger.error("accounting_shutdown_unflushed count=%d", len(self.records))


# A crash normally loses only the unflushed ~100 ms window, comparable to Redis
# AOF everysec. Outages or event-loop stalls can enlarge it up to the buffer cap.
# The request path never waits on accounting; enforcement is consequently delayed.
