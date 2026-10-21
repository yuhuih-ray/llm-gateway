"""Bounded, process-local accounting; no request waits on accounting I/O."""

import asyncio
import logging
from collections import defaultdict, deque
from itertools import islice
from time import monotonic
from uuid import UUID, uuid4

from arq import create_pool
from arq.connections import RedisSettings
from starlette.datastructures import State

from llm_gateway.budgets import SCALE, TTL, spend_key, warn_threshold
from llm_gateway.config import get_settings
from llm_gateway.ratelimit import CHARGE_LUA, token_bucket
from llm_gateway.usage import UsageRecord

logger = logging.getLogger(__name__)
BATCH_SIZE = 500
INTERVAL = 0.1
SHUTDOWN_TIMEOUT = 5.0

# Receipts make a retry safe even if Redis applied a command but its reply was
# lost. They have no expiry while a batch is pending, and are deleted on delivery.
# A process crash can leave small orphan receipts; it cannot replay that buffer.
TOKEN_LUA = (
    "if redis.call('EXISTS', KEYS[2]) == 1 then return 0 end\n"
    + CHARGE_LUA.replace(
        "return tostring(tokens)", "redis.call('SET', KEYS[2], '1')\nreturn 1"
    )
)
SPEND_LUA = """
local previous = redis.call('GET', KEYS[2])
if previous then return previous end
redis.call('INCRBY', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
-- GET preserves integer precision above 2**53, unlike Lua's INCRBY return.
local spent = redis.call('GET', KEYS[1])
redis.call('SET', KEYS[2], spent)
return spent
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
            receipts = []
            # Apply counters before publishing rows so budget reconciliation
            # cannot observe this batch's DB rows before its Redis increments.
            if not self.delivered:
                for tenant, amount in tokens.items():
                    bucket = token_bucket(tenant)
                    receipt = f"accounting:{self.batch_id}:{bucket.key}"
                    receipts.append(receipt)
                    await pool.eval(
                        TOKEN_LUA,
                        2,
                        bucket.key,
                        receipt,
                        bucket.capacity,
                        bucket.refill,
                        amount,
                    )
                for key, amount in spend.items():
                    receipt = f"accounting:{self.batch_id}:{key}"
                    receipts.append(receipt)
                    total = await pool.eval(SPEND_LUA, 2, key, receipt, amount, TTL)
                    hint = hints[key]
                    await warn_threshold(
                        pool,
                        hint.tenant_id,
                        hint.monthly_budget_usd,
                        int(total),
                        hint.created_at,
                    )
                await pool.enqueue_job(
                    "write_usage_batch",
                    [record.model_dump(mode="json") for record in self.batch],
                    _job_id=self.batch_id,
                )
                self.delivered = True
            else:
                receipts = [
                    f"accounting:{self.batch_id}:{token_bucket(tenant).key}"
                    for tenant in tokens
                ] + [f"accounting:{self.batch_id}:{key}" for key in spend]
            if receipts:
                await pool.delete(*receipts)
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
                    await self.task
            except TimeoutError:
                pass
        if self.records:
            logger.error("accounting_shutdown_unflushed count=%d", len(self.records))


# A crash normally loses only the unflushed ~100 ms window, comparable to Redis
# AOF everysec. Outages or event-loop stalls can enlarge it up to the buffer cap.
# The request path never waits on accounting; enforcement is consequently delayed.
