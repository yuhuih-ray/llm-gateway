import asyncio
import logging
import math
import threading
from dataclasses import dataclass
from time import monotonic
from uuid import UUID

from fastapi.responses import JSONResponse

from llm_gateway.config import get_settings

logger = logging.getLogger(__name__)

# Redis executes one script at a time, so this all-or-nothing decision is atomic.
# Separate GET/SET calls from multiple gateway processes would race.
LUA = """
-- Redis TIME gives all gateway instances the same clock, avoiding host clock skew.
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local tokens, ttls = {}, {}
local allowed, retry_ms = 1, 0
for i, key in ipairs(KEYS) do
    local offset = (i - 1) * 3
    local capacity = tonumber(ARGV[offset + 1])
    local refill = tonumber(ARGV[offset + 2])
    local cost = tonumber(ARGV[offset + 3])
    local stored = redis.call('HMGET', key, 'tokens', 'updated_at')
    local previous = tonumber(stored[1]) or capacity
    local updated = tonumber(stored[2]) or now
    tokens[i] = math.min(capacity, previous + math.max(0, now - updated) * refill)
    ttls[i] = math.max(1, math.ceil(2000 * capacity / refill))
    if tokens[i] < cost then
        allowed = 0
        retry_ms = math.max(retry_ms, math.ceil((cost - tokens[i]) / refill * 1000))
    end
end
local remaining = math.huge
for i, key in ipairs(KEYS) do
    if allowed == 1 then tokens[i] = tokens[i] - tonumber(ARGV[(i - 1) * 3 + 3]) end
    remaining = math.min(remaining, tokens[i])
    redis.call('HSET', key, 'tokens', tokens[i], 'updated_at', now)
    redis.call('PEXPIRE', key, ttls[i])
end
return {allowed, tostring(remaining), retry_ms}
"""


@dataclass(frozen=True)
class Bucket:
    key: str
    capacity: float
    refill: float
    cost: float = 1

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(v) and v > 0
                for v in (self.capacity, self.refill, self.cost)
            )
            or self.cost > self.capacity
        ):
            raise ValueError(
                "Bucket values must be positive, finite, and cost <= capacity"
            )


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: float
    retry_after_ms: int


class RateLimiter:
    def __init__(self) -> None:
        self.buckets: dict[str, tuple[float, float, float]] = {}
        self.lock = threading.Lock()
        self.last_warning = -math.inf
        self.last_cleanup = -math.inf

    def fallback(
        self, buckets: list[Bucket]
    ) -> Decision:  # No await within the decision.
        now = monotonic()
        with self.lock:
            # Redis is non-critical for serving (as in readiness), but an outage
            # must not remove limits entirely. N processes can allow N times the
            # configured rate in this degraded, independently bounded mode.
            if now - self.last_warning >= 30:
                logger.warning(
                    "Rate limit Redis unavailable; using bounded in-process fallback"
                )
                self.last_warning = now
            if now - self.last_cleanup >= 30:
                self.buckets = {k: v for k, v in self.buckets.items() if v[2] > now}
                self.last_cleanup = now
            tokens = []
            retry_ms = 0
            for bucket in buckets:
                previous, updated, expires = self.buckets.get(
                    bucket.key, (bucket.capacity, now, now)
                )
                if expires <= now:
                    previous, updated = bucket.capacity, now
                value = min(
                    bucket.capacity, previous + max(0, now - updated) * bucket.refill
                )
                tokens.append(value)
                if value < bucket.cost:
                    retry_ms = max(
                        retry_ms,
                        math.ceil((bucket.cost - value) / bucket.refill * 1000),
                    )
            allowed = retry_ms == 0
            for bucket, value in zip(buckets, tokens):
                self.buckets[bucket.key] = (
                    value - bucket.cost if allowed else value,
                    now,
                    now + 2 * bucket.capacity / bucket.refill,
                )
            return Decision(
                allowed, min(self.buckets[b.key][0] for b in buckets), retry_ms
            )

    async def check(
        self, pool, buckets: list[Bucket]
    ) -> Decision:  # Lifespan-owned Redis client.
        if not buckets or len({b.key for b in buckets}) != len(buckets):
            raise ValueError("Expected distinct bucket keys")
        try:
            async with asyncio.timeout(0.1):
                if pool is None:
                    raise ConnectionError("Redis pool missing")
                result = await pool.eval(
                    LUA,
                    len(buckets),
                    *(b.key for b in buckets),
                    *(
                        value
                        for b in buckets
                        for value in (b.capacity, b.refill, b.cost)
                    ),
                )
                return Decision(bool(int(result[0])), float(result[1]), int(result[2]))
        except Exception:
            return self.fallback(buckets)


def request_buckets(tenant_id: UUID, provider: str) -> list[Bucket]:
    settings = get_settings()
    # TODO: per-tenant overrides are intentionally deferred.
    buckets = [
        Bucket(
            f"ratelimit:tenant:{tenant_id}",
            settings.tenant_request_capacity,
            settings.tenant_request_refill,
        )
    ]
    if provider == "gemini":
        buckets.append(
            Bucket(
                "ratelimit:provider:gemini",
                settings.gemini_request_capacity,
                settings.gemini_request_refill,
            )
        )
    return buckets


def rejected(decision: Decision) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={
            "Retry-After": str(max(1, math.ceil(decision.retry_after_ms / 1000))),
            "X-RateLimit-Remaining": str(max(0, math.floor(decision.remaining))),
        },
        content={
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        },
    )
