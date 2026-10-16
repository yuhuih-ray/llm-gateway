import asyncio
import hashlib
import ipaddress
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from uuid import UUID

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_gateway.config import get_settings

logger = logging.getLogger(__name__)
MAX_LOCAL_BUCKETS = 10_000

# Redis executes one script at a time, so this all-or-nothing decision is atomic.
# Separate GET/SET calls from multiple gateway processes would race.
LUA = """
-- Redis TIME gives all gateway instances the same clock, avoiding host clock skew.
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local tokens, ttls = {}, {}
local allowed, retry_ms = 1, 0
for i, key in ipairs(KEYS) do
    local offset = (i - 1) * 4
    local capacity = tonumber(ARGV[offset + 1])
    local refill = tonumber(ARGV[offset + 2])
    local cost = tonumber(ARGV[offset + 3])
    local check_only = tonumber(ARGV[offset + 4]) == 1
    local stored = redis.call('HMGET', key, 'tokens', 'updated_at')
    local previous = tonumber(stored[1]) or capacity
    local updated = tonumber(stored[2]) or now
    tokens[i] = math.min(capacity, previous + math.max(0, now - updated) * refill)
    ttls[i] = math.max(1, math.ceil(2000 * capacity / refill))
    if (check_only and tokens[i] <= 0) or (not check_only and tokens[i] < cost) then
        allowed = 0
        retry_ms = math.max(retry_ms, check_only and (math.floor(-tokens[i] / refill * 1000) + 1) or math.ceil((cost - tokens[i]) / refill * 1000))
    end
end
local remaining = math.huge
for i, key in ipairs(KEYS) do
    if allowed == 1 and tonumber(ARGV[(i - 1) * 4 + 4]) == 0 then
        tokens[i] = tokens[i] - tonumber(ARGV[(i - 1) * 4 + 3])
    end
    remaining = math.min(remaining, tokens[i])
    redis.call('HSET', key, 'tokens', tokens[i], 'updated_at', now)
    redis.call('PEXPIRE', key, ttls[i])
end
return {allowed, tostring(remaining), retry_ms}
"""


CHARGE_LUA = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local capacity, refill, amount = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local stored = redis.call('HMGET', KEYS[1], 'tokens', 'updated_at')
local tokens = math.min(capacity, (tonumber(stored[1]) or capacity)
    + math.max(0, now - (tonumber(stored[2]) or now)) * refill)
tokens = math.max(-capacity, tokens - amount)
redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated_at', now)
redis.call('PEXPIRE', KEYS[1], math.max(1, math.ceil(2000 * capacity / refill)))
return tostring(tokens)
"""


REFUND_LUA = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local capacity, refill, amount = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local stored = redis.call('HMGET', KEYS[1], 'tokens', 'updated_at')
local tokens = math.min(capacity, (tonumber(stored[1]) or capacity)
    + math.max(0, now - (tonumber(stored[2]) or now)) * refill + amount)
redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated_at', now)
redis.call('PEXPIRE', KEYS[1], math.max(1, math.ceil(2000 * capacity / refill)))
return tostring(tokens)
"""


@dataclass(frozen=True)
class Bucket:
    key: str
    capacity: float
    refill: float
    cost: float = 1
    check_only: bool = False

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
        self.buckets: OrderedDict[str, tuple[float, float, float]] = OrderedDict()
        self.lock = threading.Lock()
        self.last_warning = -math.inf

    def fallback(
        self, buckets: list[Bucket], charge: float | None = None
    ) -> Decision:  # No await within the decision.
        with self.lock:
            now = monotonic()
            # Redis is non-critical for serving (as in readiness), but an outage
            # must not remove limits entirely. N processes can allow N times the
            # configured rate in this degraded, independently bounded mode.
            if now - self.last_warning >= 30:
                logger.warning(
                    "Rate limit Redis unavailable; using bounded in-process fallback"
                )
                self.last_warning = now
            # Without a bound, a long Redis outage plus many distinct keys
            # (e.g. rotating IPs) would grow process memory without limit.
            # Prune on access; expired buckets are fully refilled when recreated.
            # Debt needs up to twice the empty-to-full interval to refill.
            for key in [key for key, value in self.buckets.items() if value[2] <= now]:
                del self.buckets[key]
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
                if charge is None and (
                    value <= 0 if bucket.check_only else value < bucket.cost
                ):
                    retry_ms = max(
                        retry_ms,
                        (math.floor(-value / bucket.refill * 1000) + 1)
                        if bucket.check_only
                        else math.ceil((bucket.cost - value) / bucket.refill * 1000),
                    )
            allowed = retry_ms == 0
            remaining = math.inf
            for bucket, value in zip(buckets, tokens):
                if charge is not None:
                    value = min(bucket.capacity, max(-bucket.capacity, value - charge))
                elif allowed and not bucket.check_only:
                    value -= bucket.cost
                remaining = min(remaining, value)
                if (
                    bucket.key not in self.buckets
                    and len(self.buckets) >= MAX_LOCAL_BUCKETS
                ):
                    self.buckets.popitem(last=False)
                self.buckets[bucket.key] = (
                    value,
                    now,
                    now + max(bucket.capacity, bucket.capacity - value) / bucket.refill,
                )
                self.buckets.move_to_end(bucket.key)
            return Decision(allowed, remaining, retry_ms)

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
                        for value in (b.capacity, b.refill, b.cost, int(b.check_only))
                    ),
                )
                return Decision(bool(int(result[0])), float(result[1]), int(result[2]))
        except Exception:
            return self.fallback(buckets)

    async def charge(self, pool, bucket: Bucket, amount: float) -> None:
        # Debt bounds how long anomalous usage can block a tenant. Billing still
        # uses usage_logs, never this capped enforcement balance.
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Charge must be finite and nonnegative")
        try:
            async with asyncio.timeout(0.1):
                if pool is None:
                    raise ConnectionError("Redis pool missing")
                await pool.eval(
                    CHARGE_LUA, 1, bucket.key, bucket.capacity, bucket.refill, amount
                )
        except Exception:
            self.fallback([bucket], charge=amount)

    async def refund(self, pool, bucket: Bucket, amount: float) -> None:
        # Return a successful login's reservation; never exceed capacity.
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Refund must be finite and nonnegative")
        try:
            async with asyncio.timeout(0.1):
                if pool is None:
                    raise ConnectionError("Redis pool missing")
                await pool.eval(
                    REFUND_LUA, 1, bucket.key, bucket.capacity, bucket.refill, amount
                )
        except Exception:
            self.fallback([bucket], charge=-amount)


rate_limiter = RateLimiter()


def token_bucket(tenant_id: UUID) -> Bucket:
    settings = get_settings()
    return Bucket(
        f"ratelimit:tokens:{tenant_id}",
        settings.tenant_token_capacity,
        settings.tenant_token_refill,
        check_only=True,
    )


def client_ip(request: Request) -> str:
    trusted = [
        ipaddress.ip_network(value, strict=False)
        for value in get_settings().trusted_proxies
    ]

    def is_trusted(value: str) -> bool:
        try:
            address = ipaddress.ip_address(value)
            return any(address in network for network in trusted)
        except ValueError:
            return False

    peer = request.client.host if request.client else "unknown"
    # Blindly trusting X-Forwarded-For lets clients rotate spoofed IPs. With an
    # untrusted proxy setup every client instead shares that proxy's IP.
    if not is_trusted(peer):
        return peer
    for value in reversed(request.headers.get("x-forwarded-for", "").split(",")):
        value = value.strip()
        try:
            address = str(ipaddress.ip_address(value))
        except ValueError:
            return peer  # A malformed chain is not a trustworthy identity.
        if not is_trusted(address):
            return address
    return peer


def login_buckets(request: Request, email: str) -> list[Bucket]:
    # Per-email limits permit temporary victim lockout; modest limits recover
    # within minutes. Hash normalized emails so Redis keys contain no addresses.
    digest = hashlib.sha256(email.strip().lower().encode()).hexdigest()
    return [
        Bucket(f"ratelimit:login:ip:{client_ip(request)}", 10, 1 / 6),
        Bucket(f"ratelimit:login:email:{digest}", 5, 1 / 60),
    ]


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
    buckets.append(token_bucket(tenant_id))
    return buckets


def rejected(decision: Decision, *, admin: bool = False) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={
            "Retry-After": str(max(1, math.ceil(decision.retry_after_ms / 1000))),
            "X-RateLimit-Remaining": str(max(0, math.floor(decision.remaining))),
        },
        content={"detail": "Rate limit exceeded"}
        if admin
        else {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        },
    )
