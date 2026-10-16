import asyncio
from uuid import uuid4

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from testcontainers.redis import RedisContainer

from llm_gateway.ratelimit import Bucket, RateLimiter

pytestmark = pytest.mark.anyio


@pytest.fixture
async def redis_pool():
    with RedisContainer("redis:8") as redis:
        pool = await create_pool(
            RedisSettings(
                host=redis.get_container_host_ip(),
                port=int(redis.get_exposed_port(6379)),
            )
        )
        try:
            yield pool
        finally:
            await pool.aclose()


async def test_atomic_concurrent_buckets(redis_pool):
    limiter = RateLimiter()
    bucket = Bucket(str(uuid4()), 10, 0.001)
    decisions = await asyncio.gather(
        *(limiter.check(redis_pool, [bucket]) for _ in range(50))
    )
    assert sum(d.allowed for d in decisions) == 10
    assert not limiter.buckets  # This exercised Redis, not fallback.
    assert 0 < await redis_pool.pttl(bucket.key) <= 20000000


async def test_refill_and_retry_delay(redis_pool):
    limiter = RateLimiter()
    bucket = Bucket(str(uuid4()), 2, 20, cost=2)
    assert (await limiter.check(redis_pool, [bucket])).allowed
    denied = await limiter.check(redis_pool, [bucket])
    assert not denied.allowed
    assert 0 < denied.retry_after_ms <= 100
    assert abs(denied.retry_after_ms - (2 - denied.remaining) / 20 * 1000) <= 1
    await asyncio.sleep(0.12)
    again = await limiter.check(redis_pool, [bucket])
    assert again.allowed
    assert again.remaining == 0  # Refilled to capacity, then consumed two tokens.


async def test_all_or_nothing(redis_pool):
    limiter = RateLimiter()
    tenant = Bucket(str(uuid4()), 10, 0.001)
    provider = Bucket(str(uuid4()), 1, 0.001)
    assert (await limiter.check(redis_pool, [provider])).allowed
    assert not (await limiter.check(redis_pool, [tenant, provider])).allowed
    assert float(await redis_pool.hget(tenant.key, "tokens")) == 10
    assert not limiter.buckets
