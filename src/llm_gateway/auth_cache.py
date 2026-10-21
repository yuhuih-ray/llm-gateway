"""Short-lived authentication snapshots in the main, non-evicting Redis."""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis

from llm_gateway.config import get_settings

logger = logging.getLogger(__name__)
TTL = 60
TIMEOUT = 0.1
Action = Literal["revoke", "suspend", "clear"]


class CachedAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: UUID
    api_key_id: UUID
    tenant_status: str
    monthly_budget_usd: Decimal | None


def cache_key(key_hash: str) -> str:  # SHA-256 of the entire API key.
    # A prefix key would let a wrong secret with a known prefix pass authentication.
    # Store neither the full secret nor a prefix in the Redis key.
    return f"auth:{key_hash}"


async def read_auth(
    pool,  # Lifespan main Redis client.
    key_hash: str,  # Digest of the supplied full key; verification may still be pending.
) -> CachedAuth | Literal["revoked", "suspended"] | None:
    try:
        async with asyncio.timeout(TIMEOUT):
            value = await pool.get(cache_key(key_hash))
        if value is None:
            return None
        payload = json.loads(value)
        if payload == {"revoked": True}:
            return "revoked"
        if payload == {"revoked": True, "tenant_status": "suspended"}:
            return "suspended"
        return CachedAuth.model_validate(payload)
    except Exception:
        # Redis outages and malformed cached values always fall back to the DB.
        return None


async def fill_auth(
    pool,  # Same lifespan client as reads.
    key_hash: str,  # DB-verified full-key digest.
    value: CachedAuth,  # Detached DB snapshot; its session has closed.
) -> None:
    try:
        async with asyncio.timeout(TIMEOUT):
            # A lookup begun before revocation can finish after its commit. NX
            # prevents that delayed fill from replacing the revocation tombstone.
            await pool.set(
                cache_key(key_hash), value.model_dump_json(), ex=TTL, nx=True
            )
    except Exception:
        pass  # Cache availability must never determine authentication success.


async def invalidate_auth(
    pool,  # Existing main Redis client; callers have already committed and closed DB sessions.
    key_hashes: list[
        str
    ],  # All affected keys, collected inside the mutation transaction.
    action: Action,
) -> None:
    if not key_hashes:
        return
    try:
        async with asyncio.timeout(TIMEOUT):
            async with pool.pipeline(transaction=False) as pipe:
                for key_hash in key_hashes:
                    key = cache_key(key_hash)
                    if action == "clear":
                        pipe.delete(key)
                    else:
                        tombstone: dict[str, bool | str] = {"revoked": True}
                        if action == "suspend":
                            # Distinguish tenant suspension (403) from key revocation (401).
                            tombstone["tenant_status"] = "suspended"
                        pipe.set(key, json.dumps(tombstone), ex=TTL)
                await pipe.execute()
    except Exception:
        logger.error(
            "auth_cache_invalidation_failed action=%s count=%d; staleness bounded by 60s TTL",
            action,
            len(key_hashes),
        )


def create_cli_client() -> Redis:
    return Redis.from_url(get_settings().redis_url)


async def invalidate_cli_auth(key_hashes: list[str], action: Action) -> None:
    # CLI has no lifespan client. Create its Redis client only after DB work ends.
    if not key_hashes:
        return
    try:
        async with create_cli_client() as pool:
            await invalidate_auth(pool, key_hashes, action)
    except Exception:
        logger.error(
            "auth_cache_invalidation_failed action=%s count=%d; staleness bounded by 60s TTL",
            action,
            len(key_hashes),
        )
