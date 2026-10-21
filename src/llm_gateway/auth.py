import hashlib
import hmac
import logging
import re
import secrets
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_gateway.auth_cache import CachedAuth, fill_auth, read_auth
from llm_gateway.db import get_sessionmaker
from llm_gateway.models import ApiKey, Tenant
from llm_gateway.ratelimit import Bucket, Decision, RateLimiter, client_ip, rejected
from llm_gateway.timing import measure

logger = logging.getLogger(__name__)
failed_auth_limiter = RateLimiter()


def generate_key() -> str:
    return f"gw_{secrets.token_hex(4)}_{secrets.token_urlsafe(32)}"


def parse_key(key: str) -> str:  # Full bearer key; never log it.
    parts = key.split("_", 2)
    if (
        len(parts) != 3
        or parts[0] != "gw"
        or re.fullmatch(r"[0-9a-f]{8}", parts[1]) is None
        or re.fullmatch(r"[A-Za-z0-9_-]{43}", parts[2]) is None
    ):
        raise ValueError("Invalid API key")
    return f"gw_{parts[1]}"


def hash_key(key: str) -> str:  # Hash the entire key, including its prefix.
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class AuthContext:
    tenant_id: UUID
    api_key_id: UUID
    monthly_budget_usd: Decimal | None = None


class AuthenticationError(Exception):
    def __init__(
        self,
        message: str,  # Public error message.
        status_code: int = 401,  # Authentication or permission failure.
    ) -> None:
        self.message = message
        self.status_code = status_code


async def authentication_error_handler(
    request: Request,  # Required by FastAPI's exception handler interface.
    exc: Exception,  # Only sanitized messages are returned.
) -> JSONResponse:
    assert isinstance(exc, AuthenticationError)
    if isinstance(exc, FailedAuthLimit):
        return rejected(exc.decision)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": "authentication_error"
                if exc.status_code == 401
                else "permission_error",
            }
        },
        headers={"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None,
    )


class FailedAuthLimit(AuthenticationError):
    def __init__(
        self, decision: Decision
    ) -> None:  # Admission metadata for Retry-After.
        super().__init__("Rate limit exceeded", 429)
        self.decision = decision


def credentials(authorization: str | None) -> tuple[str, str]:
    if authorization is None:
        logger.warning("API key rejected: missing header; prefix=none")
        raise AuthenticationError("Missing API key")
    scheme, separator, key = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        logger.warning("API key rejected: invalid scheme; prefix=none")
        raise AuthenticationError("Invalid API key")
    try:
        return key, parse_key(key)
    except ValueError:
        logger.warning("API key rejected: malformed key; prefix=none")
        raise AuthenticationError("Invalid API key") from None


async def lookup_auth(
    session_factory: async_sessionmaker[AsyncSession],  # Session ends before Redis I/O.
    prefix: str,  # Indexed DB lookup only; never a cache identity.
    digest: str,  # Hash of the full supplied key, verified before caching.
) -> CachedAuth:
    async with session_factory() as session:
        row = (
            await session.execute(
                select(ApiKey, Tenant)
                .join(Tenant, ApiKey.tenant_id == Tenant.id)
                .where(ApiKey.key_prefix == prefix)
            )
        ).one_or_none()
    # Do not cache unknown prefixes or hash mismatches: random keys make negative
    # caching useless and fill memory. Only verified, existing keys can be cached.
    if row is None:
        logger.warning("API key rejected: unknown prefix; prefix=%s", prefix)
        raise AuthenticationError("Invalid API key")
    api_key, tenant = row
    if not hmac.compare_digest(api_key.key_hash, digest):
        logger.warning("API key rejected: hash mismatch; prefix=%s", prefix)
        raise AuthenticationError("Invalid API key")
    if api_key.revoked_at is not None:
        logger.warning("API key rejected: revoked key; prefix=%s", prefix)
        raise AuthenticationError("Invalid API key")
    return CachedAuth(
        tenant_id=tenant.id,
        api_key_id=api_key.id,
        tenant_status=tenant.status,
        monthly_budget_usd=tenant.monthly_budget_usd,
    )


def context(value: CachedAuth) -> AuthContext:
    if value.tenant_status != "active":
        raise AuthenticationError("Tenant is suspended", 403)
    return AuthContext(value.tenant_id, value.api_key_id, value.monthly_budget_usd)


async def authenticate(
    request: Request,  # Peer IP, lifespan Redis client, and optional timing state.
    session_factory: Annotated[
        async_sessionmaker[AsyncSession], Depends(get_sessionmaker)
    ],  # Factory keeps the lookup session independent of the request lifetime.
    authorization: Annotated[str | None, Header()] = None,  # Raw header; never logged.
) -> AuthContext:
    with measure(request, "auth"):
        pool = getattr(request.app.state, "usage_pool", None)
        invalid = None
        try:
            key, prefix = credentials(authorization)
        except AuthenticationError as exc:
            invalid = exc
        cached = None
        if invalid is None:
            digest = hash_key(key)
            cached = await read_auth(pool, digest)
            if isinstance(cached, CachedAuth):
                return context(cached)
            if cached == "suspended":
                raise AuthenticationError("Tenant is suspended", 403)
        bucket = Bucket(f"ratelimit:failed-auth:ip:{client_ip(request)}", 20, 1 / 3)
        # Reserve before DB work so concurrent invalid attempts cannot bypass the
        # limit. Refund every result except 401; valid cache hits need no DB gate.
        decision = await failed_auth_limiter.check(pool, [bucket])
        if not decision.allowed:
            raise FailedAuthLimit(decision)
        failed = False
        try:
            if invalid is not None:
                raise invalid
            if cached == "revoked":
                raise AuthenticationError("Invalid API key")
            value = await lookup_auth(session_factory, prefix, digest)
            await fill_auth(pool, digest, value)
            return context(value)
        except AuthenticationError as exc:
            failed = exc.status_code == 401
            raise
        finally:
            if not failed:
                await failed_auth_limiter.refund(pool, bucket, 1)
