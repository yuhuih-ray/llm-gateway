import hashlib
import hmac
import logging
import re
import secrets
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_gateway.db import get_session
from llm_gateway.models import ApiKey, Tenant

logger = logging.getLogger(__name__)


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


async def authenticate(
    session: Annotated[AsyncSession, Depends(get_session)],  # Request-scoped session.
    authorization: Annotated[str | None, Header()] = None,  # Raw header; never logged.
) -> AuthContext:
    if authorization is None:
        logger.warning("API key rejected: missing header; prefix=none")
        raise AuthenticationError("Missing API key")
    scheme, separator, key = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        logger.warning("API key rejected: invalid scheme; prefix=none")
        raise AuthenticationError("Invalid API key")
    try:
        prefix = parse_key(key)
    except ValueError:
        logger.warning("API key rejected: malformed key; prefix=none")
        raise AuthenticationError("Invalid API key") from None
    row = (
        await session.execute(
            select(ApiKey, Tenant)
            .join(Tenant, ApiKey.tenant_id == Tenant.id)
            .where(ApiKey.key_prefix == prefix)
        )
    ).one_or_none()
    if row is None:
        logger.warning("API key rejected: unknown prefix; prefix=%s", prefix)
        raise AuthenticationError("Invalid API key")
    api_key, tenant = row
    if not hmac.compare_digest(api_key.key_hash, hash_key(key)):
        logger.warning("API key rejected: hash mismatch; prefix=%s", prefix)
        raise AuthenticationError("Invalid API key")
    if api_key.revoked_at is not None:
        logger.warning("API key rejected: revoked key; prefix=%s", prefix)
        raise AuthenticationError("Invalid API key")
    if tenant.status != "active":
        logger.warning("API key rejected: suspended tenant; prefix=%s", prefix)
        raise AuthenticationError("Tenant is suspended", 403)
    return AuthContext(tenant_id=tenant.id, api_key_id=api_key.id)
