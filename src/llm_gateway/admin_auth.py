import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Callable, Coroutine, Literal
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_gateway.config import get_settings
from llm_gateway.db import get_sessionmaker
from llm_gateway.models import User

logger = logging.getLogger(__name__)
Role = Literal["viewer", "member", "admin"]
ROLE_LEVEL = {"viewer": 0, "member": 1, "admin": 2}
password_hasher = PasswordHasher()
# API keys have high random entropy, so SHA-256 suffices. Human passwords need
# Argon2id's memory and CPU cost to resist offline guessing.
# Fixed for this process; no account corresponds to this random dummy password.
DUMMY_HASH = password_hasher.hash(secrets.token_urlsafe(32))
Sessions = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)]


def hash_password(password: str) -> str:  # Plaintext must never be logged.
    return password_hasher.hash(password)


def verify_password(hashed: str, password: str) -> bool:  # Includes dummy verification.
    try:
        return password_hasher.verify(hashed, password)
    except (VerificationError, InvalidHashError):
        return False


def jwt_secret() -> str:
    value = get_settings().jwt_secret
    if value is None or len(value.get_secret_value().encode("utf-8")) < 32:
        logger.error(
            "Admin authentication unavailable: JWT_SECRET must contain at least 32 bytes"
        )
        raise HTTPException(503, "Admin authentication is not configured")
    return value.get_secret_value()


Secret = Annotated[str, Depends(jwt_secret)]


def unauthorized(message: str = "Invalid user token") -> HTTPException:
    return HTTPException(401, message, headers={"WWW-Authenticate": "Bearer"})


def issue_token(
    user: User, secret: str
) -> str:  # Secret is validated by the dependency.
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": str(user.id),
            "tenant_id": str(user.tenant_id),
            "role": user.role,
            "iat": now,
            "exp": now + timedelta(minutes=15),
            "iss": "llm-gateway",
            "aud": "llm-gateway-admin",
        },
        secret,
        algorithm="HS256",
    )


@dataclass(frozen=True)
class UserContext:
    user_id: UUID
    tenant_id: UUID
    role: str


async def authenticate_user(
    secret: Secret,
    sessions: Sessions,
    authorization: Annotated[str | None, Header()] = None,  # Never log credentials.
) -> UserContext:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token or token.startswith("gw_"):
        raise unauthorized()
    try:
        # Pinning prevents alg:none and algorithm-confusion attacks.
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            issuer="llm-gateway",
            audience="llm-gateway-admin",
            options={
                "require": ["sub", "tenant_id", "role", "iat", "exp", "iss", "aud"]
            },
        )
        user_id, tenant_id = UUID(claims["sub"]), UUID(claims["tenant_id"])
    except (jwt.InvalidTokenError, ValueError, TypeError, AttributeError):
        raise unauthorized() from None
    async with sessions() as session:
        user = await session.scalar(
            select(User).where(User.id == user_id, User.tenant_id == tenant_id)
        )
    if user is None or user.disabled_at is not None:
        raise unauthorized()
    # Demoted/disabled users retain cryptographically valid tokens until expiry;
    # authorization must always use the current database role and status.
    return UserContext(user.id, user.tenant_id, user.role)


def require_role(
    minimum: Role,  # Inclusive viewer < member < admin hierarchy.
) -> Callable[..., Coroutine[Any, Any, UserContext]]:
    async def check(
        user: Annotated[UserContext, Depends(authenticate_user)],
    ) -> UserContext:
        if ROLE_LEVEL.get(user.role, -1) < ROLE_LEVEL[minimum]:
            raise HTTPException(403, "Insufficient permissions")
        return user

    return check
