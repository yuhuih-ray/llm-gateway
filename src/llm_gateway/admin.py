from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from sqlalchemy import func, select, update
from starlette.concurrency import run_in_threadpool

from llm_gateway.admin_auth import (
    DUMMY_HASH,
    Secret,
    Sessions,
    UserContext,
    hash_password,
    issue_token,
    password_hasher,
    require_role,
    unauthorized,
    verify_password,
)
from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.models import ApiKey, UsageLog, User

router = APIRouter()
Viewer = Annotated[UserContext, Depends(require_role("viewer"))]
Member = Annotated[UserContext, Depends(require_role("member"))]
Admin = Annotated[UserContext, Depends(require_role("admin"))]


class LoginRequest(BaseModel):
    email: str = Field(min_length=1)
    password: SecretStr

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = 900


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    secret: Secret,
    sessions: Sessions,
    authorization: Annotated[str | None, Header()] = None,
) -> TokenResponse:
    # Login accepts passwords only, never API-key or bearer-token authentication.
    if authorization is not None:
        raise unauthorized()
    async with sessions() as session:
        user = await session.scalar(select(User).where(User.email == body.email))
    hashed = user.password_hash if user is not None else DUMMY_HASH
    password = body.password.get_secret_value()
    valid = await run_in_threadpool(verify_password, hashed, password)
    if user is None or not valid or user.disabled_at is not None:
        raise unauthorized("Invalid email or password")
    if password_hasher.check_needs_rehash(hashed):
        new_hash = await run_in_threadpool(hash_password, password)
        async with sessions.begin() as session:
            await session.execute(
                update(User)
                .where(
                    User.id == user.id,
                    User.tenant_id == user.tenant_id,
                    User.password_hash == hashed,
                )
                .values(password_hash=new_hash)
            )
    return TokenResponse(access_token=issue_token(user, secret))


class KeyInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    key_prefix: str
    created_at: datetime
    revoked_at: datetime | None


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1)


class CreatedKey(KeyInfo):
    key: str


@router.get("/admin/keys", response_model=list[KeyInfo])
async def list_keys(user: Viewer, sessions: Sessions) -> list[KeyInfo]:
    async with sessions() as session:
        keys = await session.scalars(
            select(ApiKey)
            .where(ApiKey.tenant_id == user.tenant_id)
            .order_by(ApiKey.created_at)
        )
        return [KeyInfo.model_validate(key) for key in keys]


@router.post("/admin/keys", response_model=CreatedKey)
async def create_key(
    body: CreateKeyRequest, user: Member, sessions: Sessions
) -> CreatedKey:
    key = generate_key()
    async with sessions.begin() as session:
        row = ApiKey(
            tenant_id=user.tenant_id,
            name=body.name,
            key_prefix=parse_key(key),
            key_hash=hash_key(key),
        )
        session.add(row)
        await session.flush()
        result = CreatedKey(**KeyInfo.model_validate(row).model_dump(), key=key)
    return result


@router.post("/admin/keys/{key_id}/revoke", response_model=KeyInfo)
async def revoke_key(key_id: UUID, user: Admin, sessions: Sessions) -> KeyInfo:
    async with sessions.begin() as session:
        row = await session.scalar(
            update(ApiKey)
            .where(ApiKey.id == key_id, ApiKey.tenant_id == user.tenant_id)
            .values(revoked_at=func.coalesce(ApiKey.revoked_at, func.now()))
            .returning(ApiKey)
        )
        if row is None:
            raise HTTPException(404, "Key not found")
        result = KeyInfo.model_validate(row)
    return result


class UsageTotals(BaseModel):
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: str = "0.00000000"


class ModelUsage(UsageTotals):
    model: str | None


class UsageReport(BaseModel):
    start: datetime
    end: datetime
    totals: UsageTotals
    models: list[ModelUsage]


@router.get("/admin/usage", response_model=UsageReport)
async def usage_report(
    user: Viewer,
    sessions: Sessions,
    start: datetime | None = None,  # Inclusive UTC timestamp.
    end: datetime | None = None,  # Exclusive UTC timestamp.
) -> UsageReport:
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(days=30)
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(422, "start and end must include a timezone")
    if not timedelta(0) < end - start <= timedelta(days=90):
        raise HTTPException(422, "Range must be positive and at most 90 days")
    async with sessions() as session:
        rows = (
            await session.execute(
                select(
                    UsageLog.model_selected,
                    func.count().label("requests"),
                    func.coalesce(func.sum(UsageLog.prompt_tokens), 0).label(
                        "prompt_tokens"
                    ),
                    func.coalesce(func.sum(UsageLog.completion_tokens), 0).label(
                        "completion_tokens"
                    ),
                    func.coalesce(func.sum(UsageLog.reasoning_tokens), 0).label(
                        "reasoning_tokens"
                    ),
                    func.coalesce(func.sum(UsageLog.cost), 0).label("cost"),
                )
                .where(
                    UsageLog.tenant_id == user.tenant_id,
                    UsageLog.created_at >= start,
                    UsageLog.created_at < end,
                )
                .group_by(UsageLog.model_selected)
                .order_by(UsageLog.model_selected)
            )
        ).all()
    models = [
        ModelUsage(
            model=row.model_selected,
            requests=row.requests,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            reasoning_tokens=row.reasoning_tokens,
            cost=format(row.cost, ".8f"),
        )
        for row in rows
    ]
    totals = UsageTotals(
        requests=sum(row.requests for row in models),
        prompt_tokens=sum(row.prompt_tokens for row in models),
        completion_tokens=sum(row.completion_tokens for row in models),
        reasoning_tokens=sum(row.reasoning_tokens for row in models),
        cost=format(sum((Decimal(row.cost) for row in models), Decimal(0)), ".8f"),
    )
    return UsageReport(start=start, end=end, totals=totals, models=models)
