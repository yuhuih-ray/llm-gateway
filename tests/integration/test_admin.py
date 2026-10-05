import argparse
import base64
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import httpx2
import jwt
import pytest
from argon2 import PasswordHasher
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway import admin, admin_auth, cli
from llm_gateway.admin_auth import hash_password, issue_token, verify_password
from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.config import Settings
from llm_gateway.db import get_sessionmaker
from llm_gateway.main import app
from llm_gateway.models import ApiKey, Tenant, UsageLog, User

pytestmark = pytest.mark.anyio
PASSWORD = "integration-test-password"
SECRET = "test-only-signing-secret-32-bytes-long"
PASSWORD_HASH = hash_password(PASSWORD)


@pytest.fixture
async def setup(engine, monkeypatch):
    monkeypatch.setattr(
        admin_auth,
        "get_settings",
        lambda: Settings(_env_file=None, jwt_secret=SECRET),
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions.begin() as session:
        tenants = [Tenant(name="a"), Tenant(name="b")]
        session.add_all(tenants)
        await session.flush()
        users = [
            User(
                tenant_id=t.id,
                email=f" {t.name.upper()}@Example.com ",
                password_hash=PASSWORD_HASH,
                role="admin",
            )
            for t in tenants
        ]
        keys = [
            ApiKey(
                tenant_id=t.id,
                name=t.name,
                key_prefix=parse_key(key),
                key_hash=hash_key(key),
            )
            for t, key in zip(tenants, [generate_key(), generate_key()])
        ]
        session.add_all(users + keys)
    app.dependency_overrides[get_sessionmaker] = lambda: sessions
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield SimpleNamespace(
                client=client,
                sessions=sessions,
                users=users,
                keys=keys,
                tenants=tenants,
            )
    finally:
        app.dependency_overrides.clear()


def headers(user):  # Tokens contain the role at issue time.
    return {"Authorization": f"Bearer {issue_token(user, SECRET)}"}


async def test_password_round_trip_and_login(setup):
    assert PASSWORD_HASH.startswith("$argon2id$")
    assert verify_password(PASSWORD_HASH, PASSWORD)
    assert not verify_password(PASSWORD_HASH, "wrong")
    response = await setup.client.post(
        "/auth/login", json={"email": " A@EXAMPLE.COM ", "password": PASSWORD}
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["expires_in"] == 900
    claims = jwt.decode(
        response.json()["access_token"],
        SECRET,
        algorithms=["HS256"],
        audience="llm-gateway-admin",
        issuer="llm-gateway",
    )
    assert claims["exp"] - claims["iat"] == 900
    assert claims["sub"] == str(setup.users[0].id)
    assert claims["tenant_id"] == str(setup.tenants[0].id)
    assert claims["role"] == "admin"


async def test_unknown_email_verifies_dummy_and_matches_wrong_password(
    setup, monkeypatch
):
    wrong = await setup.client.post(
        "/auth/login", json={"email": "a@example.com", "password": "wrong"}
    )
    hasher = Mock(wraps=admin_auth.password_hasher)
    monkeypatch.setattr(admin_auth, "password_hasher", hasher)
    unknown = await setup.client.post(
        "/auth/login", json={"email": "unknown@example.com", "password": "wrong"}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "Invalid email or password"}
    hasher.verify.assert_called_once_with(admin.DUMMY_HASH, "wrong")


async def test_login_rehashes_old_parameters(setup):
    old = PasswordHasher(time_cost=1).hash(PASSWORD)
    async with setup.sessions.begin() as session:
        await session.execute(
            update(User).where(User.id == setup.users[0].id).values(password_hash=old)
        )
    response = await setup.client.post(
        "/auth/login", json={"email": "a@example.com", "password": PASSWORD}
    )
    assert response.status_code == 200
    async with setup.sessions() as session:
        user = await session.get(User, setup.users[0].id)
    assert user.password_hash != old
    assert verify_password(user.password_hash, PASSWORD)
    assert not admin.password_hasher.check_needs_rehash(user.password_hash)


@pytest.mark.parametrize(
    "case", ["tampered", "expired", "none", "audience", "issuer", "missing_exp"]
)
async def test_invalid_jwt(setup, case):
    token = issue_token(setup.users[0], SECRET)
    claims = jwt.decode(token, options={"verify_signature": False})
    if case == "tampered":
        parts = token.split(".")
        claims["sub"] = str(uuid4())
        parts[1] = (
            base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        )
        token = ".".join(parts)
    elif case == "none":
        token = jwt.encode(claims, key="", algorithm="none")
    else:
        if case == "expired":
            claims["exp"] = claims["iat"] - 1
        elif case == "audience":
            claims["aud"] = "wrong"
        elif case == "issuer":
            claims["iss"] = "wrong"
        else:
            del claims["exp"]
        token = jwt.encode(claims, SECRET, algorithm="HS256")
    response = await setup.client.get(
        "/admin/keys", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize("case", ["disabled", "missing", "demoted", "tenant"])
async def test_database_authority_over_still_valid_token(setup, case):
    auth = headers(setup.users[0])
    async with setup.sessions.begin() as session:
        condition = User.id == setup.users[0].id
        if case == "missing":
            await session.execute(delete(User).where(condition))
        else:
            values = (
                {"disabled_at": datetime.now(timezone.utc)}
                if case == "disabled"
                else (
                    {"role": "viewer"}
                    if case == "demoted"
                    else {"tenant_id": setup.tenants[1].id}
                )
            )
            await session.execute(update(User).where(condition).values(**values))
    response = await setup.client.post(
        f"/admin/keys/{setup.keys[0].id}/revoke", headers=auth
    )
    assert response.status_code == (403 if case == "demoted" else 401)


@pytest.mark.parametrize("role", ["viewer", "member", "admin"])
@pytest.mark.parametrize("endpoint", ["login", "list", "create", "revoke", "usage"])
async def test_rbac_matrix(setup, role, endpoint):
    async with setup.sessions.begin() as session:
        await session.execute(
            update(User).where(User.id == setup.users[0].id).values(role=role)
        )
    # The stale token claims admin; permissions must still follow the DB matrix.
    auth = headers(setup.users[0])
    if endpoint == "login":
        response = await setup.client.post(
            "/auth/login", json={"email": "a@example.com", "password": PASSWORD}
        )
    elif endpoint == "list":
        response = await setup.client.get("/admin/keys", headers=auth)
    elif endpoint == "create":
        response = await setup.client.post(
            "/admin/keys", headers=auth, json={"name": "new"}
        )
    elif endpoint == "revoke":
        response = await setup.client.post(
            f"/admin/keys/{setup.keys[0].id}/revoke", headers=auth
        )
    else:
        response = await setup.client.get("/admin/usage", headers=auth)
    allowed = (
        endpoint in {"login", "list", "usage"}
        or role == "admin"
        or (role == "member" and endpoint == "create")
    )
    assert response.status_code == (200 if allowed else 403)


async def test_tenant_isolation_and_usage_precision(setup):
    now = datetime.now(timezone.utc)
    async with setup.sessions.begin() as session:
        for index, (tenant, key) in enumerate(zip(setup.tenants, setup.keys)):
            session.add(
                UsageLog(
                    request_id=uuid4(),
                    tenant_id=tenant.id,
                    api_key_id=key.id,
                    model_requested="fake",
                    model_selected="fake",
                    status="success",
                    prompt_tokens=10 + index,
                    completion_tokens=20,
                    reasoning_tokens=5,
                    cost=Decimal("0.12345678") if index == 0 else Decimal("99"),
                    created_at=now,
                )
            )
        session.add(
            UsageLog(
                request_id=uuid4(),
                tenant_id=setup.tenants[0].id,
                api_key_id=setup.keys[0].id,
                model_requested="fake",
                model_selected="fake",
                status="error",
                created_at=now - timedelta(days=40),
            )
        )
    auth = headers(setup.users[0])
    response = await setup.client.get("/admin/keys", headers=auth)
    assert [row["id"] for row in response.json()] == [str(setup.keys[0].id)]
    assert set(response.json()[0]) == {
        "id",
        "name",
        "key_prefix",
        "created_at",
        "revoked_at",
    }
    foreign = await setup.client.post(
        f"/admin/keys/{setup.keys[1].id}/revoke", headers=auth
    )
    missing = await setup.client.post(f"/admin/keys/{uuid4()}/revoke", headers=auth)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()
    response = await setup.client.get("/admin/usage", headers=auth)
    expected = dict(
        requests=1,
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=5,
        cost="0.12345678",
    )
    assert response.json()["totals"] == expected
    assert response.json()["models"] == [dict(model="fake", **expected)]
    empty = await setup.client.get(
        "/admin/usage",
        headers=auth,
        params={
            "start": (now - timedelta(days=2)).isoformat(),
            "end": (now - timedelta(days=1)).isoformat(),
        },
    )
    assert empty.json()["totals"]["requests"] == 0
    assert empty.json()["totals"]["cost"] == "0.00000000"


async def test_key_created_once_and_revoke_idempotent(setup):
    auth = headers(setup.users[0])
    response = await setup.client.post(
        "/admin/keys", headers=auth, json={"name": "created"}
    )
    assert response.status_code == 200
    created = response.json()
    assert parse_key(created["key"]) == created["key_prefix"]
    async with setup.sessions() as session:
        row = await session.get(ApiKey, UUID(created["id"]))
        assert row.tenant_id == setup.users[0].tenant_id
        assert row.key_hash == hash_key(created["key"])
        assert created["key"] not in [
            str(getattr(row, c.name)) for c in ApiKey.__table__.columns
        ]
    listing = await setup.client.get("/admin/keys", headers=auth)
    assert created["key"] not in listing.text
    assert row.key_hash not in listing.text
    path = f"/admin/keys/{created['id']}/revoke"
    first = await setup.client.post(path, headers=auth)
    second = await setup.client.post(path, headers=auth)
    assert first.status_code == second.status_code == 200
    assert first.json()["revoked_at"] is not None
    assert first.json()["revoked_at"] == second.json()["revoked_at"]


@pytest.mark.parametrize("path", ["/admin/keys", "/auth/login"])
async def test_api_key_cannot_authenticate_users(setup, path):
    key = generate_key()
    async with setup.sessions.begin() as session:
        session.add(
            ApiKey(
                tenant_id=setup.tenants[0].id,
                name="valid",
                key_prefix=parse_key(key),
                key_hash=hash_key(key),
            )
        )
    auth = {"Authorization": f"Bearer {key}"}
    response = (
        await setup.client.get(path, headers=auth)
        if path.startswith("/admin")
        else await setup.client.post(
            path, headers=auth, json={"email": "a@example.com", "password": PASSWORD}
        )
    )
    assert response.status_code == 401


async def test_jwt_cannot_authenticate_gateway(setup):
    response = await setup.client.post(
        "/v1/chat/completions",
        headers=headers(setup.users[0]),
        json={"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401


@pytest.mark.parametrize("value", [None, "short"])
async def test_unconfigured_admin_is_503(setup, monkeypatch, caplog, value):
    monkeypatch.setattr(
        admin_auth, "get_settings", lambda: Settings(_env_file=None, jwt_secret=value)
    )
    assert (
        await setup.client.get("/admin/keys", headers=headers(setup.users[0]))
    ).status_code == 503
    assert (
        await setup.client.post(
            "/auth/login", json={"email": "a@example.com", "password": PASSWORD}
        )
    ).status_code == 503
    assert "JWT_SECRET must contain at least 32 bytes" in caplog.text
    assert (await setup.client.get("/health")).status_code == 200


@pytest.mark.parametrize("days", [0, -1, 91])
async def test_usage_range_validation(setup, days):
    start = datetime.now(timezone.utc)
    response = await setup.client.get(
        "/admin/usage",
        headers=headers(setup.users[0]),
        params={
            "start": start.isoformat(),
            "end": (start + timedelta(days=days)).isoformat(),
        },
    )
    assert response.status_code == 422


async def test_cli_create_user(setup, engine, monkeypatch):
    prompt = Mock(return_value=PASSWORD)
    monkeypatch.setattr(cli.getpass, "getpass", prompt)
    monkeypatch.setattr(cli, "get_sessionmaker", lambda: setup.sessions)
    monkeypatch.setattr(cli, "get_engine", lambda: engine)
    await cli.run(
        argparse.Namespace(
            command="create-user", tenant="a", email=" CLI@Example.COM ", role="viewer"
        )
    )
    prompt.assert_called_once()
    async with setup.sessions() as session:
        row = await session.scalar(select(User).where(User.email == "cli@example.com"))
    assert row is not None
    assert row.role == "viewer"
    assert row.tenant_id == setup.tenants[0].id
    assert row.password_hash != PASSWORD
    assert verify_password(row.password_hash, PASSWORD)
