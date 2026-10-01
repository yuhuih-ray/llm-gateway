import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx2
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.db import get_session
from llm_gateway.main import app
from llm_gateway.models import ApiKey, Tenant

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client(engine):  # Dedicated testcontainer engine.
    async def session_override():
        async with async_sessionmaker(engine)() as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        del app.dependency_overrides[get_session]


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "missing",
        "scheme",
        "malformed",
        "unknown",
        "wrong",
        "revoked",
        "suspended",
    ],
)
async def test_authentication(
    engine, client, case, caplog
):  # Isolated DB, app, and failure case.
    key = generate_key()
    async with async_sessionmaker(engine)() as session:
        tenant = Tenant(
            name="auth", status="suspended" if case == "suspended" else "active"
        )
        session.add(tenant)
        await session.flush()
        session.add(
            ApiKey(
                tenant_id=tenant.id,
                name="test",
                key_prefix=parse_key(key),
                key_hash=hash_key(key),
                revoked_at=datetime.now(timezone.utc) if case == "revoked" else None,
            )
        )
        await session.commit()
    headers = {"Authorization": f"Bearer {key}"}
    if case == "missing":
        headers = {}
    elif case == "scheme":
        headers = {"Authorization": f"Basic {key}"}
    elif case == "malformed":
        headers = {"Authorization": "Bearer invalid"}
    elif case == "unknown":
        headers = {"Authorization": f"Bearer {generate_key()}"}
    elif case == "wrong":
        headers = {
            "Authorization": f"Bearer {parse_key(key)}_{generate_key().split('_', 2)[2]}"
        }
    response = await client.post(
        "/v1/chat/completions", headers=headers, json={"model": "fake", "messages": []}
    )
    if case == "valid":
        assert response.status_code == 200
    elif case == "suspended":
        assert response.status_code == 403
        assert response.json() == {
            "error": {"message": "Tenant is suspended", "type": "permission_error"}
        }
    else:
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json() == {
            "error": {
                "message": "Missing API key"
                if case == "missing"
                else "Invalid API key",
                "type": "authentication_error",
            }
        }
    assert key not in caplog.text
    assert hash_key(key) not in caplog.text
    assert (await client.get("/health")).status_code == 200


async def test_cli_key_storage(
    engine, database_url
):  # CLI must target only the container.
    def cli(*arguments):  # CLI command arguments.
        result = subprocess.run(
            [sys.executable, "-m", "llm_gateway.cli", *arguments],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "DATABASE_URL": database_url},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return result

    cli("create-tenant", "cli-tenant")
    result = cli("create-key", "--tenant", "cli-tenant", "--name", "cli-key")
    key = result.stdout.strip()
    prefix = parse_key(key)
    assert "will not be shown again" in result.stderr
    async with engine.connect() as connection:
        row = (await connection.execute(select(ApiKey.__table__))).one()
        assert key not in [str(value) for value in row]
        assert row.key_prefix == prefix
        assert row.key_hash == hash_key(key)
    cli("revoke-key", prefix)
    async with engine.connect() as connection:
        assert (
            await connection.execute(select(ApiKey.revoked_at))
        ).scalar_one() is not None
