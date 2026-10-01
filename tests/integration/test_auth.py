import asyncio
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx2
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.db import get_sessionmaker
from llm_gateway.main import app, get_provider
from llm_gateway.models import ApiKey, Tenant
from llm_gateway.providers import FakeProvider
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client(engine):  # Dedicated testcontainer engine.
    app.dependency_overrides[get_sessionmaker] = lambda: async_sessionmaker(engine)
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        del app.dependency_overrides[get_sessionmaker]


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


async def test_auth_releases_connection_before_provider(
    engine, database_url
):  # Existing fixture migrates and cleans the isolated database.
    key = generate_key()
    async with async_sessionmaker(engine)() as session:
        tenant = Tenant(name="pool-regression")
        session.add(tenant)
        await session.flush()
        session.add(
            ApiKey(
                tenant_id=tenant.id,
                name="pool-key",
                key_prefix=parse_key(key),
                key_hash=hash_key(key),
            )
        )
        await session.commit()

    class SlowProvider:
        async def complete(
            self,
            request: ChatCompletionRequest,  # Preserve the normal fake response.
        ) -> ChatCompletionResponse:
            await asyncio.sleep(2)
            return await FakeProvider().complete(request)

    limited_engine = create_async_engine(
        database_url, pool_size=1, max_overflow=0, pool_timeout=1
    )
    app.dependency_overrides[get_sessionmaker] = lambda: async_sessionmaker(
        limited_engine
    )
    app.dependency_overrides[get_provider] = lambda: SlowProvider()
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Holding the auth connection during the 2-second provider call makes
            # the other requests exceed the 1-second timeout of this one-slot pool.
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": "fake", "messages": []},
                    )
                    for _ in range(3)
                ),
                return_exceptions=True,
            )
        assert all(
            isinstance(response, httpx2.Response) and response.status_code == 200
            for response in responses
        ), responses
    finally:
        del app.dependency_overrides[get_sessionmaker]
        del app.dependency_overrides[get_provider]
        await limited_engine.dispose()
