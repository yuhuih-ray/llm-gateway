import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(items):  # Mark every test in this directory.
    for item in items:
        if Path(__file__).parent in item.path.parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def database_url():
    with PostgresContainer("postgres:18", driver="asyncpg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture(scope="session")
def migrate(database_url):  # Always use the throwaway container URL.
    def run(*arguments):  # Alembic command and target revision.
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *arguments],
            cwd=ROOT,
            env={**os.environ, "DATABASE_URL": database_url},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    run("upgrade", "head")
    return run


@pytest.fixture
async def engine(database_url, migrate):  # Dedicated engine; never use get_engine().
    engine = create_async_engine(database_url)
    try:
        yield engine
    finally:
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "TRUNCATE tenants, api_keys, usage_logs RESTART IDENTITY CASCADE"
                    )
                )
        finally:
            await engine.dispose()
