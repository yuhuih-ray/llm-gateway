"""Seed only llm_gateway_bench; run with uv run python scripts/seed_usage.py."""

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import datetime

import asyncpg
from sqlalchemy.engine import make_url

from llm_gateway.config import get_settings
from llm_gateway.registry import MODELS

DATABASE = "llm_gateway_bench"
AS_OF = "2026-10-05T00:00:00+00:00"


def bench_url():
    # Borrow Compose connection credentials, never connect to the development DB.
    return make_url(get_settings().database_url).set(database=DATABASE)


async def connect():
    connection = await asyncpg.connect(
        bench_url().set(drivername="postgresql").render_as_string(hide_password=False)
    )
    assert await connection.fetchval("SELECT current_database()") == DATABASE
    return connection


async def seed(reset: bool, as_of: datetime):  # Reset is restricted to the bench DB.
    url = bench_url()
    maintenance = await asyncpg.connect(
        url.set(drivername="postgresql", database="postgres").render_as_string(
            hide_password=False
        )
    )
    try:
        if not await maintenance.fetchval(
            "SELECT 1 FROM pg_database WHERE datname=$1", DATABASE
        ):
            await maintenance.execute('CREATE DATABASE "llm_gateway_bench"')
    finally:
        await maintenance.close()
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env={**os.environ, "DATABASE_URL": url.render_as_string(hide_password=False)},
        check=True,
    )
    conn = await connect()
    try:
        if reset:
            await conn.execute(
                "TRUNCATE users, usage_logs, api_keys, tenants RESTART IDENTITY CASCADE"
            )
        if await conn.fetchval("SELECT count(*) FROM tenants"):
            raise RuntimeError("Benchmark is populated; pass --reset to reseed it")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS bench_metadata (as_of timestamptz NOT NULL)"
        )
        await conn.execute("TRUNCATE bench_metadata")
        await conn.execute("INSERT INTO bench_metadata VALUES ($1)", as_of)
        await conn.execute("""
            CREATE TEMP TABLE prices (
                model text, upstream text, input numeric, output numeric,
                long_input numeric, long_output numeric, slot int
            )
        """)
        await conn.copy_records_to_table(
            "prices",
            records=[
                (
                    name,
                    entry.upstream_model,
                    entry.input_price,
                    entry.output_price,
                    entry.long_input_price,
                    entry.long_output_price,
                    i,
                )
                for i, (name, entry) in enumerate(MODELS.items())
            ],
        )
        print(
            "Seeding 1,000 tenants and 3,000,000 rows (Zipf exponent 1.35)", flush=True
        )
        async with conn.transaction():
            await conn.execute("""
                CREATE TEMP TABLE tenant_counts AS
                WITH weights AS (
                    SELECT rank, power(rank::numeric, -1.35) AS weight
                    FROM generate_series(1,1000) rank
                ), counts AS (
                    SELECT rank, floor(3000000 * weight / sum(weight) OVER ())::int AS n
                    FROM weights
                )
                SELECT rank, n + CASE WHEN rank=1 THEN 3000000-sum(n) OVER () ELSE 0 END AS n
                FROM counts;
                INSERT INTO tenants (id, name)
                SELECT md5('tenant-'||rank)::uuid, 'bench-'||lpad(rank::text,4,'0')
                FROM tenant_counts;
                INSERT INTO api_keys (id, tenant_id, name, key_prefix, key_hash)
                SELECT md5('key-'||rank)::uuid, md5('tenant-'||rank)::uuid,
                       'synthetic', 'bench_'||rank, repeat('0',64)
                FROM tenant_counts;
            """)
            await conn.execute(
                """
                INSERT INTO usage_logs (
                    request_id, tenant_id, api_key_id, model_requested, model_selected,
                    upstream_model, prompt_tokens, completion_tokens, reasoning_tokens,
                    cost, latency_ms, status, error_type, created_at
                )
                WITH ids AS (
                    SELECT rank, row_number() OVER (ORDER BY rank, local_id) AS g
                    FROM tenant_counts CROSS JOIN LATERAL generate_series(1,n) local_id
                ), samples AS (
                    SELECT *, (hashtextextended(g::text,1)&9223372036854775807)%100 AS status_slot,
                        (hashtextextended(g::text,2)&9223372036854775807)%4 AS model_slot,
                        (hashtextextended(g::text,3)&9223372036854775807)%12000+50 AS prompt,
                        (hashtextextended(g::text,4)&9223372036854775807)%2033+16 AS completion
                    FROM ids
                ), tokens AS (
                    SELECT *, CASE WHEN model='fake' THEN 0
                        WHEN model='gemini-pro' AND g%100=0 THEN prompt+200000
                        ELSE prompt END AS pt,
                        CASE WHEN model='fake' THEN 0 ELSE completion END AS ct
                    FROM samples JOIN prices ON slot=model_slot
                )
                SELECT md5('request-'||g)::uuid, md5('tenant-'||rank)::uuid,
                    md5('key-'||rank)::uuid, model, model, upstream,
                    CASE WHEN status_slot<95 THEN pt END,
                    CASE WHEN status_slot<95 THEN ct END,
                    CASE WHEN status_slot>=95 THEN NULL
                        WHEN model IN ('fake','gemini-flash-lite') THEN 0 ELSE ct/3 END,
                    CASE WHEN status_slot<95 THEN round((pt *
                        CASE WHEN pt>200000 AND long_input IS NOT NULL THEN long_input ELSE input END
                        + ct * CASE WHEN pt>200000 AND long_output IS NOT NULL THEN long_output ELSE output END
                        ) / 1000000, 8) END,
                    100+(g%10000)::int,
                    CASE WHEN status_slot<95 THEN 'success' WHEN status_slot<98 THEN 'error' ELSE 'cancelled' END,
                    CASE WHEN status_slot BETWEEN 95 AND 97 THEN 'upstream_error' END,
                    $1::timestamptz - (((hashtextextended(g::text,5)&9223372036854775807)%15552000000000)+1)
                        * interval '1 microsecond'
                FROM tokens
                -- Deterministic permutation avoids physically clustering rows by tenant.
                ORDER BY (g*15485863)%3000001
            """,
                as_of,
            )
        await conn.execute("VACUUM (ANALYZE) usage_logs")
        print(
            await conn.fetch("""
            SELECT t.name, count(*) AS rows FROM usage_logs u JOIN tenants t ON t.id=u.tenant_id
            GROUP BY t.name ORDER BY rows DESC LIMIT 3
        """),
            flush=True,
        )
        print("Seed complete; VACUUM ANALYZE complete", flush=True)
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--as-of", default=AS_OF)
    args = parser.parse_args()
    asyncio.run(seed(args.reset, datetime.fromisoformat(args.as_of)))
