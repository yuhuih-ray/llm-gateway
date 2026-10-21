"""Administration restricted to the fixed, isolated load-test services."""

import argparse
import asyncio
import json
import os
import secrets
import shlex
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_gateway.auth import generate_key, hash_key, parse_key
from llm_gateway.models import ApiKey, Tenant, UsageLog

STATE = Path(__file__).resolve().parent / ".state"
GATEWAY = "http://127.0.0.1:18000"
REDIS_URL = "redis://127.0.0.1:16379/0"
CACHE_URL = "redis://127.0.0.1:16380/0"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(
        path, "w", opener=lambda name, flags: os.open(name, flags, 0o600)
    ) as file:
        os.fchmod(file.fileno(), 0o600)
        file.write(json.dumps(value, indent=2) + "\n")


def initialize() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "env"
    if path.exists():
        return
    password = secrets.token_hex(24)
    values = {
        "LOADTEST_DB_PASSWORD": password,
        "DATABASE_URL": f"postgresql+asyncpg://gateway_loadtest:{password}@127.0.0.1:15432/gateway_loadtest",
        "REDIS_URL": REDIS_URL,
        "CACHE_REDIS_URL": CACHE_URL,
        "TENANT_REQUEST_CAPACITY": "1000000000000",
        "TENANT_REQUEST_REFILL": "1000000000",
        "TENANT_TOKEN_CAPACITY": "1000000000000",
        "TENANT_TOKEN_REFILL": "1000000000",
        "GEMINI_REQUEST_CAPACITY": "1000000000000",
        "GEMINI_REQUEST_REFILL": "1000000000",
        "GEMINI_API_KEY": "",  # Fake-only load tests never need provider credentials.
        "JWT_SECRET": "",
        "FAKE_DELAY_MS": "0",
        "SERVER_TIMING": "false",
    }
    # Exclusive creation and private permissions, even outside the shell wrapper.
    with open(
        path, "x", opener=lambda name, flags: os.open(name, flags, 0o600)
    ) as file:
        file.write(
            "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items())
        )


def validate_environment() -> str:
    # Refuse dev URLs even if a caller bypasses common.sh or edits the state file.
    url = os.environ.get("DATABASE_URL", "")
    parsed = make_url(url)
    if (
        parsed.drivername != "postgresql+asyncpg"
        or parsed.host != "127.0.0.1"
        or parsed.port != 15432
        or parsed.database != "gateway_loadtest"
        or parsed.username != "gateway_loadtest"
        or os.environ.get("REDIS_URL") != REDIS_URL
        or os.environ.get("CACHE_REDIS_URL") != CACHE_URL
    ):
        raise ValueError("Refusing non-loadtest database or Redis endpoints")
    return url


async def seed(url: str) -> None:  # Validated isolated database, never the dev URL.
    engine = create_async_engine(url)
    try:
        async with async_sessionmaker(engine)() as session:
            async with session.begin():
                existing = await session.scalar(
                    select(func.count()).select_from(Tenant)
                )
                if existing:
                    if existing != 50 or not (STATE / "keys.json").exists():
                        raise RuntimeError(
                            "Unexpected seed state; teardown and setup again"
                        )
                    return
                keys = []
                for index in range(50):
                    tenant = Tenant(
                        name=f"loadtest-{index:02d}", monthly_budget_usd=None
                    )
                    session.add(tenant)
                    await session.flush()
                    key = generate_key()
                    session.add(
                        ApiKey(
                            tenant_id=tenant.id,
                            name="loadtest",
                            key_prefix=parse_key(key),
                            key_hash=hash_key(key),
                        )
                    )
                    keys.append(key)
        write_json(STATE / "keys.json", keys)
    finally:
        await engine.dispose()


async def usage_count(url: str) -> int:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return int(
                await connection.scalar(select(func.count()).select_from(UsageLog)) or 0
            )
    finally:
        await engine.dispose()


async def queue_state(pool: Redis) -> dict:
    # ARQ uses a sorted set, not a Redis list. In-progress jobs may still be in it.
    return {
        "queue_length": await pool.zcard("arq:queue"),
        "in_progress": len([key async for key in pool.scan_iter("arq:in-progress:*")]),
    }


async def reset(url: str) -> None:
    async with Redis.from_url(REDIS_URL) as queue, Redis.from_url(CACHE_URL) as cache:
        if any((await queue_state(queue)).values()):
            raise RuntimeError(
                "Prior ARQ work remains; drain it or teardown/setup before rerunning"
            )
        for client, pattern in (
            (queue, "ratelimit:*"),
            (queue, "budget:*"),
            (cache, "cache:*"),
        ):
            keys = [key async for key in client.scan_iter(pattern)]
            for offset in range(0, len(keys), 500):
                await client.delete(*keys[offset : offset + 500])
    write_json(
        STATE / "baseline.json",
        {
            "usage_rows": await usage_count(url),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def metric_count(summary: dict, name: str) -> int:
    # --summary-export uses flat counters; also accept k6's values-wrapped format.
    metric = summary["metrics"][name]
    return int(metric.get("values", metric)["count"])


async def finish(url: str, args: argparse.Namespace) -> None:
    baseline = json.loads((STATE / "baseline.json").read_text())
    summary = json.loads(Path(args.summary).read_text())
    sent = metric_count(summary, "requests_sent")
    successful = metric_count(summary, "requests_successful")
    before = baseline["usage_rows"]
    started = time.perf_counter()
    async with Redis.from_url(REDIS_URL) as queue:
        initial = await queue_state(queue)
        while True:
            state = await queue_state(queue)
            written = await usage_count(url) - before
            if written >= sent and not any(state.values()):
                break
            if time.perf_counter() - started >= 30:
                break
            await asyncio.sleep(0.25)
    write_json(
        Path(args.output),
        {
            "scenario": args.scenario,
            "smoke": args.smoke,
            "k6_exit_code": args.exit_code,
            "requests_sent": sent,
            "requests_successful": successful,
            "usage_rows_written": written,
            "usage_rows_before": before,
            "usage_rows_minus_requests_sent": written - sent,
            "arq_at_k6_end": initial,
            "arq_after_drain": state,
            "drain_wait_seconds": round(time.perf_counter() - started, 3),
            "drained": not any(state.values()) and written >= sent,
            "started_at": baseline["started_at"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "fake_delay_ms": int(os.environ["FAKE_DELAY_MS"]),
            "server_timing": os.environ["SERVER_TIMING"] == "true",
        },
    )


def wait_ready() -> None:
    for _ in range(100):
        try:
            with urllib.request.urlopen(
                GATEWAY + "/health/ready", timeout=1
            ) as response:
                if json.load(response)["checks"] == {
                    "postgres": "ok",
                    "usage_queue": "ok",
                }:
                    return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Gateway not ready; inspect .state gateway log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["init", "validate", "seed", "reset", "ready", "port", "finish"],
    )
    parser.add_argument("--summary")
    parser.add_argument("--output")
    parser.add_argument("--scenario", choices=["s1", "s2", "s3"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    if args.command == "init":
        initialize()
        return
    url = validate_environment()
    if args.command == "seed":
        asyncio.run(seed(url))
    elif args.command == "reset":
        asyncio.run(reset(url))
    elif args.command == "ready":
        wait_ready()
    elif args.command == "port":
        # Fail before launch instead of accidentally benchmarking another process.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 18000))
    elif args.command == "finish":
        if not args.summary or not args.output or not args.scenario:
            parser.error("finish requires --summary, --output, and --scenario")
        asyncio.run(finish(url, args))


if __name__ == "__main__":
    main()
