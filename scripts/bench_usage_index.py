"""Measure temporary candidate indexes only on llm_gateway_bench."""

import asyncio
import json
import platform
import statistics
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from seed_usage import connect
from sqlalchemy.dialects import postgresql

from llm_gateway.admin import usage_report

VARIANTS = {
    "A": None,
    "B": "(tenant_id, created_at)",
    "C": "(tenant_id, created_at) INCLUDE (model_selected, prompt_tokens, completion_tokens, reasoning_tokens, cost)",
}
INDEX = "bench_usage_candidate"


async def exact_query(tenant, start, end):  # Capture the actual endpoint statement.
    class Capture:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, statement):
            self.statement = statement
            return SimpleNamespace(all=lambda: [])

    capture = Capture()
    await usage_report(SimpleNamespace(tenant_id=tenant), lambda: capture, start, end)
    return str(
        capture.statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def scans(node):  # Parallel-aware JSON uses the same node types as serial plans.
    found = set()
    if node.get("Relation Name") == "usage_logs" and "Scan" in node["Node Type"]:
        found.add(
            ("Parallel " if node.get("Parallel Aware") else "") + node["Node Type"]
        )
    for child in node.get("Plans", []):
        found.update(scans(child))
    return sorted(found)


def host(command):  # Environment commands contain no credentials.
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


async def main():
    conn = await connect()
    results, plans = [], {}
    try:
        # Refuse to benchmark a changed baseline rather than silently measuring it.
        indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename='usage_logs'"
        )
        assert {r["indexname"] for r in indexes} <= {
            "pk_usage_logs",
            "uq_usage_logs_request_id",
            INDEX,
        }, "Unexpected usage index in benchmark database"
        await conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
        assert await conn.fetchval("SELECT count(*) FROM usage_logs") == 3000000
        tenants = await conn.fetch("""
            SELECT t.id, t.name, count(u.id) AS n FROM tenants t
            LEFT JOIN usage_logs u ON u.tenant_id=t.id GROUP BY t.id
            ORDER BY n DESC, t.name
        """)
        assert len(tenants) == 1000
        # Lower central rank, with the arithmetic median also reported for 1,000 tenants.
        largest, median = tenants[0], tenants[499]
        end = await conn.fetchval("SELECT as_of FROM bench_metadata")
        environment = {
            "host": platform.platform(),
            "cpu": host(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "host_memory_bytes": host(["sysctl", "-n", "hw.memsize"]),
            "docker": host(
                [
                    "docker",
                    "info",
                    "--format",
                    "CPUs={{.NCPU}} memory_bytes={{.MemTotal}} kernel={{.KernelVersion}}",
                ]
            ),
            "postgres": await conn.fetchval("SELECT version()"),
            "settings": dict(
                (r["name"], r["setting"])
                for r in await conn.fetch("""
                SELECT name, setting FROM pg_settings WHERE name IN
                ('shared_buffers','work_mem','effective_cache_size','random_page_cost',
                 'max_parallel_workers_per_gather','jit','synchronous_commit')
            """)
            ),
            "heap_bytes": await conn.fetchval("SELECT pg_relation_size('usage_logs')"),
            "correlation": {
                r["attname"]: r["correlation"]
                for r in await conn.fetch(
                    "SELECT attname, correlation FROM pg_stats WHERE schemaname='public' AND tablename='usage_logs' AND attname IN ('created_at','tenant_id') ORDER BY attname"
                )
            },
            "rows": 3000000,
            "tenants": 1000,
            "as_of": end.isoformat(),
            "status_counts": dict(
                (r["status"], r["n"])
                for r in await conn.fetch(
                    "SELECT status,count(*) n FROM usage_logs GROUP BY status"
                )
            ),
        }
        cases = [
            ("largest / 30d", largest["id"], 30),
            ("median / 30d", median["id"], 30),
            ("largest / 90d", largest["id"], 90),
        ]
        for variant, definition in VARIANTS.items():
            print(f"Variant {variant}: {definition}", flush=True)
            if definition:
                await conn.execute(f"CREATE INDEX {INDEX} ON usage_logs {definition}")
            await conn.execute("VACUUM (ANALYZE) usage_logs")
            size = (
                await conn.fetchval(f"SELECT pg_relation_size('{INDEX}')")
                if definition
                else 0
            )
            for label, tenant, days in cases:
                sql = await exact_query(tenant, end - timedelta(days=days), end)
                runs = [
                    json.loads(
                        await conn.fetchval(
                            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql
                        )
                    )[0]
                    for _ in range(5)
                ]
                median_ms = statistics.median(r["Execution Time"] for r in runs)
                representative = min(
                    runs, key=lambda r: abs(r["Execution Time"] - median_ms)
                )
                plan = representative["Plan"]
                row = dict(
                    variant=variant,
                    case=label,
                    ms=median_ms,
                    scan=", ".join(scans(plan)),
                    hit=plan.get("Shared Hit Blocks", 0),
                    read=plan.get("Shared Read Blocks", 0),
                    index_bytes=size,
                    times_ms=[r["Execution Time"] for r in runs],
                )
                results.append(row)
                if label == "largest / 30d":
                    plans[variant] = representative
                print(json.dumps(row), flush=True)
            # Same first 100,000 baseline rows each time; only request IDs differ.
            started = time.perf_counter()
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO usage_logs (
                        request_id, tenant_id, api_key_id, model_requested, model_selected,
                        upstream_model, prompt_tokens, completion_tokens, reasoning_tokens,
                        cost, latency_ms, status, error_type, created_at
                    )
                    SELECT md5('extra-'||id)::uuid, tenant_id, api_key_id, model_requested,
                        model_selected, upstream_model, prompt_tokens, completion_tokens,
                        reasoning_tokens, cost, latency_ms, status, error_type, created_at
                    FROM usage_logs WHERE id <= 100000
                """)
            insert_ms = (time.perf_counter() - started) * 1000
            for row in results:
                if row["variant"] == variant:
                    row["insert_ms"] = insert_ms
            print(
                f"100,000-row insert including commit: {insert_ms:.3f} ms", flush=True
            )
            await conn.execute("DELETE FROM usage_logs WHERE id > 3000000")
            await conn.execute("VACUUM (ANALYZE) usage_logs")
            await conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
        assert await conn.fetchval("SELECT count(*) FROM usage_logs") == 3000000
        best = min(
            (r for r in results if r["case"] == "largest / 30d"), key=lambda r: r["ms"]
        )["variant"]
        table = [
            "| Variant | Case | Median ms | Scan | Shared hit | Shared read | Hit + read | Index MiB | Insert 100k ms |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in results:
            table.append(
                f"| {r['variant']} | {r['case']} | {r['ms']:.3f} | {r['scan']} | {r['hit']} | {r['read']} | {r['hit'] + r['read']} | {r['index_bytes'] / 1048576:.2f} | {r['insert_ms']:.3f} |"
            )
        counts = "\n".join(f"- {r['name']}: {r['n']:,} rows" for r in tenants[:3])
        doc = f"""# Run 2 (append-ordered data)

## Reproduce

Run from the repository root with Compose Postgres available:

```sh
uv run python scripts/seed_usage.py --reset
uv run python scripts/bench_usage_index.py
```

Both scripts derive connection credentials from Settings but force the database
name to llm_gateway_bench and check current_database() before data operations.
Database creation uses the postgres maintenance database. No development database
connection is made. --reset deletes only benchmark data. No migration adds an index.
The benchmark drops candidate indexes on completion or failure.

## Environment and data

```json
{json.dumps(environment, indent=2)}
```

Deterministic set-based seed: 1,000 tenants and keys; Zipf exponent 1.35, integer
counts with the remainder assigned to rank 1. Insert order is created_at ascending,
with g as a stable tie-breaker. Tenants, models, tokens and statuses retain the same
independent per-row sampling. Hash-derived timestamps cover the preceding 180 days
relative to the fixed as_of above. Model choice is uniform across the four registry
models; prompt tokens 50–12,049 (1% of Pro rows add 200,000), completion tokens
16–2,048 including reasoning. Fake token counts/cost are zero. Unknown usage on
errors/cancellations stays NULL. Decimal SQL prices come directly from the registry,
including Pro's long-context tier. Successful Flash/Pro reasoning is one third of
completion tokens. This is synthetic data, not a production traffic trace.

{counts}
- Median tenant used (rank 500, ties by name): {median["name"]}: {median["n"]:,} rows
- Arithmetic median of ranks 500 and 501: {statistics.median([r["n"] for r in tenants]):,.1f} rows

VACUUM ANALYZE runs after seeding and before measurements. A retains the schema's
primary-key and unique-request-ID indexes, but has no report-query index.
B: (tenant_id, created_at); C: the same keys with all report columns INCLUDEd.
D is omitted in Run 2. The exact SQL is captured by invoking usage_report with
a recording session and compiling its statement with PostgreSQL literal parameters;
there is no separately maintained query copy and no application-code change.

## Measurements

Each case runs five times in A/B/C order, with no cold-cache reset. Execution
Time is PostgreSQL EXPLAIN's server execution time, not client round-trip time.
Buffers and full plan come from the median-time run; root buffer counters include
children, so they are not summed again. Parallel scans are labeled explicitly.
Index size excludes baseline constraint indexes. Insert timing is one wall-clock
sample per variant, including transaction commit, copying the same first 100,000
baseline rows with fresh request IDs. Since physical order changed, these are now
the oldest 100,000 baseline rows; timestamps are copied unchanged, just as in Run 1.
This retained write test measures index maintenance, not new-timestamp append throughput.
Extra rows are deleted and VACUUM ANALYZE
runs before the next variant. Index creation time is not part of query timing.

{chr(10).join(table)}

## Why physical time order matters for B

B locates tenant/time matches in its B-tree but still needs heap columns to compute
the aggregates. With shuffled timestamps, a 30-day range can touch heap pages
throughout the full 180-day table. With append ordering, matching rows lie in a
contiguous recent-time region even though tenants are interleaved. Bitmap scans
can therefore visit fewer heap pages, and ordinary index scans have better heap
locality. The measured planner choice is reported rather than forced. C can avoid
heap access altogether when the visibility map permits index-only scans.

## Decision

Recommend variant {best} for this report workload: it has the lowest median for
the largest-tenant 30-day case. The table also exposes its 90-day and small-tenant
performance, storage and insertion trade-offs. Consider B instead if write/storage
cost dominates; physical time locality can also improve its large-tenant reports. These repeated-run
synthetic measurements are not an SLA; the single insert sample and fixed variant
order are susceptible to background load/cache and WAL/checkpoint effects. Shared
reads may be served by the OS cache; no cold-cache claim is made. No index is
installed on the development database.

## Full EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON), median run

### A, largest tenant / 30 days

```json
{json.dumps(plans["A"], indent=2)}
```

### {best}, largest tenant / 30 days

```json
{json.dumps(plans[best], indent=2)}
```

## All five execution-time samples

```json
{json.dumps(results, indent=2)}
```
"""
        path = Path("docs/benchmarks/usage-report-index.md")
        previous = path.read_text().split("# Run 2 (append-ordered data)", 1)[0]
        path.write_text(previous.rstrip() + "\n\n" + doc)
        print(counts)
        print(f"Median: {median['name']} {median['n']}")
        print("\n".join(table), flush=True)
    finally:
        try:
            await conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
        finally:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
