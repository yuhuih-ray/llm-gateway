# Usage report index benchmark

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
{
  "host": "macOS-27.0-arm64-arm-64bit",
  "cpu": "Apple M4",
  "host_memory_bytes": "25769803776",
  "docker": "CPUs=10 memory_bytes=8321712128 kernel=6.12.76-linuxkit",
  "postgres": "PostgreSQL 18.6 (Debian 18.6-1.pgdg13+2) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 14.2.0-19) 14.2.0, 64-bit",
  "settings": {
    "effective_cache_size": "524288",
    "jit": "on",
    "max_parallel_workers_per_gather": "2",
    "random_page_cost": "4",
    "shared_buffers": "16384",
    "synchronous_commit": "on",
    "work_mem": "4096"
  },
  "heap_bytes": 538828800,
  "rows": 3000000,
  "tenants": 1000,
  "as_of": "2026-10-05T00:00:00+00:00",
  "status_counts": {
    "cancelled": 60019,
    "error": 90414,
    "success": 2849567
  }
}
```

Deterministic set-based seed: 1,000 tenants and keys; Zipf exponent 1.35, integer
counts with the remainder assigned to rank 1. A deterministic permutation mixes
physical insertion order. Hash-derived timestamps cover the preceding 180 days
relative to the fixed as_of above. Model choice is uniform across the four registry
models; prompt tokens 50–12,049 (1% of Pro rows add 200,000), completion tokens
16–2,048 including reasoning. Fake token counts/cost are zero. Unknown usage on
errors/cancellations stays NULL. Decimal SQL prices come directly from the registry,
including Pro's long-context tier. Successful Flash/Pro reasoning is one third of
completion tokens. This is synthetic data, not a production traffic trace.

- bench-0001: 936,633 rows
- bench-0002: 367,241 rows
- bench-0003: 212,436 rows
- Median tenant used (rank 500, ties by name): bench-0500: 212 rows
- Arithmetic median of ranks 500 and 501: 212.0 rows

VACUUM ANALYZE runs after seeding and before measurements. A retains the schema's
primary-key and unique-request-ID indexes, but has no report-query index.
B: (tenant_id, created_at); C: the same keys with all report columns INCLUDEd;
D: (created_at, tenant_id). The exact SQL is captured by invoking usage_report with
a recording session and compiling its statement with PostgreSQL literal parameters;
there is no separately maintained query copy and no application-code change.

## Measurements

Each case runs five times in A/B/C/D order, with no cold-cache reset. Execution
Time is PostgreSQL EXPLAIN's server execution time, not client round-trip time.
Buffers and full plan come from the median-time run; root buffer counters include
children, so they are not summed again. Parallel scans are labeled explicitly.
Index size excludes baseline constraint indexes. Insert timing is one wall-clock
sample per variant, including transaction commit, copying the same first 100,000
baseline rows with fresh request IDs. Extra rows are deleted and VACUUM ANALYZE
runs before the next variant. Index creation time is not part of query timing.

| Variant | Case | Median ms | Scan | Shared hit | Shared read | Hit + read | Index MiB | Insert 100k ms |
|---|---|---:|---|---:|---:|---:|---:|---:|
| A | largest / 30d | 88.832 | Parallel Seq Scan | 9237 | 56554 | 65791 | 0.00 | 990.531 |
| A | median / 30d | 90.031 | Parallel Seq Scan | 9579 | 56272 | 65851 | 0.00 | 990.531 |
| A | largest / 90d | 87.242 | Parallel Seq Scan | 11775 | 54016 | 65791 | 0.00 | 990.531 |
| B | largest / 30d | 64.716 | Parallel Bitmap Heap Scan | 53 | 61487 | 61540 | 116.21 | 1189.062 |
| B | median / 30d | 0.038 | Bitmap Heap Scan | 39 | 0 | 39 | 116.21 | 1189.062 |
| B | largest / 90d | 86.582 | Parallel Seq Scan | 15988 | 49803 | 65791 | 116.21 | 1189.062 |
| C | largest / 30d | 20.778 | Parallel Index Only Scan | 80422 | 0 | 80422 | 224.86 | 1650.368 |
| C | median / 30d | 0.024 | Index Only Scan | 25 | 0 | 25 | 224.86 | 1650.368 |
| C | largest / 90d | 41.571 | Parallel Index Only Scan | 241829 | 0 | 241829 | 224.86 | 1650.368 |
| D | largest / 30d | 67.182 | Parallel Seq Scan | 2016 | 63775 | 65791 | 115.89 | 1180.309 |
| D | median / 30d | 4.346 | Index Scan | 2501 | 0 | 2501 | 115.89 | 1180.309 |
| D | largest / 90d | 85.875 | Parallel Seq Scan | 3180 | 62611 | 65791 | 115.89 | 1180.309 |

## Decision

Recommend **C** for this report workload. It wins all three measured cases:
4.28x faster than A for the largest tenant / 30 days and 2.10x faster for 90 days.
The covering index is 224.86 MiB versus B's 116.21 MiB. Its single insertion sample
was 1,650.368 ms versus A's 990.531 ms (66.6% slower), so the read improvement has
a material write/storage cost. B is an alternative if writes/storage dominate and
reports are mainly for small tenants: 0.038 ms there with about 20% insert overhead.
D is inferior to B for the small-tenant case and gives no meaningful large-tenant
index plan advantage. C's recorded largest-tenant 30-day plan has zero heap fetches.

Measured on 2026-10-05. These repeated-run synthetic measurements are not an SLA;
the single insert sample and fixed variant order are susceptible to background
load/cache and WAL/checkpoint effects. Shared reads may be served by the OS cache;
no cold-cache claim is made. No index is installed on the development database.

## Full EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON), median run

### A, largest tenant / 30 days

```json
{
  "Plan": {
    "Node Type": "Aggregate",
    "Strategy": "Sorted",
    "Partial Mode": "Finalize",
    "Parallel Aware": false,
    "Async Capable": false,
    "Startup Cost": 89620.27,
    "Total Cost": 89621.66,
    "Plan Rows": 4,
    "Plan Width": 75,
    "Actual Startup Time": 86.508,
    "Actual Total Time": 88.809,
    "Actual Rows": 4.0,
    "Actual Loops": 1,
    "Disabled": false,
    "Group Key": [
      "model_selected"
    ],
    "Shared Hit Blocks": 9237,
    "Shared Read Blocks": 56554,
    "Shared Dirtied Blocks": 0,
    "Shared Written Blocks": 0,
    "Local Hit Blocks": 0,
    "Local Read Blocks": 0,
    "Local Dirtied Blocks": 0,
    "Local Written Blocks": 0,
    "Temp Read Blocks": 0,
    "Temp Written Blocks": 0,
    "Plans": [
      {
        "Node Type": "Gather Merge",
        "Parent Relationship": "Outer",
        "Parallel Aware": false,
        "Async Capable": false,
        "Startup Cost": 89620.27,
        "Total Cost": 89621.44,
        "Plan Rows": 10,
        "Plan Width": 75,
        "Actual Startup Time": 86.502,
        "Actual Total Time": 88.801,
        "Actual Rows": 12.0,
        "Actual Loops": 1,
        "Disabled": false,
        "Workers Planned": 2,
        "Workers Launched": 2,
        "Shared Hit Blocks": 9237,
        "Shared Read Blocks": 56554,
        "Shared Dirtied Blocks": 0,
        "Shared Written Blocks": 0,
        "Local Hit Blocks": 0,
        "Local Read Blocks": 0,
        "Local Dirtied Blocks": 0,
        "Local Written Blocks": 0,
        "Temp Read Blocks": 0,
        "Temp Written Blocks": 0,
        "Plans": [
          {
            "Node Type": "Sort",
            "Parent Relationship": "Outer",
            "Parallel Aware": false,
            "Async Capable": false,
            "Startup Cost": 88620.25,
            "Total Cost": 88620.26,
            "Plan Rows": 4,
            "Plan Width": 75,
            "Actual Startup Time": 80.111,
            "Actual Total Time": 80.112,
            "Actual Rows": 4.0,
            "Actual Loops": 3,
            "Disabled": false,
            "Sort Key": [
              "model_selected"
            ],
            "Sort Method": "quicksort",
            "Sort Space Used": 25,
            "Sort Space Type": "Memory",
            "Shared Hit Blocks": 9237,
            "Shared Read Blocks": 56554,
            "Shared Dirtied Blocks": 0,
            "Shared Written Blocks": 0,
            "Local Hit Blocks": 0,
            "Local Read Blocks": 0,
            "Local Dirtied Blocks": 0,
            "Local Written Blocks": 0,
            "Temp Read Blocks": 0,
            "Temp Written Blocks": 0,
            "Workers": [
              {
                "Worker Number": 0,
                "Sort Method": "quicksort",
                "Sort Space Used": 25,
                "Sort Space Type": "Memory"
              },
              {
                "Worker Number": 1,
                "Sort Method": "quicksort",
                "Sort Space Used": 25,
                "Sort Space Type": "Memory"
              }
            ],
            "Plans": [
              {
                "Node Type": "Aggregate",
                "Strategy": "Hashed",
                "Partial Mode": "Partial",
                "Parent Relationship": "Outer",
                "Parallel Aware": false,
                "Async Capable": false,
                "Startup Cost": 88620.16,
                "Total Cost": 88620.21,
                "Plan Rows": 4,
                "Plan Width": 75,
                "Actual Startup Time": 79.944,
                "Actual Total Time": 79.946,
                "Actual Rows": 4.0,
                "Actual Loops": 3,
                "Disabled": false,
                "Group Key": [
                  "model_selected"
                ],
                "Planned Partitions": 0,
                "HashAgg Batches": 1,
                "Peak Memory Usage": 32,
                "Disk Usage": 0,
                "Shared Hit Blocks": 9221,
                "Shared Read Blocks": 56554,
                "Shared Dirtied Blocks": 0,
                "Shared Written Blocks": 0,
                "Local Hit Blocks": 0,
                "Local Read Blocks": 0,
                "Local Dirtied Blocks": 0,
                "Local Written Blocks": 0,
                "Temp Read Blocks": 0,
                "Temp Written Blocks": 0,
                "Workers": [
                  {
                    "Worker Number": 0,
                    "HashAgg Batches": 1,
                    "Peak Memory Usage": 32,
                    "Disk Usage": 0
                  },
                  {
                    "Worker Number": 1,
                    "HashAgg Batches": 1,
                    "Peak Memory Usage": 32,
                    "Disk Usage": 0
                  }
                ],
                "Plans": [
                  {
                    "Node Type": "Seq Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": true,
                    "Async Capable": false,
                    "Relation Name": "usage_logs",
                    "Alias": "usage_logs",
                    "Startup Cost": 0.0,
                    "Total Cost": 87648.83,
                    "Plan Rows": 64755,
                    "Plan Width": 29,
                    "Actual Startup Time": 0.246,
                    "Actual Total Time": 72.406,
                    "Actual Rows": 51962.0,
                    "Actual Loops": 3,
                    "Disabled": false,
                    "Filter": "((created_at >= '2026-09-05 00:00:00+00'::timestamp with time zone) AND (created_at < '2026-10-05 00:00:00+00'::timestamp with time zone) AND (tenant_id = 'e000342e-22c2-b525-5299-b35c4d538065'::uuid))",
                    "Rows Removed by Filter": 948038,
                    "Shared Hit Blocks": 9221,
                    "Shared Read Blocks": 56554,
                    "Shared Dirtied Blocks": 0,
                    "Shared Written Blocks": 0,
                    "Local Hit Blocks": 0,
                    "Local Read Blocks": 0,
                    "Local Dirtied Blocks": 0,
                    "Local Written Blocks": 0,
                    "Temp Read Blocks": 0,
                    "Temp Written Blocks": 0,
                    "Workers": []
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  },
  "Planning": {
    "Shared Hit Blocks": 0,
    "Shared Read Blocks": 0,
    "Shared Dirtied Blocks": 0,
    "Shared Written Blocks": 0,
    "Local Hit Blocks": 0,
    "Local Read Blocks": 0,
    "Local Dirtied Blocks": 0,
    "Local Written Blocks": 0,
    "Temp Read Blocks": 0,
    "Temp Written Blocks": 0
  },
  "Planning Time": 0.079,
  "Triggers": [],
  "Execution Time": 88.832
}
```

### C, largest tenant / 30 days

```json
{
  "Plan": {
    "Node Type": "Aggregate",
    "Strategy": "Sorted",
    "Partial Mode": "Finalize",
    "Parallel Aware": false,
    "Async Capable": false,
    "Startup Cost": 10537.32,
    "Total Cost": 10538.7,
    "Plan Rows": 4,
    "Plan Width": 75,
    "Actual Startup Time": 18.286,
    "Actual Total Time": 20.756,
    "Actual Rows": 4.0,
    "Actual Loops": 1,
    "Disabled": false,
    "Group Key": [
      "model_selected"
    ],
    "Shared Hit Blocks": 80422,
    "Shared Read Blocks": 0,
    "Shared Dirtied Blocks": 0,
    "Shared Written Blocks": 0,
    "Local Hit Blocks": 0,
    "Local Read Blocks": 0,
    "Local Dirtied Blocks": 0,
    "Local Written Blocks": 0,
    "Temp Read Blocks": 0,
    "Temp Written Blocks": 0,
    "Plans": [
      {
        "Node Type": "Gather Merge",
        "Parent Relationship": "Outer",
        "Parallel Aware": false,
        "Async Capable": false,
        "Startup Cost": 10537.32,
        "Total Cost": 10538.48,
        "Plan Rows": 10,
        "Plan Width": 75,
        "Actual Startup Time": 18.278,
        "Actual Total Time": 20.747,
        "Actual Rows": 12.0,
        "Actual Loops": 1,
        "Disabled": false,
        "Workers Planned": 2,
        "Workers Launched": 2,
        "Shared Hit Blocks": 80422,
        "Shared Read Blocks": 0,
        "Shared Dirtied Blocks": 0,
        "Shared Written Blocks": 0,
        "Local Hit Blocks": 0,
        "Local Read Blocks": 0,
        "Local Dirtied Blocks": 0,
        "Local Written Blocks": 0,
        "Temp Read Blocks": 0,
        "Temp Written Blocks": 0,
        "Plans": [
          {
            "Node Type": "Sort",
            "Parent Relationship": "Outer",
            "Parallel Aware": false,
            "Async Capable": false,
            "Startup Cost": 9537.29,
            "Total Cost": 9537.3,
            "Plan Rows": 4,
            "Plan Width": 75,
            "Actual Startup Time": 12.475,
            "Actual Total Time": 12.477,
            "Actual Rows": 4.0,
            "Actual Loops": 3,
            "Disabled": false,
            "Sort Key": [
              "model_selected"
            ],
            "Sort Method": "quicksort",
            "Sort Space Used": 25,
            "Sort Space Type": "Memory",
            "Shared Hit Blocks": 80422,
            "Shared Read Blocks": 0,
            "Shared Dirtied Blocks": 0,
            "Shared Written Blocks": 0,
            "Local Hit Blocks": 0,
            "Local Read Blocks": 0,
            "Local Dirtied Blocks": 0,
            "Local Written Blocks": 0,
            "Temp Read Blocks": 0,
            "Temp Written Blocks": 0,
            "Workers": [
              {
                "Worker Number": 0,
                "Sort Method": "quicksort",
                "Sort Space Used": 25,
                "Sort Space Type": "Memory"
              },
              {
                "Worker Number": 1,
                "Sort Method": "quicksort",
                "Sort Space Used": 25,
                "Sort Space Type": "Memory"
              }
            ],
            "Plans": [
              {
                "Node Type": "Aggregate",
                "Strategy": "Hashed",
                "Partial Mode": "Partial",
                "Parent Relationship": "Outer",
                "Parallel Aware": false,
                "Async Capable": false,
                "Startup Cost": 9537.2,
                "Total Cost": 9537.25,
                "Plan Rows": 4,
                "Plan Width": 75,
                "Actual Startup Time": 12.342,
                "Actual Total Time": 12.344,
                "Actual Rows": 4.0,
                "Actual Loops": 3,
                "Disabled": false,
                "Group Key": [
                  "model_selected"
                ],
                "Planned Partitions": 0,
                "HashAgg Batches": 1,
                "Peak Memory Usage": 32,
                "Disk Usage": 0,
                "Shared Hit Blocks": 80406,
                "Shared Read Blocks": 0,
                "Shared Dirtied Blocks": 0,
                "Shared Written Blocks": 0,
                "Local Hit Blocks": 0,
                "Local Read Blocks": 0,
                "Local Dirtied Blocks": 0,
                "Local Written Blocks": 0,
                "Temp Read Blocks": 0,
                "Temp Written Blocks": 0,
                "Workers": [
                  {
                    "Worker Number": 0,
                    "HashAgg Batches": 1,
                    "Peak Memory Usage": 32,
                    "Disk Usage": 0
                  },
                  {
                    "Worker Number": 1,
                    "HashAgg Batches": 1,
                    "Peak Memory Usage": 32,
                    "Disk Usage": 0
                  }
                ],
                "Plans": [
                  {
                    "Node Type": "Index Only Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": true,
                    "Async Capable": false,
                    "Scan Direction": "Forward",
                    "Index Name": "bench_usage_candidate",
                    "Relation Name": "usage_logs",
                    "Alias": "usage_logs",
                    "Startup Cost": 0.56,
                    "Total Cost": 8565.02,
                    "Plan Rows": 64812,
                    "Plan Width": 28,
                    "Actual Startup Time": 0.118,
                    "Actual Total Time": 6.547,
                    "Actual Rows": 51962.0,
                    "Actual Loops": 3,
                    "Disabled": false,
                    "Index Cond": "((tenant_id = 'e000342e-22c2-b525-5299-b35c4d538065'::uuid) AND (created_at >= '2026-09-05 00:00:00+00'::timestamp with time zone) AND (created_at < '2026-10-05 00:00:00+00'::timestamp with time zone))",
                    "Rows Removed by Index Recheck": 0,
                    "Heap Fetches": 0,
                    "Index Searches": 1,
                    "Shared Hit Blocks": 80406,
                    "Shared Read Blocks": 0,
                    "Shared Dirtied Blocks": 0,
                    "Shared Written Blocks": 0,
                    "Local Hit Blocks": 0,
                    "Local Read Blocks": 0,
                    "Local Dirtied Blocks": 0,
                    "Local Written Blocks": 0,
                    "Temp Read Blocks": 0,
                    "Temp Written Blocks": 0,
                    "Workers": []
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  },
  "Planning": {
    "Shared Hit Blocks": 0,
    "Shared Read Blocks": 0,
    "Shared Dirtied Blocks": 0,
    "Shared Written Blocks": 0,
    "Local Hit Blocks": 0,
    "Local Read Blocks": 0,
    "Local Dirtied Blocks": 0,
    "Local Written Blocks": 0,
    "Temp Read Blocks": 0,
    "Temp Written Blocks": 0
  },
  "Planning Time": 0.08,
  "Triggers": [],
  "Execution Time": 20.778
}
```

## All five execution-time samples

```json
[
  {
    "variant": "A",
    "case": "largest / 30d",
    "ms": 88.832,
    "scan": "Parallel Seq Scan",
    "hit": 9237,
    "read": 56554,
    "index_bytes": 0,
    "times_ms": [
      92.884,
      140.334,
      77.629,
      87.022,
      88.832
    ],
    "insert_ms": 990.5307909939438
  },
  {
    "variant": "A",
    "case": "median / 30d",
    "ms": 90.031,
    "scan": "Parallel Seq Scan",
    "hit": 9579,
    "read": 56272,
    "index_bytes": 0,
    "times_ms": [
      90.031,
      96.235,
      91.502,
      59.511,
      62.466
    ],
    "insert_ms": 990.5307909939438
  },
  {
    "variant": "A",
    "case": "largest / 90d",
    "ms": 87.242,
    "scan": "Parallel Seq Scan",
    "hit": 11775,
    "read": 54016,
    "index_bytes": 0,
    "times_ms": [
      87.607,
      84.074,
      93.655,
      87.242,
      86.231
    ],
    "insert_ms": 990.5307909939438
  },
  {
    "variant": "B",
    "case": "largest / 30d",
    "ms": 64.716,
    "scan": "Parallel Bitmap Heap Scan",
    "hit": 53,
    "read": 61487,
    "index_bytes": 121856000,
    "times_ms": [
      84.425,
      62.882,
      67.513,
      63.782,
      64.716
    ],
    "insert_ms": 1189.061624929309
  },
  {
    "variant": "B",
    "case": "median / 30d",
    "ms": 0.038,
    "scan": "Bitmap Heap Scan",
    "hit": 39,
    "read": 0,
    "index_bytes": 121856000,
    "times_ms": [
      0.195,
      0.038,
      0.041,
      0.036,
      0.03
    ],
    "insert_ms": 1189.061624929309
  },
  {
    "variant": "B",
    "case": "largest / 90d",
    "ms": 86.582,
    "scan": "Parallel Seq Scan",
    "hit": 15988,
    "read": 49803,
    "index_bytes": 121856000,
    "times_ms": [
      87.196,
      86.555,
      90.067,
      84.482,
      86.582
    ],
    "insert_ms": 1189.061624929309
  },
  {
    "variant": "C",
    "case": "largest / 30d",
    "ms": 20.778,
    "scan": "Parallel Index Only Scan",
    "hit": 80422,
    "read": 0,
    "index_bytes": 235782144,
    "times_ms": [
      18.959,
      22.769,
      20.778,
      17.504,
      22.073
    ],
    "insert_ms": 1650.3676250576973
  },
  {
    "variant": "C",
    "case": "median / 30d",
    "ms": 0.024,
    "scan": "Index Only Scan",
    "hit": 25,
    "read": 0,
    "index_bytes": 235782144,
    "times_ms": [
      0.055,
      0.025,
      0.024,
      0.02,
      0.018
    ],
    "insert_ms": 1650.3676250576973
  },
  {
    "variant": "C",
    "case": "largest / 90d",
    "ms": 41.571,
    "scan": "Parallel Index Only Scan",
    "hit": 241829,
    "read": 0,
    "index_bytes": 235782144,
    "times_ms": [
      50.909,
      41.571,
      41.428,
      41.232,
      44.931
    ],
    "insert_ms": 1650.3676250576973
  },
  {
    "variant": "D",
    "case": "largest / 30d",
    "ms": 67.182,
    "scan": "Parallel Seq Scan",
    "hit": 2016,
    "read": 63775,
    "index_bytes": 121520128,
    "times_ms": [
      63.606,
      64.887,
      68.423,
      69.034,
      67.182
    ],
    "insert_ms": 1180.3094581700861
  },
  {
    "variant": "D",
    "case": "median / 30d",
    "ms": 4.346,
    "scan": "Index Scan",
    "hit": 2501,
    "read": 0,
    "index_bytes": 121520128,
    "times_ms": [
      6.821,
      4.546,
      4.346,
      4.005,
      3.918
    ],
    "insert_ms": 1180.3094581700861
  },
  {
    "variant": "D",
    "case": "largest / 90d",
    "ms": 85.875,
    "scan": "Parallel Seq Scan",
    "hit": 3180,
    "read": 62611,
    "index_bytes": 121520128,
    "times_ms": [
      85.685,
      85.018,
      86.917,
      85.875,
      88.268
    ],
    "insert_ms": 1180.3094581700861
  }
]
```
