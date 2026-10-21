# Gateway k6 load-test tooling

This directory contains tooling only. Running `run.sh` deliberately generates load;
setup and inspection do not. No benchmark results are checked in.

## Prerequisites

- Docker with Compose v2 and a running Docker daemon.
- Python 3.12 and uv; from the repository root run `uv sync`.
- k6: `brew install k6` on macOS.
- Free localhost ports: PostgreSQL **15432**, queue Redis **16379**, cache Redis
  **16380**, gateway **18000**. Do not repoint these scripts at dev services.

The fixed Compose project `llm-gateway-loadtest` has its own three volumes. Its
PostgreSQL database/user are `gateway_loadtest`. Setup generates a random database
password and 50 tenants with one API key each, no monthly budgets, and extremely
high request/provider/token limits. The real limiter still runs. No Gemini key is
needed. State, keys, logs, and smoke output live in the gitignored `.state/`
directory with private credentials files. No dev `.env` is loaded by the processes.

## Exact commands

From the repository root:

```sh
uv sync
brew install k6
./scripts/loadtest/setup.sh
./scripts/loadtest/run.sh s1
./scripts/loadtest/run.sh s2
./scripts/loadtest/run.sh s3
./scripts/loadtest/teardown.sh
```

Run commands sequentially. A directory lock prevents concurrent runs, setup, or
teardown. After a forcibly killed shell, verify that its gateway, ARQ worker, and
k6 process have exited before removing the stale `.state/lock` directory.
Teardown removes only this Compose project's containers and volumes and invalidates
its local keys; it does not remove dev containers or volumes.

For the small end-to-end verification instead of the full scenarios:

```sh
./scripts/loadtest/setup.sh
./scripts/loadtest/run.sh s1 --smoke
./scripts/loadtest/teardown.sh
```

Smoke mode schedules **20 RPS for 10 seconds**, with no other s1 plateaus, and writes
only to `.state/smoke/`. It uses the same launch, authentication, limiter, queue,
accounting, summary, metadata, and shutdown path. Never commit its output.

## Scenarios

| Scenario | Arrival schedule | FAKE_DELAY_MS | SERVER_TIMING |
| --- | --- | --- | --- |
| s1 | 30-second warm-up at 100 RPS, then 100, 200, 400, 800, 1600 RPS; five consecutive 60-second plateaus | 0 | false |
| s2 | 200 RPS for 60 seconds | 0 | true |
| s3 | 200 RPS for 120 seconds | 2000 | false |

Every plateau uses `constant-arrival-rate`, not a closed VU loop. Closed loops
silently reduce offered RPS when responses slow down and hide overload. The scripts
make one non-streaming fake completion per iteration, assign keys round-robin using
the scenario-wide iteration index, and omit `X-Gateway-Cache`. `temperature: 0`
alone does not opt into caching. No per-iteration sleep is added.

s1 retains a latency Trend for each plateau. s2 records `server_auth_ms`,
`server_ratelimit_ms`, `server_budget_ms`, `server_cache_ms`, `server_provider_ms`,
and `server_total_ms` Trends and checks that every response contains all stages.
s3 reserves 600 VUs for roughly 400 concurrent requests implied by 200 RPS and a
2-second provider delay. Each scenario has a finite VU ceiling and up to 30 seconds
of graceful completion time; inspect `dropped_iterations` and generator resource
usage before interpreting gateway saturation. Non-200 responses, unexpected cache
hits, missing timing fields in s2, and dropped iterations fail k6 thresholds.

## Lifecycle and output

`run.sh` refuses non-loadtest endpoints and an occupied gateway port. It checks for
leftover ARQ work, deletes only `ratelimit:*`, `budget:*`, and `cache:*` keys from
the isolated Redis instances, and captures the initial usage-row count. It launches
one gateway worker without reload, with WARNING logging and no access log, plus one
ARQ worker. Both use the dedicated database/Redis URLs. Child processes are stopped
on completion or shell interruption; the Compose services remain until teardown.

Each k6 invocation explicitly uses `--summary-export`:

- `docs/benchmarks/gateway-load/s1.json` and `s1-meta.json`
- `docs/benchmarks/gateway-load/s2.json` and `s2-meta.json`
- `docs/benchmarks/gateway-load/s3.json` and `s3-meta.json`

Generated JSON is gitignored and overwritten on the next matching scenario run.
Logs are `.state/<scenario>-gateway.log`, `-worker.log`, and `-k6.log`. Scripts print
paths rather than dumping logs or credentials. A nonzero k6 exit is preserved after
metadata capture and cleanup.

Metadata records requests sent and successful, the **usage-row delta** rather than
all historic rows, ARQ queue length immediately after k6, and queue length plus
in-progress jobs after a bounded 30-second drain observation. ARQ's queue is a sorted
set (`ZCARD`), not a list. `drained=false` or a row deficit is preserved, not hidden;
failed requests, dropped usage, and a slow worker must be distinguished. Before a
new run, unresolved prior jobs require draining or teardown/setup so they cannot
contaminate the next row delta. A dropped k6 iteration was never sent and is not
part of `requests_sent`.

## Gateway timing semantics

`FAKE_DELAY_MS` defaults to 0 and adds one asynchronous delay per fake completion
(including once before the first streamed event). Existing simulated per-chunk
stream pacing is separate. It does not delay other providers.

`SERVER_TIMING` defaults to false. When true, `/v1/chat/completions` responses have
a `Server-Timing` header with fixed names `auth`, `ratelimit`, `budget`, `cache`,
`provider`, and `total`; durations are milliseconds measured with `time.perf_counter`.
No keys, tenant IDs, prompts, or descriptions enter this header. Disabled timing
does not read the performance clock. Skipped stages report zero; cache read/write
times accumulate when applicable.

`total` runs from ASGI entry to response headers, including parsing/routing and
serialization overhead not assigned to a named stage. It excludes network body
transfer and post-response token/budget/usage accounting. For SSE, `provider` covers
stream setup through the first upstream event and `total` ends at the SSE headers;
neither claims to measure full stream duration. Post-delivery SSE cache writes also
cannot appear in an already-sent header. The k6 scenarios use non-streaming
responses. Timing is diagnostic instrumentation and adds some measurement overhead.

The s1 warm-up is excluded from all `latency_rps_*` stage trends. Overall request
counters and usage-row accounting include it. The 10-second smoke mode skips warm-up.
