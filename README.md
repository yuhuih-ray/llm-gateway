# LLM Gateway

A learning-oriented FastAPI project. Provides a health endpoint and a minimal non-streaming chat completion API.

## Start the gateway

Install Docker with Docker Compose, copy `.env.example` to `.env` if you have not
configured it yet, and set the database credentials and any provider keys there.
From the project directory:

```sh
docker compose up -d --wait
```

Compose builds one Python 3.12 image with runtime dependencies only, runs database
migrations, and starts the gateway and ARQ usage worker as a non-root user.
The gateway is published only on `127.0.0.1:8000`; visit
`http://127.0.0.1:8000/health/live` to get `{"status":"ok"}`.
Container database and Redis URLs use Compose service names. Other settings,
including `VLLM_BASE_URL`, come from `.env`. A provider on the host must use a
container-reachable address (for example `host.docker.internal` on Docker Desktop).

After changing application code, rebuild with
`docker compose up -d --build --wait`. Inspect services with `docker compose ps`
and worker output with `docker compose logs worker`.

### Host-run development alternative

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python 3.12 is pinned in `.python-version`. Keep the localhost database/Redis URLs
in `.env`; Compose overrides them only inside containers. Stop any containerized
gateway/worker before starting host processes, then use:

```sh
docker compose stop gateway worker
docker compose up -d --wait postgres redis redis-cache
uv sync
uv run alembic upgrade head
uv run uvicorn llm_gateway.main:app --reload --no-proxy-headers
```

In a second terminal, using the same `.env`:

```sh
uv run arq llm_gateway.worker.WorkerSettings
```

## Run tests

Docker must be running. Integration tests create a throwaway PostgreSQL 18
container and apply Alembic migrations; they never use the Compose database.
From the project directory:

```sh
uv run pytest
```

To skip Docker-dependent integration tests:

```sh
uv run pytest -m "not integration"
```

## Development checks

After `uv sync` (which includes development dependencies):

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

To apply formatting, run `uv run ruff format .`.

## Chat completions

With the server running:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"fake","messages":[{"role":"user","content":"Hello"}]}'
```

Messages support `system`, `user`, and `assistant` roles with string content.
The local FakeProvider echoes the model and always returns:

```json
{"id":"chatcmpl-fake","model":"fake","choices":[{"message":{"role":"assistant","content":"Hello from FakeProvider."},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"completion_tokens_details":{"reasoning_tokens":0}}}
```

The fake ID is fixed, not unique. The `fake` model makes no external calls and reports zero usage.

## Local infrastructure

Compose includes PostgreSQL, the durable main Redis, a disposable cache Redis,
the migration job, gateway, and usage worker. The example credentials are
placeholders for local development only.

```sh
# Only copy this if .env does not already exist:
cp .env.example .env
docker compose up -d --wait
docker compose ps
```

Published services bind to localhost. PostgreSQL and the main Redis store data
in named volumes; cache Redis has no persistence.
Stop the services while preserving data:

```sh
docker compose down
```

`docker compose down -v` deletes all data in these volumes.

## Database migrations

With `.env` configured as above, `docker compose up -d --wait` runs
`alembic upgrade head` in the one-shot `migrate` service before the gateway or
worker starts. To apply migrations manually during host development:

```sh
docker compose up -d --wait postgres
uv run alembic upgrade head
```

To remove all application tables and their data:

```sh
uv run alembic downgrade base
```

Migrations read `DATABASE_URL` from the environment or `.env`. Importing the app
does not require configuration.

### Checking for invalid indexes

After a failed concurrent index build, check for invalid indexes:

```sql
SELECT indexrelid::regclass FROM pg_index WHERE NOT indisvalid;
```

Drop the failed index before retrying the migration. For the usage report index,
run the following outside a transaction, then rerun `uv run alembic upgrade head`:

```sql
DROP INDEX CONCURRENTLY public.ix_usage_logs_tenant_id_created_at;
```

## API keys

With Compose running, create a tenant and key:

```sh
docker compose exec gateway python -m llm_gateway.cli create-tenant local
docker compose exec gateway python -m llm_gateway.cli create-key --tenant local --name development
```

For host-run development, replace `docker compose exec gateway python` with
`uv run python` in these CLI commands.

The CLI prints the full key once. Store it securely and set `GATEWAY_API_KEY`
in your shell to use the authenticated curl example above. Only the key prefix
and SHA-256 hash are stored. `/health` remains public.

Revoke a key using its `gw_<id>` prefix:

```sh
docker compose exec gateway python -m llm_gateway.cli revoke-key 'gw_<id>'
```

## Gemini models

Set `GEMINI_API_KEY` in your ignored `.env` file. The gateway creates the SDK
client only when a Gemini request is made. Timeouts are fixed per model in the registry:
Flash and Flash-Lite allow 60 seconds to first streaming text (or the whole non-streaming call);
Pro allows 120 seconds. All allow at most 30 seconds between streaming events.
These deadlines budget for the configured cap (roughly 1.5 times cap divided by
measured throughput), not short sample maxima. An idle timeout after the commit
point cannot be retried.

| Gateway model | Upstream model |
| --- | --- |
| `gemini-flash-lite` | `gemini-3.1-flash-lite` |
| `gemini-flash` | `gemini-3.8-flash` |
| `gemini-pro` | `gemini-3.1-pro-preview` |
| `fake` | Deterministic local provider |

Both Gemini IDs were verified with real generation requests. Pro is a preview
model that may retire on short notice; run the live tests to detect availability changes.
Use one of these gateway names in the authenticated curl example. Requests
require at least one message and optionally accept positive `max_tokens` and
`temperature` from 0 to 2. Responses include token `usage` and identify the gateway model that answered. Responses are non-streaming by default; usage records are queued for the ARQ worker.

Connection errors, connect timeouts, and HTTP 429/5xx get at most
two retries with jittered exponential backoff. Read timeouts and gateway deadlines
return 504 without retrying potentially billed generation. After SSE headers are
sent, timeouts instead terminate the stream with an error event; no retry occurs.
For explicit model names, fallback is disabled unless the caller sends `X-Gateway-Fallback: on`.
The registry currently permits `local-qwen` to fall back to `gemini-flash-lite`;
other models have no fallback targets. An explicitly named model must not silently
change: callers may require local data handling or fixed behavior. Opting in allows
the prompt to go to the configured cloud target and incur its normal charges.
`model="auto"` enables fallback by default, as described below.

Before the first event/full completion, connection failures, timeouts, upstream
429/5xx, concurrency overload, or an open circuit can select a fallback. Validation
and credential errors and `content_filter` never do. After the first event there
is no fallback. Each target gets its own input/output limits, provider rate bucket,
budget check, and concurrency slot; the tenant request limit is consumed once.
An exhausted paid target returns 429 `insufficient_quota` (cached answers remain
free). Cache entries stay isolated by the actual model.

`X-Gateway-Model` and the response `model` identify the answering model;
`X-Gateway-Fallback-From` is present only when fallback answered. Usage keeps the
original `model_requested`, actual `model_selected` and `upstream_model`, and
`router_info` with `fallback_from` and a sanitized `fallback_reason`.

Each provider, including Gemini, has a per-process circuit breaker. Five consecutive
failed calls (after their retries) open it for 30 seconds. Connection errors,
timeouts, and 5xx count; 4xx and slot overload do not. One half-open probe is allowed;
a completion or first streaming event closes the circuit, and a failed probe
reopens it. Without fallback, an open circuit returns 503 `upstream_unavailable`
with `Retry-After`. Circuit transitions are logged at WARNING.

`REQUEST_DEADLINE_SECONDS` defaults to 120. It bounds the entire request from
arrival through the first streaming event or full completion, including auth,
retries, backoff, slot waits, and all fallback attempts. Each wait is limited by
the remaining budget; expiration returns 504 `Request deadline exceeded`.
After the streaming commit point, the registry idle timeout applies instead.

### Automatic model routing

Send `"model": "auto"` with `X-Gateway-Quality: economy`, `balanced`, or `high`.
The default is `ROUTER_DEFAULT_QUALITY=balanced`; other quality values return 400
for auto requests. The router loads `src/llm_gateway/router_artifacts/three-tier.json`
once, lazily, and reuses it. Set `ROUTER_ARTIFACT` to override its path. A missing
or invalid artifact returns 503 `router_unavailable` for auto requests; explicit
model names continue to work. Restart the process to replace a loaded artifact.

Routing features use **only the last user message's text**, concatenating text
parts in order. System instructions, previous messages, and a trailing assistant
message are not router inputs. Providers still receive the complete conversation.
Messages accept strings or arrays of `{"type": "text", "text": "..."}` parts;
other content types are unsupported. Auto requires at least one user message.

Before routing, models whose input limits cannot accommodate the full conversation
are removed. The routed model is tried first, followed by every more expensive
eligible model in artifact order: `local-qwen`, `gemini-flash-lite`, `gemini-flash`.
Fallback is automatic under the existing pre-commit failure rules, without an
`X-Gateway-Fallback` opt-in. A full local concurrency slot immediately escalates
with reason `upstream_overloaded`; auto does not wait for that slot. Each target
still passes its provider rate limit, budget, and concurrency checks. Choose an
explicit model when prompts must stay local or behavior must remain fixed.

The response `model` and `X-Gateway-Model` name the actual answering model. Usage
keeps `model_requested="auto"`, the actual `model_selected`, and `router_info` with
`artifact_run`, `quality`, `tau`, per-model `probabilities`, and `routed_model`.
Fallback adds `fallback_from` (the initial routed model) and `fallback_reason`.

Measured results on 651 held-out benchmark questions are shown below. The
all-`gemini-flash` baseline had 94.62% accuracy and 4.78 s mean latency. Costs and
latencies come from recorded labels and exclude router overhead and live load.

| Quality | Tau | Accuracy | Accuracy retained | Cost reduction vs Flash | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 0.92 | 94.47% | 99.84% | 15.05% | 3.95 s |
| balanced | 0.89 | 93.39% | 98.70% | 35.31% | 3.03 s |
| economy | 0.74 | 87.86% | 92.86% | 76.63% | 2.54 s |

Source: [three-tier per-level evaluation](docs/benchmarks/router/results/three-tier-levels.json).
The router was trained on benchmark questions, not real traffic, and does not
generalize its quality targets to unseen domains. On the 246-question OOD split
(law/history), high/balanced/economy accuracy was **82.52% / 79.67% / 72.76%**, versus
82.93% for all-Flash. Economy lost 10.16 percentage points despite its 8-point
train allowance; balanced lost 3.25 points despite a 3-point allowance. See the
[`ood` results](docs/benchmarks/router/results/three-tier-levels.json). These levels
are measured tradeoffs, not accuracy guarantees for arbitrary conversations.

Default tests and CI exclude live API calls. To run the live test, export
`GEMINI_API_KEY` in your shell and run:

```sh
uv run pytest -m live
```

Without that environment variable, the live test skips. The live test sends a
short prompt to Gemini and may incur API charges.

Thinking is fixed in the registry: Flash-Lite requests disabled thinking with
`thinking_budget=0`; Flash and Pro use `LOW`. Flash accepted budget zero but still
reported reasoning on a math probe, so it uses its lowest accepted explicit level.
Each model/thinking pair is a distinct configuration shared by training and serving;
a thinking variant can later have its own gateway model name.
Changing this policy changes quality, latency, and cost. `max_tokens` is used
as the output token limit after applying the registry default and cap and includes internal thinking tokens,
matching OpenAI reasoning-model semantics. A small limit can be exhausted before
visible text is generated. `usage.completion_tokens` includes visible and thinking
tokens, with thinking reported in `completion_tokens_details.reasoning_tokens`.
Terminal reasons are `stop`, `length` (token limit), or `content_filter` (safety).

Gemini requests default to 2048 output tokens when `max_tokens` is omitted;
values above 8192 are clamped to 8192. These limits include billed thinking
tokens. The `fake` model has no token limit. Settings are cached until process
restart, and a lazily created Gemini async client is reused until app shutdown.

## Streaming chat completions

Use `stream: true` and `curl -N` to receive Server-Sent Events as text arrives:

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-flash","messages":[{"role":"user","content":"Explain HTTP streaming in a paragraph."}],"stream":true,"stream_options":{"include_usage":true}}'
```

The stream sends an assistant role chunk, content deltas, a finish chunk, and
`data: [DONE]`. With `include_usage`, a chunk with empty `choices` and final usage
precedes `[DONE]`; usage can be null if the provider does not report it.
Gateway token defaults and caps apply identically to both response modes.

Retries occur only before the first provider event and before HTTP headers are
sent. After streaming starts, failures or idle timeouts produce one sanitized
error event and close the stream without `[DONE]`. Disconnecting closes the
upstream iterator. Usage records are queued for the ARQ worker.

## Usage recording worker

The worker starts with the rest of the stack:

```sh
docker compose up -d --wait
```

It uses the same image and `.env` as the gateway, waits for migrations and Redis,
and restarts unless stopped. For separate host processes, see the development
alternative above. Usage batch jobs have an explicit 30-day expiration, rather
than ARQ's one-day default, so a worker outage does not discard a day's backlog.
Queue age in readiness helps detect a stalled worker before that retention limit.

Each chat request returns `X-Request-ID`. Requests reaching a provider synchronously
append a record with status `success`, `error`, or `cancelled` to a per-process
buffer. Authentication, validation, and model-lookup rejections do not record usage.
Unknown token counts and cost remain null. The gateway calculates Decimal USD cost
including reasoning tokens; the worker inserts batches in one statement, deduplicating
by `(request_id, created_at)` and preserving request time. Previously queued
`write_usage` jobs remain supported.

`ACCOUNTING_BUFFER_SIZE` defaults to 10,000 records, including batches being retried.
The flusher wakes every 100 ms or at 500 waiting records and processes up to 500 per
batch. It aggregates token charges per tenant and exact integer spend per tenant/UTC
month in one Lua call (including 80% warning checks), then publishes one
`write_usage_batch` job and deletes one batch receipt. These three client calls
replace per-tenant Redis calls. The receipt and stable job ID protect retries from
duplicate charges. Accounting never waits on the request path.
Redis failures retain records with backoff capped at 2 seconds and one warning per
30 seconds. At capacity, new records are dropped, the dropped counter increases,
and ERROR `usage_record_dropped` includes the full record for recovery. Shutdown
allows 5 seconds to drain, then cancels and awaits the flusher before logging
`accounting_shutdown_unflushed count=N` if needed.

A gateway crash normally loses only the unflushed ~100 ms window; outages or event-loop
stalls can enlarge that window up to the buffer cap. Redis AOF `everysec` can lose
roughly one second. This is best-effort accounting, not lossless billing across
crashes. The worker retries transient database failures up to five tries, logging
`usage_record_failed` per record on final failure; integrity errors are not retried.

Prices use the official standard paid text rates verified on 2026-10-04, even if
an upstream account has a free allowance. Flash's $0.75/$3.75 per-million rates
expire December 31, 2026 and must be reverified before 2027. Pro uses $4/$18 when
prompt tokens exceed 200,000. Registry comments cite the official pricing page.
Downgrading the usage migration retains cancelled rows as errors because the
previous schema has no cancelled state.

## API key authentication cache

Verified API keys are cached in the main Redis for 60 seconds under
`auth:<SHA-256 of the complete key>`. Values contain tenant/key IDs, tenant status,
and the monthly budget as a decimal string (or null). Neither secrets nor prefixes
are cache keys. Unknown keys and wrong secrets are never cached. Reads and fills
have a 100 ms timeout; Redis errors fall back to the database.

After committing, admin API and CLI revocations write a 60-second tombstone.
Fills use `SET NX` so a delayed lookup cannot overwrite that tombstone. Tenant
suspension writes tombstones for every key with a status marker preserving 403;
key revocation returns 401. Reactivation and budget changes delete all of the
tenant's auth entries. Operator commands are:

```sh
uv run python -m llm_gateway.cli suspend-tenant --tenant example
uv run python -m llm_gateway.cli reactivate-tenant --tenant example
uv run python -m llm_gateway.cli set-budget --tenant example --usd 25.00
```

Invalidation errors log ERROR `auth_cache_invalidation_failed` without rolling
back the DB change. Cached authorization or budget data can remain stale until
its 60-second TTL expires; in-flight lookups may also briefly use their earlier
DB snapshot. Direct database changes bypass invalidation and have the same TTL
bound. Use the CLI/admin mutation paths for immediate invalidation.

Failed `/v1` authentications consume one IP token per 401: capacity 20, refill one
per 3 seconds. Before an uncached DB lookup, a token is reserved and refunded for
non-401 outcomes, including 403, preventing concurrent failures from bypassing the
limit. An exhausted bucket returns 429 before DB work. Valid cache hits still work
without DB lookup. IPs use the existing trusted-proxy rules and Redis outages use
the bounded local rate-limit fallback.

## Human admin API

Human users authenticate with passwords and short-lived JWTs. Gateway API keys
only authenticate `/v1/chat/completions`; they cannot authenticate admin or login
requests. JWTs cannot authenticate gateway completions.

Set `JWT_SECRET` in `.env` to a randomly generated secret of at least 32 bytes
(for example, generate one with `uv run python -c "import secrets; print(secrets.token_hex(32))"`).
Keep it private. Missing or short secrets make admin authentication return 503;
the app and gateway can still start. Tokens expire after 15 minutes; there are
no refresh tokens or logout endpoint. Role changes and user disablement take
effect on the next authenticated request because authorization reads the database.

| Endpoint | Viewer | Member | Admin |
| --- | --- | --- | --- |
| `POST /auth/login` (email/password) | Yes | Yes | Yes |
| `GET /admin/keys` | Yes | Yes | Yes |
| `POST /admin/keys` | No | Yes | Yes |
| `POST /admin/keys/{id}/revoke` | No | No | Yes |
| `GET /admin/usage` | Yes | Yes | Yes |

Apply migrations and create a user in an existing tenant. The CLI prompts for
the password without echoing it; never pass a password as a command-line argument.

```sh
docker compose up -d --wait
docker compose exec gateway python -m llm_gateway.cli create-user --tenant local --email admin@example.com --role admin
```

Then log in and save the token locally. These examples use `jq`;
credentials are passed to curl through standard input rather than process arguments.

```bash
read -r -s -p "Password: " PASSWORD; printf '\n'
LOGIN=$(printf '%s' "$PASSWORD" | jq -Rs '{email:"admin@example.com", password:.}' |
  curl -sS http://localhost:8000/auth/login -H 'Content-Type: application/json' --data-binary @-)
unset PASSWORD
TOKEN=$(printf '%s' "$LOGIN" | jq -r .access_token)
unset LOGIN

# Send the bearer header via stdin, keeping it out of curl's argv.
admin_curl() {
  printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" |
    curl -sS --config - "$@"
}
admin_curl http://localhost:8000/admin/keys
admin_curl http://localhost:8000/admin/keys \
  -H 'Content-Type: application/json' --data '{"name":"local-client"}'
admin_curl http://localhost:8000/admin/usage
unset TOKEN
```

Save the `key` from the create response securely: it is returned once. Lists and
revocation responses never include the full key or its hash. Revoking an already
revoked key preserves its original revocation time. Foreign-tenant and missing
key IDs both return 404.

Usage is scoped to the caller's tenant and includes all recorded statuses.
Optional `start` (inclusive) and `end` (exclusive) are timezone-aware ISO 8601
timestamps. The default is the last 30 days; ranges must be positive and at most
90 days. Results contain `totals` and a `models` breakdown. Unknown token/cost
values contribute zero to sums, while every record contributes to request count;
cost strings preserve eight decimal places. Queued records appear after the worker
writes them.

## Monthly usage partitions and retention

`usage_logs` is range-partitioned by UTC `created_at`. Upgrade attaches the old
heap as `usage_logs_legacy` without copying rows. Its upper bound is the first day
of the next UTC month, computed during migration. **Until that cutover, current-month
rows continue to enter legacy**; creating a separate overlapping current-month
partition is impossible. Three monthly partitions from cutover and an empty
`usage_logs_default` partition are created initially.

Stop **all ARQ worker processes** before upgrading or downgrading. Keep the gateway
queue available if desired; queued jobs retain their original request timestamps.
Admin usage reports may be unavailable during preparation because the old table
has been renamed and the new parent is not attached yet.

```sh
# Stop each worker gracefully with Ctrl-C and wait for it to exit.
uv run alembic upgrade head
uv run alembic check
uv run arq llm_gateway.worker.WorkerSettings
```

The hand-written migration uses committed preparation steps for concurrent indexes.
If it fails, keep workers stopped: inspect invalid indexes (see above) and the
migration state, then restore or finish preparation before retrying. Do not blindly
rerun a partly completed rename. Future-dated legacy rows at/after cutover cause a
preflight failure before any rename. Both the primary key `(id, created_at)` and
unique key `(request_id, created_at)` include the partition key because PostgreSQL
has no global unique index across partitions. The parent identity sequence starts
above the legacy high-water mark. Worker retries use the same `created_at` from
the gateway payload and still deduplicate correctly.

The worker runs `ensure_partitions` at startup and daily at 00:05 UTC. It ensures
coverage for the current UTC month and the next two months. Any existing range
that fully contains a month satisfies it, including legacy before cutover; the job
never creates an overlapping monthly range. Existing partitions, including the extra initial month,
are left intact. Maintenance jobs use a shared transaction advisory lock so
multiple workers cannot race each other on partition DDL.

`USAGE_RETENTION_MONTHS` defaults to `12` and must be positive. Daily at 00:15 UTC,
`drop_expired_partitions` computes the first day of the current UTC month minus
that many calendar months. It detaches and drops monthly partitions whose upper
bound is **strictly before** the cutoff, conservatively retaining a partition
whose upper bound equals it. Legacy is dropped only when its complete range has
expired by the same rule. Drops permanently remove those rows and are logged.
The default partition is never dropped by retention.

An ERROR `usage_default_partition_not_empty count=N` requires operator attention:
check the row timestamps and missing partitions. If default contains rows in a
month that needs creating, maintenance logs `usage_partition_creation_blocked`
and leaves those rows untouched; it does not silently move or delete data. Resolve
the misplaced rows during controlled maintenance, then rerun partition creation.

Downgrade restores the legacy heap without copying rows. It refuses with a clear
error if monthly/default partitions contain any rows, legacy has already been
removed by retention, or legacy contains duplicates that violate the old unique
keys. Resolve those conditions first; downgrade never discards those records.

## Liveness and readiness

- `GET /health/live` returns `200 {"status":"ok"}` without checking dependencies.
  Use it for orchestrator **restart** decisions. `/health` remains an alias.
- `GET /health/ready` runs PostgreSQL `SELECT 1` in a short-lived session with a
  1-second timeout. PostgreSQL failure returns 503; success returns 200. Use this
  endpoint for **traffic routing**, so an outage removes instances from traffic
  without restarting them all.
- Readiness checks the existing usage queue pool and its ARQ sorted set within a
  0.5-second timeout. A missing or failing queue reports `unavailable` but does not
  fail readiness: requests can still succeed; accounting retries in its buffer.
- The top-level `usage_queue` object reports `depth` (jobs, not usage rows) and
  `oldest_job_age_seconds`. Empty queues report zero; unavailable metrics are null.
  Age is seconds past the oldest ARQ queue score (scheduled Unix milliseconds),
  clamped to zero for future deferred jobs. Deferred retries reset this due-time age.
  Queue depth or age never fails readiness. An age over 10 minutes logs a WARNING
  at most once per minute per process, so operators can detect a stalled worker.

Readiness also returns `status` (`ready` or `not_ready`) and `checks` for
`postgres` and `usage_queue` (`ok` or `unavailable`). Details are logged server-side.
All health endpoints are unauthenticated, never record usage, and never call
Gemini or another model provider.

## Request rate limits

Chat completions use request-based token buckets after authentication and model
lookup, before provider execution or stream headers. Defaults:

| Bucket | Capacity | Refill |
| --- | ---: | ---: |
| Each tenant (all models combined) | 20 requests | 2 requests/second |
| Gemini provider (all tenants and Gemini models combined) | 60 requests | 5 requests/second |

Fake has no provider-wide bucket. Configure `TENANT_REQUEST_CAPACITY`,
`TENANT_REQUEST_REFILL`, `GEMINI_REQUEST_CAPACITY`, and `GEMINI_REQUEST_REFILL` in
the environment. Per-tenant overrides, token-based limits, and login limits are
not included. One atomic Redis script checks all applicable buckets together;
a rejected request consumes no tokens and produces no usage record.

Rejections return JSON HTTP 429 (including streaming requests):

```json
{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","code":"rate_limit_exceeded"}}
```

`Retry-After` is the wait rounded up to whole seconds (minimum 1).
`X-RateLimit-Remaining` is the whole tokens remaining in the most constrained
bucket. Idle Redis hashes expire after approximately twice their refill time.

Redis uses the shared lifespan pool and a 100 ms call deadline. On failure, a
process-local bucket with the same parameters keeps limits bounded instead of
failing requests merely because Redis is down. This is approximate: N gateway
processes can permit up to N times the configured limit during an outage, and
local state resets on process restart. Fallback warnings occur at most once every
30 seconds per process. Readiness still treats Redis as non-critical.

The local fallback retains at most 10,000 buckets per process, evicting the least
recently used. On each fallback check it drops buckets idle for a full
refill-from-empty interval (longer for debt, until fully refilled). Evicted keys
restart full if seen again, so high key
churn can weaken fallback enforcement; the cap protects process memory during
long Redis outages.

### Token and login limits

Each tenant also has a token bucket: `TENANT_TOKEN_CAPACITY=100000` and
`TENANT_TOKEN_REFILL=1000` tokens/second. Admission checks that its balance is
positive in the same atomic call as the request limits, without reserving tokens.
After response delivery, actual prompt plus completion tokens are charged;
completion includes reasoning tokens. The balance can go negative, floored at
minus capacity. A 429's `Retry-After` allows the balance to become positive again.
The debt floor bounds the blocking time from anomalous usage; billing still uses
`usage_logs`. Concurrent admitted requests can consume beyond the current balance.
Unknown counts, such as cancelled generations without final usage, are not charged.
This known gap remains bounded by request limits and the upcoming cost quota.
Charges run in the batching flusher and are retained during Redis failures.
Admission still uses the local fallback; delayed charges allow additional overshoot.

Login checks run before database lookup and Argon2 verification. Each IP gets a
capacity of 10 attempts, refilling one per 6 seconds. Each normalized email gets
capacity 5, refilling one per minute: each attempt reserves one token before the
user lookup and Argon2 verification. A successful login refunds it (capped at
capacity); failed attempts keep the reservation. This prevents parallel attempts
from bypassing the email limit. Redis keys use
SHA-256 of the normalized email. Per-email limits let an attacker temporarily lock
out a victim; these modest limits recover within minutes. Login 429s use the admin
error body `{"detail":"Rate limit exceeded"}`, with `Retry-After` and
`X-RateLimit-Remaining` headers. Both limits remain active locally during Redis
outages, with the same per-process approximation described above.

`TRUSTED_PROXIES` defaults to an empty list. Configure it as a JSON array of IPs or
CIDRs, for example `TRUSTED_PROXIES='["10.0.0.0/24", "::1"]'`. Only when the direct
peer is trusted is the right-most untrusted `X-Forwarded-For` address used. Otherwise
the peer address is used and forwarded headers are ignored. Configure the server
so `request.client.host` retains the direct peer (for Uvicorn, use
`uv run uvicorn llm_gateway.main:app --no-proxy-headers`); proxy header rewriting
before application code would bypass this trust boundary. An untrusted proxy
makes its clients share that proxy's IP limit.

### Input limits and monthly cost budgets

`MAX_REQUEST_BYTES` defaults to 1,048,576 bytes (1 MiB) for every endpoint.
Oversized `Content-Length` is rejected immediately; actual bytes are also counted
for chunked or understated requests before JSON parsing. HTTP 413 uses
`error.type: invalid_request_error`, `error.code: request_too_large`.

Each registry entry has `max_input_chars`, default 100,000, counting all system,
history, and user message contents. This is a character limit, not an estimated
token limit, since tokenization varies by language. After authentication and model
lookup, oversized input returns HTTP 400 with `context_length_exceeded` before
rate limiting or provider invocation. Recorded prompt tokens can calibrate this
limit later.

Operators set budgets using the CLI (after `uv run alembic upgrade head`):

```sh
uv run python -m llm_gateway.cli set-budget --tenant example --usd 25.00
uv run python -m llm_gateway.cli set-budget --tenant example --usd none
```

Amounts are nonnegative USD with at most two decimal places; `none` means unlimited.
Tenant administrators cannot change their budgets through the admin API.
`GET /admin/usage` includes `monthly_budget_usd` (a string or null for unlimited)
and `spent_this_month` (a decimal string), independently of the requested report
range. Spend comes from Redis, falling back to the current month's database sum
when Redis is unavailable.

Requests starting when spend is at or above the budget receive HTTP 429 without
`Retry-After`:

```json
{"error":{"message":"Monthly budget exceeded","type":"insufficient_quota","code":"insufficient_quota"}}
```

In-flight requests finish. Overshoot includes in-flight requests and accounting
waiting in the buffer, each bounded by input length and output caps. Fake requests
cost zero. Cost is added after response delivery as integer units of 1e-8 USD,
never floating point, using the request-start UTC month; counters expire after
about 40 days. A warning fires once per tenant-month at 80% of the budget.

Budget admission reads have a 100 ms deadline. If the check cannot read Redis,
requests are allowed and warnings are throttled to once per 30 seconds per process;
`usage_logs` remains billing truth. Unknown costs cannot be added. The ARQ worker
reconciles budgeted tenants at startup and hourly, setting each counter to the
maximum of its existing value and the month's database sum. This restores counters
after data loss without discarding spend whose usage records are still queued.
During Redis outages or before reconciliation, budgets are best-effort; request
and token admission limits retain their local fallback, with token charges deferred.

## Exact response cache

A semantic cache was [evaluated and rejected](docs/benchmarks/semantic-cache.md).

`CACHE_REDIS_URL` defaults to `redis://localhost:6380/0`. Compose runs a separate
`redis-cache` instance with 256 MB, `allkeys-lru`, AOF disabled, and no persistence.
Queue and rate-limit Redis uses `noeviction` and AOF: those records must never be
evicted. Cached answers are disposable and must be evictable, so the two workloads
cannot share an instance.

Caching requires explicit opt-in: send `X-Gateway-Cache: on` and set
`temperature: 0`. The gateway cannot tell whether a prompt is time-sensitive;
deterministic output does not imply consent to reuse an answer for 24 hours.
The caller who understands the request must opt in. Send
`Cache-Control: no-cache` to bypass both reads and writes. Successful answers with
finish reason `stop` or `length` expire after 24 hours; errors, filtered answers,
and cancelled streams are never stored. Keys hash the tenant, gateway and upstream
models, thinking configuration, full ordered messages, and generation parameters
including the applied token limit. Streaming and non-streaming share cached answers.

`X-Cache: HIT` identifies replayed answers with fresh response IDs; `X-Cache: MISS`
includes bypassed requests. Hits retain original token counts in usage logs with
`cache_hit=true` and zero cost, without charging the token bucket. Request rate
limits still apply. Cache lookup precedes the budget check, so exhausted tenants
can receive free hits. Each cache operation has a 50 ms deadline; failures fall
back to normal generation and warnings are limited to one per 30 seconds.

TODO: Add cache stampede protection for concurrent identical misses.

TODO: admission by in-flight tokens rather than request count

## Self-hosted vLLM

On the GPU machine, install Docker and NVIDIA Container Toolkit, set
`VLLM_API_KEY` to a private bearer key, then start the pinned server image:

```sh
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_API_KEY \
  vllm/vllm-openai:v0.11.0 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen2.5-7b-awq \
  --max-model-len 8192 \
  --generation-config vllm
```

The image and serving flags follow the [vLLM Docker documentation](https://docs.vllm.ai/en/v0.11.0/deployment/docker.html)
and [server arguments](https://docs.vllm.ai/en/v0.11.0/cli/serve.html); the weights
are [Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ).
Set these on the gateway (or in its `.env`):

```sh
VLLM_BASE_URL=http://gpu-host:8000/v1
VLLM_API_KEY='<same-private-bearer-key>'
```

Clients request `model: "local-qwen"`. Neither variable is required to import or
start the gateway; using this provider without configuration returns 502.
The shared async HTTP client is created lazily and closed at shutdown. Connect
attempts time out after 2 seconds; the registry allows 60 seconds to the first
stream event (or complete response), then 30 seconds between stream events.
Connect failures and retryable HTTP statuses use the existing three-attempt
retry policy; read timeouts and streams that have emitted text are not retried.

`VLLM_MAX_CONCURRENCY` defaults to 16 per gateway process. It covers the entire
response, including streamed delivery and disconnect cleanup. Waiting for a slot
is limited to 2 seconds; overload returns 503, `Retry-After: 1`, and error type
`upstream_overloaded`, without usage accounting. Cached answers do not occupy a
provider slot. Gemini has no concurrency gate. More gateway workers multiply the
local concurrency allowance; a cross-process limit would require Redis.

**Measured capacity.** Measured with `vllm bench serve` on the RTX 5070 Ti
(16 GB), vLLM v0.11.0, Qwen2.5-7B-Instruct-AWQ (`awq_marlin`),
`--max-model-len 8192`, and `--gpu-memory-utilization 0.90`.
KV cache: 160,432 tokens. Raw results are in
[docs/benchmarks/serving/throughput/](docs/benchmarks/serving/throughput/).

- Short requests (512 input / 256 output tokens): a single stream takes about
  11 ms/token. At concurrency 16, throughput is about 1,050 output tokens/s with
  p99 time per output token (TPOT) of 17.5 ms; throughput peaks at concurrency 32
  (about 1,510 output tokens/s).
- Long requests (4,096 input / 256 output tokens): concurrency 16 degrades
  (p99 TPOT 63 ms, p99 end-to-end latency about 27 s); concurrency 32 collapses
  (about 80 output tokens/s, about 380 ms/token) without preemptions.
  Suspected cause, unverified: Windows driver system-memory fallback when VRAM
  is overcommitted.

The default of 16 is the largest tested value that avoids the collapse.
Runs reused the same random prompts with prefix caching enabled, so time to first
token (TTFT) after the first run in each sweep is optimistic.

The registry defaults to 1024 output tokens, caps output at 4096, and limits input
to 20,000 characters. Character limits are approximate: the server enforces its
8192-token combined prompt/output context. Prices are zero for marginal local
cost; future routing must also account for local capacity. Opt-in fallback and
per-provider circuit breakers follow the rules above.

Live checks (both response modes, skipped without `VLLM_BASE_URL`):

```sh
uv run pytest -m live_vllm tests/test_vllm_live.py
```

### Router drift monitoring

Automatic requests record unigram `oov_ratio` (unknown token occurrences divided
by all token occurrences; null for tokenless text) and classifier probabilities.
OOV uses the first classifier's vocabulary and tokenizer; cascade vectorizers
are trained on the same questions. The artifact is preloaded off the event loop
at startup when present; a load failure leaves explicit-model requests available.

Compare recent traffic with the manifest's test-split raw `question` reference:

```sh
uv run python scripts/router_monitor/drift.py --since 2026-10-26T00:00:00Z --until 2026-10-27T00:00:00Z --out data/router/drift.json
```

The script uses `DATABASE_URL`, `ROUTER_ARTIFACT` (or `--artifact`), and only `auto`
usage rows matching the artifact run, in the inclusive-start/exclusive-end time
window. It needs at least 50 requests (`--min-requests` changes this). Null or
missing feature values are excluded and sample counts are reported; an entirely
missing feature is an error. PSI uses fixed probability/OOV bins and replaces
empty-bin proportions with 0.0001: below 0.1 is stable, 0.1–0.2 is watch, and
0.2 or above is drift. Exit code 1 signals drift; insufficient data is an error
(exit 2), not a stable result. Allow the usage worker to drain before measuring.
Distribution drift is an alert to investigate, not evidence of accuracy loss.

Optional demo traffic (calls real models and may incur cost):

```sh
GATEWAY_API_KEY=... uv run python scripts/router_monitor/send_batch.py --split test --n 100 --seed 42 --quality balanced
GATEWAY_API_KEY=... uv run python scripts/router_monitor/send_batch.py --split ood --n 100 --seed 42 --quality balanced
```

Both passes send only raw question text. Use the printed start/end timestamps as
the monitoring window; model shares use successful responses as the denominator.
