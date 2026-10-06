# LLM Gateway

A learning-oriented FastAPI project. Provides a health endpoint and a minimal non-streaming chat completion API.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12 is pinned in `.python-version`. From the project directory:

```sh
uv sync
uv run uvicorn llm_gateway.main:app --reload
```

Visit `http://127.0.0.1:8000/health` to get `{"status":"ok"}`.

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

Requires Docker with Docker Compose. This starts only PostgreSQL and Redis;
the gateway continues to run locally with uv and does not connect to them yet.
The example credentials are placeholders for local development only.

```sh
cp .env.example .env
docker compose up -d
docker compose ps
```

Both services bind to localhost. PostgreSQL and Redis store data in named volumes.
Stop the services while preserving data:

```sh
docker compose down
```

`docker compose down -v` deletes all data in these volumes.

## Database migrations

With `.env` configured as above, start PostgreSQL and apply the schema:

```sh
docker compose up -d --wait
uv run alembic upgrade head
```

To remove all application tables and their data:

```sh
uv run alembic downgrade base
```

Migrations read `DATABASE_URL` from the environment or `.env`. The database is
not connected to endpoints yet; importing the app does not require configuration.

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

Start local infrastructure and apply migrations before creating a tenant and key:

```sh
uv run python -m llm_gateway.cli create-tenant local
uv run python -m llm_gateway.cli create-key --tenant local --name development
```

The CLI prints the full key once. Store it securely and set `GATEWAY_API_KEY`
in your shell to use the authenticated curl example above. Only the key prefix
and SHA-256 hash are stored. `/health` remains public.

Revoke a key using its `gw_<id>` prefix:

```sh
uv run python -m llm_gateway.cli revoke-key 'gw_<id>'
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
`temperature` from 0 to 2. Responses include token `usage` and keep the requested
gateway model name. Responses are non-streaming by default; usage records are queued for the ARQ worker.

Connection errors, connect timeouts, and HTTP 429/500/502/503/504 get at most
two retries with jittered exponential backoff. Read timeouts and gateway deadlines
return 504 without retrying potentially billed generation. After SSE headers are
sent, timeouts instead terminate the stream with an error event; no retry occurs.
There is no fallback to another model or provider.

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

Start the infrastructure and apply migrations, then run the gateway and worker
in separate terminals using the same `.env` (`DATABASE_URL` and `REDIS_URL`):

```sh
docker compose up -d --wait
uv run alembic upgrade head
uv run uvicorn llm_gateway.main:app --reload
```

```sh
uv run arq llm_gateway.worker.WorkerSettings
```

Each chat request returns `X-Request-ID`. Requests reaching a provider enqueue a
record with status `success`, `error`, or `cancelled`. Authentication, validation,
and model-lookup rejections do not enqueue records. Unknown token counts and cost
remain null. The gateway calculates Decimal USD cost including reasoning tokens;
the worker only inserts, deduplicating by request ID and preserving request time.

Enqueue attempts have a 0.5-second limit. Failures log `usage_record_dropped` with
the full record for recovery; they do not change the client's response. The worker
retries database failures up to five tries. Redis uses AOF with `everysec`, so a
crash can still lose roughly one second of queued records. This is best-effort
accounting, not a guarantee of lossless billing across gateway process crashes.

Prices use the official standard paid text rates verified on 2026-10-04, even if
an upstream account has a free allowance. Flash's $0.75/$3.75 per-million rates
expire December 31, 2026 and must be reverified before 2027. Pro uses $4/$18 when
prompt tokens exceed 200,000. Registry comments cite the official pricing page.
Downgrading the usage migration retains cancelled rows as errors because the
previous schema has no cancelled state.

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
uv run alembic upgrade head
uv run python -m llm_gateway.cli create-user --tenant local --email admin@example.com --role admin
uv run uvicorn llm_gateway.main:app --reload
```

In another terminal, log in and save the token locally. These examples use `jq`;
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
