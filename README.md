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
Flash allows 12 seconds to first streaming text (or the whole non-streaming call);
Pro allows 28 seconds. Both allow at most 1 second between streaming events.
These values are three times measured maxima, rounded up, from a small sample
on 2026-09-30 using the default output budget.

| Gateway model | Upstream model |
| --- | --- |
| `gemini-flash` | `gemini-3.8-flash` |
| `gemini-pro` | `gemini-3.1-pro-preview` |
| `fake` | Deterministic local provider |

Both Gemini IDs were verified with real generation requests. Pro is a preview
model that may retire on short notice; run the live tests to detect availability changes.
Use one of these gateway names in the authenticated curl example. Requests
require at least one message and optionally accept positive `max_tokens` and
`temperature` from 0 to 2. Responses include token `usage` and keep the requested
gateway model name. Responses are non-streaming by default; no usage records are written.

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

Thinking is fixed in the registry: Flash uses `LOW`; Pro uses its model default.
Changing this policy changes quality, latency, and cost. `max_tokens` is used
as the output token limit after applying the registry default and cap and includes internal thinking tokens,
matching OpenAI reasoning-model semantics. A small limit can be exhausted before
visible text is generated. `usage.completion_tokens` includes visible and thinking
tokens, with thinking reported in `completion_tokens_details.reasoning_tokens`.
Terminal reasons are `stop`, `length` (token limit), or `content_filter` (safety).

Gemini requests default to 1024 output tokens when `max_tokens` is omitted;
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
upstream iterator. No usage records are written.
