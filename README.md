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

From the project directory:

```sh
uv run pytest
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
  -H 'Content-Type: application/json' \
  -d '{"model":"fake-model","messages":[{"role":"user","content":"Hello"}]}'
```

Messages support `system`, `user`, and `assistant` roles with string content.
The local FakeProvider echoes the model and always returns:

```json
{"id":"chatcmpl-fake","model":"fake-model","choices":[{"message":{"role":"assistant","content":"Hello from FakeProvider."},"finish_reason":"stop"}]}
```

The fake ID is fixed, not unique. No external services are called.
