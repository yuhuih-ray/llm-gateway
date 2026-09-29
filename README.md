# LLM Gateway

A learning-oriented FastAPI project. Provides a health endpoint and a minimal non-streaming chat completion API.

## Run locally

Requires Python 3.12. From the project directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m uvicorn llm_gateway.main:app --reload
```

Visit `http://127.0.0.1:8000/health` to get `{"status":"ok"}`.

## Run tests

With the virtual environment activated:

```sh
python -m pytest
```

## Development checks

With the development dependencies installed and the environment activated:

```sh
ruff check .
ruff format --check .
mypy src
python -m pytest
```

To apply formatting, run `ruff format .`.

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
