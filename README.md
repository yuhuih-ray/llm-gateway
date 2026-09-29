# LLM Gateway

A learning-oriented FastAPI project. The first milestone provides a health endpoint.

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
