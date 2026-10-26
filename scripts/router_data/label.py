"""Collect one requested gateway model's labels, without cache or fallback opt-in."""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_gateway.registry import MODELS  # noqa: E402
from llm_gateway.usage import calculate_cost  # noqa: E402
from scripts.router_data.common import (  # noqa: E402
    ROOT,
    pending_questions,
    read_labels,
    read_manifest,
    select_questions,
)
from scripts.serving_bench.gsm8k_eval import (  # noqa: E402
    correct_answer,
    extract_answer,
    positive,
)


class ModelMismatchError(Exception):
    pass


def extract_mmlupro(text: str) -> str | None:
    matches = re.findall(r"ANSWER:\s*\(?([A-J])\)?", text, re.I)
    if not matches:
        matches = re.findall(r"answer is \(?([A-J])\)?", text, re.I)
    return matches[-1].upper() if matches else None


def retry_after(value: str | None) -> float:
    if value is None:
        return 1
    try:
        delay = float(value)
        if not 0 <= delay < float("inf"):
            return 1
        return delay
    except ValueError:
        try:
            return max(
                0,
                (
                    parsedate_to_datetime(value) - datetime.now(timezone.utc)
                ).total_seconds(),
            )
        except (ValueError, TypeError, OverflowError):
            return 1


async def request_completion(
    client: httpx.AsyncClient, payload: dict
) -> httpx.Response:
    rate_retries = failures = 0
    while True:
        try:
            response = await client.post("chat/completions", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            failures += 1
            if failures >= 3:
                raise
            await asyncio.sleep(0.5 * 2 ** (failures - 1))
            continue
        if response.status_code == 429 and rate_retries < 20:
            rate_retries += 1
            await asyncio.sleep(retry_after(response.headers.get("Retry-After")))
            continue
        if 500 <= response.status_code < 600:
            failures += 1
            if failures < 3:
                await asyncio.sleep(0.5 * 2 ** (failures - 1))
                continue
        response.raise_for_status()
        return response


async def label_question(
    client: httpx.AsyncClient,  # Gateway only; never call providers directly.
    model: str,  # Registry model used for identity verification and pricing.
    question: dict,  # Exact prompt and gold from the committed manifest.
) -> dict:
    started = time.perf_counter()
    row = dict(
        id=question["id"],
        model=model,
        correct=False,
        pred=None,
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=None,
        latency_ms=0,
        request_id=None,
        error=None,
    )
    try:
        response = await request_completion(
            client,
            {
                "model": model,
                "messages": [{"role": "user", "content": question["prompt"]}],
                "temperature": 0,
                "max_tokens": 1024,
                "stream": False,
            },
        )
        row["request_id"] = response.headers.get("X-Request-ID")
        if response.headers.get("X-Gateway-Model") != model:
            raise ModelMismatchError()
        value = response.json()
        choice = value["choices"][0]
        content = choice["message"].get("content") or ""
        pred = (
            extract_answer(content)
            if question["dataset"] == "gsm8k"
            else extract_mmlupro(content)
        )
        usage = value.get("usage") or {}
        prompt_tokens, completion_tokens = (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
        for count in (prompt_tokens, completion_tokens):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("Invalid token count")
        cost = calculate_cost(MODELS[model], prompt_tokens, completion_tokens)
        row.update(
            pred=pred,
            correct=correct_answer(question["gold"], pred)
            if question["dataset"] == "gsm8k"
            else pred is not None and pred == question["gold"],
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=str(cost) if cost is not None else None,
        )
    except httpx.HTTPStatusError as exc:
        row["error"] = f"HTTP{exc.response.status_code}"
        row["request_id"] = exc.response.headers.get("X-Request-ID")
    except Exception as exc:
        # Persist sanitized error names, never credentials or response bodies.
        # Cancellation is a BaseException and propagates without a bogus label.
        row["error"] = type(exc).__name__
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return row


async def collect(args: argparse.Namespace) -> None:
    questions = select_questions(read_manifest(), args.per_dataset)
    path = args.out_dir / f"{args.model}.jsonl"
    latest = read_labels(path)
    if any(row["model"] != args.model for row in latest.values()):
        raise ValueError("Existing label file contains another model")
    pending = pending_questions(questions, latest)
    skipped = len(questions) - len(pending)
    semaphore = asyncio.Semaphore(args.concurrency)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=httpx.Timeout(180, connect=2),
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:
        with path.open("a") as output:

            async def one(question: dict) -> dict:
                async with semaphore:
                    row = await label_question(client, args.model, question)
                # No await between append and flush, so concurrent tasks cannot interleave lines.
                output.write(json.dumps(row) + "\n")
                output.flush()
                latest[row["id"]] = row
                return row

            completed = await asyncio.gather(*(one(question) for question in pending))
    errors = sum(row["error"] is not None for row in completed)
    print(f"Done: {len(completed) - errors}; skipped: {skipped}; errors: {errors}")
    valid = [latest[q["id"]] for q in questions if latest[q["id"]]["error"] is None]
    for dataset in sorted({q["dataset"] for q in questions}):
        ids = {q["id"] for q in questions if q["dataset"] == dataset}
        rows = [row for row in valid if row["id"] in ids]
        accuracy = (
            f"{sum(row['correct'] for row in rows) / len(rows):.2%}" if rows else "n/a"
        )
        print(f"{dataset}: accuracy {accuracy} ({len(rows)} successful labels)")
    cost = sum(
        (Decimal(row["cost_usd"]) for row in valid if row["cost_usd"] is not None),
        Decimal(0),
    )
    unknown = sum(row["cost_usd"] is None for row in valid)
    print(
        f"Total cost_usd: {cost} ({unknown} labels with unknown cost; latest successful selected rows)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("GATEWAY_API_KEY"))
    parser.add_argument("--concurrency", type=positive, default=4)
    parser.add_argument("--per-dataset", type=positive)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "docs/benchmarks/router/labels"
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or GATEWAY_API_KEY is required")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
