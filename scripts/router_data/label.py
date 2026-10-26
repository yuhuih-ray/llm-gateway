"""Collect one requested gateway model's labels, without cache or fallback opt-in."""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
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

RESPONSES = ROOT / "data/router/responses"


class ModelMismatchError(Exception):
    pass


def extract_mmlupro(text: str) -> str | None:
    matches = re.findall(r"(?i:answer)\s*:\s*[*_]*\s*\(?([A-J])(?![A-Za-z])\)?", text)
    if not matches:
        matches = re.findall(
            r"(?i:answer is)\s*[*_]*\s*\(?([A-J])(?![A-Za-z])\)?", text
        )
    return matches[-1] if matches else None


def grade(question: dict, content: str) -> tuple[str | None, bool]:
    pred = (
        extract_answer(content)
        if question["dataset"] == "gsm8k"
        else extract_mmlupro(content)
    )
    correct = (
        correct_answer(question["gold"], pred)
        if question["dataset"] == "gsm8k"
        else pred is not None and pred == question["gold"]
    )
    return pred, correct


def save_response(
    model: str, question_id: str, request_id: str | None, content: str
) -> None:
    # Save before grading; these private raw responses are never committed.
    RESPONSES.mkdir(parents=True, exist_ok=True)
    with (RESPONSES / f"{model}.jsonl").open("a") as output:
        output.write(
            json.dumps(dict(id=question_id, request_id=request_id, content=content))
            + "\n"
        )
        output.flush()


def regrade(args: argparse.Namespace) -> None:  # Offline; no client or API key needed.
    path = args.out_dir / f"{args.model}.jsonl"
    response_path = RESPONSES / f"{args.model}.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(f"No saved responses: {response_path}")
    questions = {q["id"]: q for q in read_manifest()}
    history = [
        json.loads(line)
        for line in (path.read_text().splitlines() if path.exists() else [])
        if line.strip()
    ]
    if any(row["model"] != args.model for row in history):
        raise ValueError("Existing label file contains another model")
    latest = {row["id"]: row for row in history}
    attempts = {(row["id"], row["request_id"]): row for row in history}
    responses = read_labels(response_path)
    changed = skipped = 0
    for question_id, response in responses.items():
        # Match billing/latency to this response, not a later failed request.
        key = (question_id, response["request_id"])
        if key not in attempts:
            skipped += 1
            continue
        row = attempts[key].copy()
        row["pred"], row["correct"] = grade(questions[question_id], response["content"])
        row["error"] = None
        changed += any(
            row[field] != latest[question_id][field]
            for field in ("pred", "correct", "error")
        )
        latest[question_id] = row
    # Validate and grade everything before atomically replacing the label file.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            for question_id in sorted(latest):
                output.write(json.dumps(latest[question_id]) + "\n")
            output.flush()
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if skipped:
        print(
            f"Warning: skipped {skipped} saved responses without matching label metadata",
            file=sys.stderr,
        )
    print(f"Regraded: {len(responses) - skipped}; labels changed: {changed}")


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


def budget_exceeded(response: httpx.Response) -> bool:
    if response.status_code != 429:
        return False
    try:
        value = response.json()
    except ValueError:
        return False
    error = value.get("error") if isinstance(value, dict) else None
    return isinstance(error, dict) and error.get("code") == "insufficient_quota"


async def request_completion(
    client: httpx.AsyncClient,
    payload: dict,
    row: dict,  # Update final-attempt timing and total attempts even on exceptions.
) -> httpx.Response:
    rate_retries = failures = 0
    while True:
        row["attempts"] += 1
        started = time.perf_counter()
        try:
            try:
                response = await client.post("chat/completions", json=payload)
            finally:
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            failures += 1
            if failures >= 3:
                raise
            await asyncio.sleep(0.5 * 2 ** (failures - 1))
            continue
        if budget_exceeded(response):
            response.raise_for_status()
        if response.status_code == 429 or (
            response.status_code == 503 and "Retry-After" in response.headers
        ):
            if rate_retries < 20:
                rate_retries += 1
                await asyncio.sleep(retry_after(response.headers.get("Retry-After")))
                continue
            # Exhausted shared budget must not fall through to the 5xx retries.
            response.raise_for_status()
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
        attempts=0,
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
            row,
        )
        row["request_id"] = response.headers.get("X-Request-ID")
        if response.headers.get("X-Gateway-Model") != model:
            raise ModelMismatchError()
        value = response.json()
        choice = value["choices"][0]
        content = choice["message"].get("content") or ""
        if not isinstance(content, str):
            raise TypeError("Invalid completion content")
        save_response(model, question["id"], row["request_id"], content)
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
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=str(cost) if cost is not None else None,
        )
        row["pred"], row["correct"] = grade(question, content)
    except httpx.HTTPStatusError as exc:
        row["error"] = (
            "budget_exceeded"
            if budget_exceeded(exc.response)
            else f"HTTP{exc.response.status_code}"
        )
        row["request_id"] = exc.response.headers.get("X-Request-ID")
    except Exception as exc:
        # Persist sanitized error names, never credentials or response bodies.
        # Cancellation is a BaseException and propagates without a bogus label.
        row["error"] = type(exc).__name__
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
    stopped = asyncio.Event()
    started = time.perf_counter()
    done = failed = correct = consecutive_errors = 0
    stop_reason = None
    error_counts: Counter[str] = Counter()

    def progress(*, final: bool = False) -> None:
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0
        remaining = math.ceil((len(pending) - done) / rate) if rate and not final else 0
        # Match the summary: accuracy excludes failed requests.
        accuracy = correct / (done - failed) if done > failed else 0
        print(
            f"[{done}/{len(pending)}] errors={failed} correct={accuracy:.2%} "
            f"rate={rate:.2f} eta={remaining // 60:02d}:{remaining % 60:02d}",
            file=sys.stderr,
            flush=True,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=httpx.Timeout(180, connect=2),
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:
        with path.open("a") as output:

            async def one(question: dict) -> dict | None:
                nonlocal done, failed, correct, consecutive_errors, stop_reason
                async with semaphore:
                    if stopped.is_set():
                        return None
                    row = await label_question(client, args.model, question)
                    if row["error"] is None:
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        error_counts[row["error"]] += 1
                    if not stopped.is_set():
                        if row["error"] == "budget_exceeded":
                            stop_reason = "budget exceeded"
                        elif consecutive_errors >= 10:
                            stop_reason = "10 consecutive errors"
                            print("Stopped: 10 consecutive errors", flush=True)
                        if stop_reason is not None:
                            stopped.set()
                # No await between append and flush, so concurrent tasks cannot interleave lines.
                output.write(json.dumps(row) + "\n")
                output.flush()
                latest[row["id"]] = row
                done += 1
                failed += row["error"] is not None
                correct += row["error"] is None and row["correct"]
                if done % 25 == 0 and done < len(pending):
                    progress()
                return row

            completed = await asyncio.gather(*(one(question) for question in pending))
    progress(final=True)
    completed = [row for row in completed if row is not None]
    if stopped.is_set():
        print(f"Not started: {len(pending) - len(completed)} ({stop_reason})")
        print(f"Error counts: {json.dumps(dict(error_counts), sort_keys=True)}")
    errors = sum(row["error"] is not None for row in completed)
    print(f"Done: {len(completed) - errors}; skipped: {skipped}; errors: {errors}")
    valid = [
        latest[q["id"]]
        for q in questions
        if q["id"] in latest and latest[q["id"]]["error"] is None
    ]
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
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="Regrade saved responses without HTTP calls",
    )
    args = parser.parse_args()
    if args.regrade:
        regrade(args)
        return
    if not args.api_key:
        parser.error("--api-key or GATEWAY_API_KEY is required")
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
