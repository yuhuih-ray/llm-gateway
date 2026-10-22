"""Evaluate an OpenAI-compatible served model on GSM8K (no gateway changes)."""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
DATA_PATH = Path(__file__).resolve().parents[2] / "data/gsm8k/test.jsonl"
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def normalize(text: str) -> str | None:  # Numeric answer, not the full reasoning.
    value = text.strip().replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return value if number.is_finite() else None


def extract_answer(text: str, *, gold: bool = False) -> str | None:
    # The last marker overrides earlier intermediate answers.
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        if gold:
            return normalize(tail)
        match = NUMBER.search(tail)
        return normalize(match.group()) if match else None
    if gold:
        return None
    numbers = NUMBER.findall(text)
    return normalize(numbers[-1]) if numbers else None


def correct_answer(gold: str | None, pred: str | None) -> bool:
    return gold is not None and pred is not None and Decimal(gold) == Decimal(pred)


async def dataset(client: httpx.AsyncClient) -> list[dict]:  # Public download client.
    if not DATA_PATH.exists():
        response = await client.get(DATA_URL)
        response.raise_for_status()
        rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        if not rows or any(
            "question" not in row or "answer" not in row for row in rows
        ):
            raise ValueError("Invalid GSM8K dataset")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=DATA_PATH.parent, delete=False) as temp:
            temporary = Path(temp.name)
            temp.write(response.content)
        try:
            temporary.replace(DATA_PATH)
        finally:
            temporary.unlink(missing_ok=True)
    return [json.loads(line) for line in DATA_PATH.read_text().splitlines()]


async def completion(
    client: httpx.AsyncClient,  # Shared connection pool; credentials stay on this client.
    payload: dict,  # Fixed generation settings and one question.
) -> dict:
    for attempt in range(3):
        try:
            response = await client.post("chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            if attempt == 2:
                raise
        except httpx.HTTPStatusError as exc:
            if not 500 <= exc.response.status_code < 600 or attempt == 2:
                raise
        await asyncio.sleep(0.5 * 2**attempt)
    raise AssertionError("Unreachable")


async def evaluate(args: argparse.Namespace) -> None:
    # Never send the served-model bearer credential to the dataset host.
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as download:
        questions = await dataset(download)
    if args.limit is not None:
        questions = questions[: args.limit]
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        headers=headers,
        timeout=httpx.Timeout(120, connect=2),
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def one(question_id: int, question: dict) -> dict:
            gold = extract_answer(question["answer"], gold=True)
            pred, finish_reason, tokens = None, None, None
            error = None
            try:
                async with semaphore:
                    value = await completion(
                        client,
                        {
                            "model": args.model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": question["question"]
                                    + '\nPlease reason step by step, and put your final answer after "####".',
                                }
                            ],
                            "temperature": 0,
                            "max_tokens": 1024,
                            "stream": False,
                        },
                    )
                choice = value["choices"][0]
                pred = extract_answer(choice["message"].get("content") or "")
                finish_reason = choice.get("finish_reason")
                tokens = (value.get("usage") or {}).get("completion_tokens")
            except (
                httpx.HTTPError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                error = type(exc).__name__
                # Failed requests remain in the denominator; never print secrets/bodies.
                print(
                    f"Question {question_id} failed: {type(exc).__name__}",
                    file=sys.stderr,
                )
            return {
                "id": question_id,
                "gold": gold,
                "pred": pred,
                "correct": correct_answer(gold, pred),
                "finish_reason": finish_reason,
                "completion_tokens": tokens,
                "error": error,
            }

        results = await asyncio.gather(*(one(i, q) for i, q in enumerate(questions)))
    elapsed = time.perf_counter() - started
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{args.label}.jsonl"
    output.write_text(
        "".join(
            json.dumps(row) + "\n" for row in sorted(results, key=lambda row: row["id"])
        )
    )
    counts = [
        row["completion_tokens"]
        for row in results
        if row["completion_tokens"] is not None
    ]
    accuracy = sum(row["correct"] for row in results) / len(results)
    length = sum(row["finish_reason"] == "length" for row in results)
    mean = f"{sum(counts) / len(counts):.2f}" if counts else "n/a"
    print(f"Accuracy: {accuracy:.2%} ({len(results)} questions)")
    failed = sum(row["error"] is not None for row in results)
    unparsed = sum(row["pred"] is None and row["error"] is None for row in results)
    print(f"Failed requests: {failed}")
    print(f"Successful requests with no parsed answer: {unparsed}")
    print(f"finish_reason=length: {length}")
    print(f"Mean completion_tokens: {mean} ({len(counts)} reported)")
    print(f"Wall time: {elapsed:.2f}s (evaluation only)")
    print(f"Results: {output}")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=positive, default=16)
    parser.add_argument("--limit", type=positive)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("docs/benchmarks/serving/quality")
    )
    args = parser.parse_args()
    if (
        args.label in (".", "..")
        or "/" in args.label
        or "\\" in args.label
        or not args.label
    ):
        parser.error("--label must be a nonempty filename label")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
