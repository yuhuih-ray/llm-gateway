"""Send reproducible demo traffic using only raw manifest questions."""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.router_data.common import MANIFEST, read_manifest


async def send_batch(
    client: httpx.AsyncClient,  # Injectable transport; tests never send real traffic.
    questions: list[dict],
    quality: str,
    concurrency: int,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    models: Counter = Counter()
    errors: Counter = Counter()
    start = datetime.now(UTC).isoformat()
    print(f"start: {start}", flush=True)

    async def send(question):
        async with semaphore:
            try:
                response = await client.post(
                    "/chat/completions",
                    headers={"X-Gateway-Quality": quality},
                    json=dict(
                        model="auto",
                        messages=[dict(role="user", content=question["question"])],
                        temperature=0,
                        max_tokens=64,
                        stream=False,
                    ),
                )
                response.raise_for_status()
                models[response.headers.get("X-Gateway-Model", "missing")] += 1
            except httpx.HTTPError as exc:
                name = (
                    f"HTTP_{exc.response.status_code}"
                    if isinstance(exc, httpx.HTTPStatusError)
                    else type(exc).__name__
                )
                errors[name] += 1

    await asyncio.gather(*(send(question) for question in questions))
    end = datetime.now(UTC).isoformat()
    print(f"end: {end}", flush=True)
    successful = sum(models.values())
    shares = {model: count / successful for model, count in sorted(models.items())}
    print(
        f"Successful: {successful}/{len(questions)}; model shares among successful responses: {json.dumps(shares)}"
    )
    if errors:
        print(f"Errors: {json.dumps(errors, sort_keys=True)}")
    return dict(start=start, end=end, shares=shares, errors=dict(errors))


def sample_questions(manifest: Path, split: str, n: int, seed: int) -> list[dict]:
    pool = sorted(
        (row for row in read_manifest(manifest) if row["split"] == split),
        key=lambda row: row["id"],
    )
    if n < 1 or n > len(pool):
        raise ValueError(f"--n must be between 1 and {len(pool)} for split {split}")
    return random.Random(seed).sample(pool, n)


async def run(args) -> None:
    questions = sample_questions(args.manifest, args.split, args.n, args.seed)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=180,
    ) as client:
        await send_batch(client, questions, args.quality, args.concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "ood"), required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quality", choices=("economy", "balanced", "high"), default="balanced"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("GATEWAY_API_KEY"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if not args.api_key or args.concurrency < 1:
        parser.error(
            "Require --api-key (or GATEWAY_API_KEY) and positive --concurrency"
        )
    try:
        asyncio.run(run(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
