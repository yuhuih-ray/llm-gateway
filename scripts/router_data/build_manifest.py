"""Build the fixed GSM8K/MMLU-Pro router manifest; no model calls."""

import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

# Support direct script invocation as well as python -m scripts.router_data....
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.router_data.common import MANIFEST, digest, split_for  # noqa: E402
from scripts.serving_bench.gsm8k_eval import dataset, extract_answer  # noqa: E402


def build(gsm8k: list[dict], mmlupro: list[dict]) -> list[dict]:
    rows = []
    for index, item in enumerate(gsm8k):
        question_id = f"gsm8k:{index}"
        rows.append(
            {
                "id": question_id,
                "dataset": "gsm8k",
                "category": "gsm8k",
                "split": split_for(question_id, "gsm8k"),
                "gold": extract_answer(item["answer"], gold=True),
                "prompt": item["question"]
                + '\nPlease reason step by step, and put your final answer after "####".',
            }
        )
    groups = defaultdict(list)
    for item in mmlupro:
        groups[item["category"]].append(item)
    for category, items in groups.items():
        count = round(2000 * len(items) / len(mmlupro))
        for item in sorted(
            items, key=lambda item: digest(f"mmlupro:{item['question_id']}")
        )[:count]:
            question_id = f"mmlupro:{item['question_id']}"
            if not 1 <= len(item["options"]) <= 10:
                raise ValueError(f"Invalid options for {question_id}")
            options = "\n".join(
                f"{chr(65 + i)}. {option}" for i, option in enumerate(item["options"])
            )
            rows.append(
                {
                    "id": question_id,
                    "dataset": "mmlupro",
                    "category": category,
                    "split": split_for(question_id, category),
                    "gold": item["answer"],
                    "prompt": item["question"]
                    + "\n"
                    + options
                    + '\nThink step by step, then end with a final line "ANSWER: <letter>".',
                }
            )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source question id")
    return sorted(rows, key=lambda row: row["id"])


async def main() -> None:
    # Only the builder needs the optional router dependency group.
    from datasets import load_dataset

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        gsm8k = await dataset(client)
    mmlupro = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = build(gsm8k, list(mmlupro))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    for field in ("dataset", "category", "split"):
        print(f"{field}: {dict(sorted(Counter(row[field] for row in rows).items()))}")
    # Independent category rounding can differ from 2000; do not adjust quotas.
    print(f"Total: {len(rows)}; manifest: {MANIFEST}")
    print(f"MMLU-Pro dataset fingerprint: {mmlupro._fingerprint}")


if __name__ == "__main__":
    asyncio.run(main())
