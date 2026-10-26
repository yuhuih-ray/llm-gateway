"""Shared manifest identities and append-only label reading."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/benchmarks/router/questions.jsonl"


def digest(question_id: str) -> str:  # Dataset-prefixed stable identity.
    return hashlib.sha256(question_id.encode()).hexdigest()


def split_for(question_id: str, category: str) -> str:
    if category in {"law", "history"}:
        return "ood"
    return "test" if int(digest(question_id), 16) % 100 < 20 else "train"


def read_manifest(path: Path = MANIFEST) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question id in manifest")
    return rows


def read_labels(path: Path) -> dict[str, dict]:  # Last row wins, including errors.
    if not path.exists():
        return {}
    latest = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = row
    return latest


def select_questions(rows: list[dict], per_dataset: int | None) -> list[dict]:
    if per_dataset is None:
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    return [
        row
        for name in sorted(groups)
        for row in sorted(groups[name], key=lambda row: digest(row["id"]))[:per_dataset]
    ]


def pending_questions(rows: list[dict], latest: dict[str, dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row["id"] not in latest or latest[row["id"]]["error"] is not None
    ]
