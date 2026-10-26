"""Paired router-label report, using only latest successful rows shared by both files."""

import argparse
import sys
from decimal import Decimal
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.router_data.common import read_labels, read_manifest  # noqa: E402


def statistics(questions: list[dict], a: dict[str, dict], b: dict[str, dict]) -> dict:
    ids = [
        q["id"]
        for q in questions
        if q["id"] in a
        and q["id"] in b
        and a[q["id"]]["error"] is None
        and b[q["id"]]["error"] is None
    ]
    n = len(ids)
    result = {
        "n": n,
        "a_wrong_b_right": sum(not a[i]["correct"] and b[i]["correct"] for i in ids),
        "a_right_b_wrong": sum(a[i]["correct"] and not b[i]["correct"] for i in ids),
    }
    for name, labels in (("a", a), ("b", b)):
        result[f"accuracy_{name}"] = (
            sum(labels[i]["correct"] for i in ids) / n if n else None
        )
        result[f"latency_{name}"] = (
            sum(labels[i]["latency_ms"] for i in ids) / n if n else None
        )
        costs = [labels[i]["cost_usd"] for i in ids]
        result[f"cost_{name}"] = (
            sum((Decimal(cost) for cost in costs), Decimal(0))
            if n and all(cost is not None for cost in costs)
            else None
        )
    return result


def table(title: str, groups: dict[str, list[dict]], a: dict, b: dict) -> str:
    lines = [
        f"### {title}",
        "",
        "| Group | n | Accuracy A | Accuracy B | A wrong / B right | A right / B wrong | Cost A (USD) | Cost B (USD) | Mean latency A (ms) | Mean latency B (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, questions in sorted(groups.items()):
        s = statistics(questions, a, b)
        values = [group, str(s["n"])]
        values.extend(
            f"{s[f'accuracy_{name}']:.2%}"
            if s[f"accuracy_{name}"] is not None
            else "n/a"
            for name in ("a", "b")
        )
        values.extend(str(s[key]) for key in ("a_wrong_b_right", "a_right_b_wrong"))
        values.extend(
            str(s[f"cost_{name}"]) if s[f"cost_{name}"] is not None else "n/a"
            for name in ("a", "b")
        )
        values.extend(
            f"{s[f'latency_{name}']:.2f}" if s[f"latency_{name}"] is not None else "n/a"
            for name in ("a", "b")
        )
        lines.append(
            "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    for path in (args.a, args.b, args.manifest):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    a, b = read_labels(args.a), read_labels(args.b)
    questions = read_manifest(args.manifest)
    print(f"A: `{args.a}`; B: `{args.b}`\n")
    for title, field, rows in (
        ("Dataset", "dataset", questions),
        (
            "MMLU-Pro category",
            "category",
            [q for q in questions if q["dataset"] == "mmlupro"],
        ),
        ("Split", "split", questions),
    ):
        groups = {
            value: [q for q in rows if q[field] == value]
            for value in {q[field] for q in rows}
        }
        print(table(title, groups, a, b) + "\n")


if __name__ == "__main__":
    main()
