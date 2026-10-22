"""Compare paired GSM8K results; all differences and intervals are A minus B."""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path


def load(path: Path) -> dict[int, bool]:  # One evaluation result file.
    results = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if type(row["id"]) is not int or type(row["correct"]) is not bool:
            raise ValueError(f"Invalid id/correct in {path}")
        if row["id"] in results:
            raise ValueError(f"Duplicate id {row['id']} in {path}")
        results[row["id"]] = row["correct"]
    return results


def statistics(a: dict[int, bool], b: dict[int, bool]) -> dict[str, float | int]:
    shared = a.keys() & b.keys()
    n = len(shared)
    if not n:
        raise ValueError("Results have no shared question ids")
    n_ab = sum(a[i] and not b[i] for i in shared)
    n_ba = sum(b[i] and not a[i] for i in shared)
    difference = (n_ab - n_ba) / n
    se = math.sqrt(n_ab + n_ba - (n_ab - n_ba) ** 2 / n) / n
    discordant = n_ab + n_ba
    # Under the null, either direction has probability 1/2 among discordant pairs.
    # Integer arithmetic avoids overflow from separately converting large binomials.
    p = min(
        1.0,
        (2 * sum(math.comb(discordant, k) for k in range(min(n_ab, n_ba) + 1)))
        / 2**discordant,
    )
    return {
        "n": n,
        "accuracy_a": sum(a[i] for i in shared) / n,
        "accuracy_b": sum(b[i] for i in shared) / n,
        "n_ab": n_ab,
        "n_ba": n_ba,
        "difference": difference,
        "ci_low": difference - 1.96 * se,
        "ci_high": difference + 1.96 * se,
        "p": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.paths) < 2:
        parser.error("provide at least two result JSONL paths")
    try:
        files = [(str(path).replace("|", "\\|"), load(path)) for path in args.paths]
        rows = [(a, b, statistics(x, y)) for (a, x), (b, y) in combinations(files, 2)]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(
        "| A | B | Shared n | Accuracy A | Accuracy B | n_ab | n_ba | A − B | 95% CI | McNemar p |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for a, b, s in rows:
        print(
            f"| {a} | {b} | {s['n']} | {s['accuracy_a']:.2%} | {s['accuracy_b']:.2%} | "
            f"{s['n_ab']} | {s['n_ba']} | {s['difference']:.2%} | "
            f"[{s['ci_low']:.2%}, {s['ci_high']:.2%}] | {s['p']:.6g} |"
        )


if __name__ == "__main__":
    main()
