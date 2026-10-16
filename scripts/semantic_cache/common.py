"""Reproducible sampling, quantity checks, and threshold metrics."""

import hashlib
import json
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

DATA = Path(__file__).parent / ".data"
SEED = 20261015
THRESHOLDS = [value / 100 for value in range(80, 100)]
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
}
NUMERAL = r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?!\w)"
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand million billion first second third "
    "fourth fifth sixth seventh eighth ninth tenth half quarter twice double triple dozen"
).split()
QUANTITY = re.compile(NUMERAL + r"|\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.I)


def quantities(text: str) -> Counter:
    # Preserve multiplicity as well as literal digit and number-word forms.
    return Counter(match.group().lower() for match in QUANTITY.finditer(text))


def valid_paraphrase(original: str, candidate: str) -> bool:
    return (
        bool(candidate.strip())
        and candidate.strip() != original.strip()
        and quantities(original) == quantities(candidate)
    )


def perturb_number(
    text: str,  # Original question; change one numeral occurrence only.
    rng: random.Random,  # Seeded sampler shared by deterministic construction.
) -> tuple[str, dict]:
    matches = list(re.finditer(NUMERAL, text))
    if not matches:
        raise ValueError("No digit numeral to perturb")
    match = rng.choice(matches)
    original = match.group()
    changed = format(Decimal(original.replace(",", "")) + 1, "f")
    if "," in original:
        decimals = len(original.split(".")[1]) if "." in original else 0
        changed = f"{Decimal(changed):,.{decimals}f}"
    # Splicing changes exactly one occurrence, even when a numeral repeats.
    return text[: match.start()] + changed + text[match.end() :], {
        "start": match.start(),
        "end": match.end(),
        "before": original,
        "after": changed,
    }


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(
    labels: list[int],  # 1 means answer reuse is expected; 0 means it is unsafe.
    scores: list[float],  # Cosines in the identical pair order, before rounding.
) -> list[dict]:
    if len(labels) != len(scores):
        raise ValueError("Mismatched labels/scores")
    positives = sum(labels)
    negatives = len(labels) - positives
    rows = []
    for threshold in THRESHOLDS:
        true_hits = sum(
            label == 1 and score >= threshold for label, score in zip(labels, scores)
        )
        false_hits = sum(
            label == 0 and score >= threshold for label, score in zip(labels, scores)
        )
        rows.append(
            dict(
                threshold=threshold,
                true_hits=true_hits,
                false_hits=false_hits,
                hit_rate=true_hits / positives if positives else None,
                false_hit_rate=false_hits / negatives if negatives else None,
            )
        )
    return rows


def operating_point(
    rows: list[dict],  # Threshold sweep in ascending order.
    ceiling: float,  # False-hit fraction, not a percentage.
) -> dict | None:
    # Positive-only sets cannot establish a false-hit guarantee.
    return next(
        (
            row
            for row in rows
            if row["false_hit_rate"] is not None and row["false_hit_rate"] <= ceiling
        ),
        None,
    )


def lexical_overlap(a: str, b: str) -> float:
    # Character trigrams work for both English and Chinese without segmentation.
    def grams(text: str) -> set[str]:
        normalized = re.sub(r"\W", "", text.lower())
        return {normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))}

    left, right = grams(a), grams(b)
    return len(left & right) / len(left | right) if left | right else 1.0
