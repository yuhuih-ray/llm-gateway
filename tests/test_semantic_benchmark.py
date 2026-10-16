"""Benchmark mechanics only; no downloads, models, or provider calls in CI."""

import importlib.util
import random
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "semantic_common",
    Path(__file__).resolve().parents[1] / "scripts/semantic_cache/common.py",
)
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def test_metrics_denominators_and_inclusive_threshold():
    rows = common.metrics([1, 1, 0, 0, 0], [0.95, 0.94, 0.95, 0.9, 0.1])
    at_095 = next(row for row in rows if row["threshold"] == 0.95)
    assert at_095["hit_rate"] == 0.5
    assert at_095["false_hit_rate"] == 1 / 3
    assert len(rows) == 20
    assert rows[0]["threshold"] == 0.80
    assert rows[-1]["threshold"] == 0.99
    assert common.operating_point(rows, 0.001)["threshold"] == 0.96


def test_absent_class_is_not_zero_and_no_safe_threshold():
    positives = common.metrics([1], [0.99])
    assert positives[0]["false_hit_rate"] is None
    assert common.operating_point(positives, 0.01) is None
    negatives = common.metrics([0], [1.0])
    assert negatives[0]["hit_rate"] is None
    assert common.operating_point(negatives, 0.01) is None
    with pytest.raises(ValueError):
        common.metrics([1], [])


@pytest.mark.parametrize(
    "text",
    [
        "There are 10 apples and 10 pears.",
        "Pay $1,200.50 for 2 items.",
        "Use 1/2 of 50% of the 20 units.",
    ],
)
def test_perturbation_changes_one_occurrence(text):
    changed, edit = common.perturb_number(text, random.Random(12))
    assert changed != text
    assert changed == text[: edit["start"]] + edit["after"] + text[edit["end"] :]
    assert common.quantities(text) != common.quantities(changed)
    assert common.perturb_number(text, random.Random(12))[0] == changed


def test_paraphrase_number_validation():
    original = "Two people buy 10 items each and pay $1.50."
    assert common.valid_paraphrase(original, "At $1.50, two people each buy 10 items.")
    assert not common.valid_paraphrase(original, original)
    assert not common.valid_paraphrase(
        original, "Two people buy 11 items each and pay $1.50."
    )
    assert not common.valid_paraphrase(
        original, "Three people buy 10 items each and pay $1.50."
    )
    assert not common.valid_paraphrase("Buy 10 and 10 items.", "Buy 10 items.")
    assert not common.valid_paraphrase(original, "")


def test_easy_controls_exclude_related_source_variants():
    assert (
        common.lexical_overlap("Alice has ten apples.", "Alice has nine apples.") >= 0.2
    )
    assert common.lexical_overlap("今天我们去北京旅游。", "今天他们去北京旅游。") >= 0.2
    assert common.lexical_overlap("Alice bought apples.", "The sea is blue.") < 0.2
