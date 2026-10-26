"""Dependency-free TF-IDF/logistic cascade scoring of raw question text."""

import json
import math
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    probabilities: dict[str, float]
    tau: float | None


class _Classifier:
    def __init__(self, values: dict) -> None:
        self.vocabulary: dict[str, int] = values["vocabulary"]
        self.idf: list[float] = values["idf"]
        self.coefficients: list[float] = values["coefficients"]
        self.intercept: float = values["intercept"]
        self.settings: dict = values["settings"]
        self.constant: float | None = values.get("constant_probability")
        size = len(self.vocabulary)
        if sorted(self.vocabulary.values()) != list(range(size)):
            raise ValueError("Vocabulary indices must be contiguous")
        if len(self.coefficients) != size or len(self.idf) != size:
            raise ValueError(
                "Vocabulary, idf, and coefficients must have equal lengths"
            )
        if not all(
            math.isfinite(v) for v in [*self.idf, *self.coefficients, self.intercept]
        ):
            raise ValueError("Non-finite classifier weights")
        if self.constant is not None and not 0 <= self.constant <= 1:
            raise ValueError("Invalid constant probability")
        for setting in ("lowercase", "use_idf", "smooth_idf", "sublinear_tf"):
            if type(self.settings[setting]) is not bool:
                raise ValueError(f"Invalid vectorizer setting: {setting}")
        if self.settings["norm"] not in ("l1", "l2", None):
            raise ValueError("Unsupported TF-IDF norm")
        self.pattern = re.compile(self.settings["token_pattern"])
        if self.pattern.groups > 1:
            raise ValueError("Token pattern must have at most one capturing group")
        lo, hi = self.settings["ngram_range"]
        if not isinstance(lo, int) or not isinstance(hi, int) or not 1 <= lo <= hi:
            raise ValueError("Invalid ngram range")

    def probability(
        self, question: str
    ) -> float:  # Question text only, never prompt metadata.
        if self.constant is not None:
            return self.constant
        text = question.lower() if self.settings["lowercase"] else question
        words = self.pattern.findall(text)
        lo, hi = self.settings["ngram_range"]
        counts = Counter(
            self.vocabulary[term]
            for n in range(lo, hi + 1)
            for start in range(len(words) - n + 1)
            if (term := " ".join(words[start : start + n])) in self.vocabulary
        )
        weights = {}
        for index, count in sorted(counts.items()):
            tf = 1 + math.log(count) if self.settings["sublinear_tf"] else float(count)
            weights[index] = tf * self.idf[index] if self.settings["use_idf"] else tf
        norm = self.settings["norm"]
        length = (
            math.sqrt(sum(value * value for value in weights.values()))
            if norm == "l2"
            else sum(abs(value) for value in weights.values())
            if norm == "l1"
            else 1
        )
        score = self.intercept + sum(
            value / (length or 1) * self.coefficients[index]
            for index, value in weights.items()
        )
        # Stable sigmoid avoids overflow for highly confident predictions.
        if score >= 0:
            return 1 / (1 + math.exp(-score))
        exponential = math.exp(score)
        return exponential / (1 + exponential)


class Router:
    def __init__(
        self, artifact: str | Path
    ) -> None:  # Load and compile once per instance.
        path = Path(artifact)
        try:
            values = json.loads(path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Router artifact not found: {path}. Export it with scripts/router_train/export.py."
            ) from None
        try:
            if values["version"] != 1:
                raise ValueError("Unsupported artifact version")
            self.models: tuple[str, ...] = tuple(values["models"])
            self.levels: dict[str, float | None] = dict(values["levels"])
            self.run_name: str | None = values.get("metadata", {}).get("run")
            if len(self.models) < 2 or len(set(self.models)) != len(self.models):
                raise ValueError("At least two distinct models are required")
            if not self.levels or any(
                tau is not None and not 0 <= tau <= 1 for tau in self.levels.values()
            ):
                raise ValueError("Invalid level thresholds")
            self.classifiers = {
                model: _Classifier(values["classifiers"][model])
                for model in self.models[:-1]
            }
        except (KeyError, TypeError, ValueError, re.error) as exc:
            raise ValueError(f"Invalid router artifact {path}: {exc}") from exc

    def route(
        self,
        question: str,  # Same raw question field used by training.
        level: str = "balanced",  # Named quality level chosen using train OOF predictions.
        allowed_models: Collection[str]
        | None = None,  # Hard eligibility checks from serving.
    ) -> RoutingDecision:
        if level not in self.levels:
            raise ValueError(f"Unknown router level: {level}")
        tau = self.levels[level]
        probabilities = {
            name: classifier.probability(question)
            for name, classifier in self.classifiers.items()
        }
        eligible = [
            model
            for model in self.models
            if allowed_models is None or model in allowed_models
        ]
        if not eligible:
            raise ValueError("No eligible router models")
        chosen = eligible[-1]
        # null is the explicit all-last fallback when no train threshold is feasible.
        if tau is not None:
            chosen = next(
                (
                    name
                    for name, p in probabilities.items()
                    if name in eligible and p >= tau
                ),
                chosen,
            )
        return RoutingDecision(chosen, probabilities, tau)
