"""Export the train-only TF-IDF cascade and evaluate fixed quality levels offline."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.router_data.common import MANIFEST, ROOT
from scripts.router_train import train

SETTINGS = (
    "lowercase",
    "token_pattern",
    "ngram_range",
    "norm",
    "use_idf",
    "smooth_idf",
    "sublinear_tf",
)
DEFAULT_LEVELS = ["high=0.01", "balanced=0.03", "economy=0.08"]


def parse_levels(values: list[str]) -> dict[str, float]:
    levels = {}
    for value in values:
        name, separator, amount = value.partition("=")
        if not separator or not name or name in levels:
            raise ValueError("Levels must be unique name=drop pairs")
        drop = float(amount)
        if not 0 <= drop <= 1:
            raise ValueError("Level accuracy drops must be between zero and one")
        levels[name] = drop
    return levels


def serialize_classifier(classifier, params: dict) -> dict:
    if classifier is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        return dict(
            vocabulary={},
            idf=[],
            coefficients=[],
            intercept=0.0,
            constant_probability=params["constant_probability"],
            settings={key: vectorizer.get_params()[key] for key in SETTINGS},
        )
    vectorizer = classifier.named_steps["features"]
    regression = classifier.named_steps["classifier"]
    # These are train.py's fixed defaults, intentionally outside the portable format.
    if (
        vectorizer.analyzer != "word"
        or vectorizer.strip_accents is not None
        or vectorizer.stop_words is not None
        or vectorizer.binary
        or vectorizer.preprocessor is not None
        or vectorizer.tokenizer is not None
    ):
        raise ValueError("Unsupported vectorizer preprocessing for export")
    if list(regression.classes_) != [0, 1]:
        raise ValueError("Expected a binary correctness classifier")
    return dict(
        vocabulary={term: int(index) for term, index in vectorizer.vocabulary_.items()},
        idf=vectorizer.idf_.tolist()
        if vectorizer.use_idf
        else [1.0] * len(vectorizer.vocabulary_),
        coefficients=regression.coef_[0].tolist(),
        intercept=float(regression.intercept_[0]),
        settings={key: vectorizer.get_params()[key] for key in SETTINGS},
    )


def evaluate_levels(
    rows: list[
        dict
    ],  # Only held-out rows; classifiers and thresholds are already frozen.
    models: list[str],
    fitted: list,
    parameters: list[dict],
    levels: dict[str, float | None],
) -> tuple[dict, list[dict]]:
    result = {}
    fixture = []
    for split in ("test", "ood"):
        evaluation = [row for row in rows if row["split"] == split]
        if not evaluation:
            result[split] = None
            continue
        inputs = np.asarray([row["question"] for row in evaluation])
        # Score each held-out split once, reuse predictions for every fixed level.
        predictions = np.column_stack(
            [train.probabilities(m, p, inputs) for m, p in zip(fitted, parameters)]
        )
        if split == "test":
            fixture = [
                dict(
                    id=row["id"],
                    question=row["question"],
                    probabilities=prediction.tolist(),
                )
                for row, prediction in list(zip(evaluation, predictions))[:200]
            ]
        reference = train.baselines(evaluation, models)
        last = reference["single"][models[-1]]
        result[split] = {}
        for name, tau in levels.items():
            choices = (
                np.full(len(evaluation), len(models) - 1)
                if tau is None
                else train.cascade(predictions, tau)
            )
            metrics = train.selected_metrics(
                train.summarize(evaluation, choices, models), reference, models
            )
            result[split][name] = dict(
                tau=tau,
                n=len(evaluation),
                **metrics,
                last_model_accuracy=last["accuracy"],
                accuracy_retained=metrics["accuracy"] / last["accuracy"]
                if last["accuracy"]
                else None,
                last_model_mean_latency_ms=last["mean_latency_ms"],
                mean_latency_vs_last=metrics["mean_latency_ms"]
                / last["mean_latency_ms"]
                if last["mean_latency_ms"]
                else None,
            )
    return result, fixture


def markdown(report: dict) -> str:
    def number(value):
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "| Level | Split | Tau | Accuracy | Last accuracy | Accuracy retained | Cost reduction | Mean latency (ms) | Last latency (ms) | Latency ratio | Traffic shares | Accuracy vs hull | Oracle savings captured |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for split, levels in report["evaluation"].items():
        if levels is None:
            continue
        for name, row in levels.items():
            shares = ", ".join(f"{m}: {p:.2%}" for m, p in row["shares"].items())
            values = [
                name,
                split,
                *[
                    number(row[key])
                    for key in (
                        "tau",
                        "accuracy",
                        "last_model_accuracy",
                        "accuracy_retained",
                        "cost_reduction_vs_last",
                        "mean_latency_ms",
                        "last_model_mean_latency_ms",
                        "mean_latency_vs_last",
                    )
                ],
                shares,
                number(row["accuracy_vs_hull"]),
                number(row["oracle_savings_captured"]),
            ]
            lines.append(
                "| " + " | ".join(value.replace("|", "\\|") for value in values) + " |"
            )
    return "\n".join(lines)


def export_router(
    rows: list[
        dict
    ],  # The successful shared cohort, projected to question text by load_data.
    models: list[str],  # Supplied increasing price order, same as training.
    drops: dict[str, float],  # Train-only accuracy tolerance per level.
    run: str,
    artifact_dir: Path,
    result_dir: Path,
    *,
    pricing: str,  # Cost basis already applied by load_data.
    prices: dict,  # Exact reference price table used for this run.
) -> tuple[dict, dict]:
    train_rows = [row for row in rows if row["split"] == "train"]
    fitted, parameters, oof = train.fit_cascade(
        train_rows, models, "tfidf_lr", ROOT / "data/router/embeddings"
    )
    selections = {
        name: train.select_tau(train_rows, oof, models, allowed_drop=drop)
        for name, drop in drops.items()
    }
    levels = {name: choice["tau"] for name, choice in selections.items()}
    metadata = dict(
        run=run,
        pricing=pricing,
        price_table=prices,
        data_sha256=hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        feature="question text",
        created_at=datetime.now(UTC).isoformat(),
    )
    artifact = dict(
        version=1,
        models=models,
        classifiers={
            name: serialize_classifier(model, params)
            for name, model, params in zip(models[:-1], fitted, parameters)
        },
        levels=levels,
        metadata=metadata,
    )
    evaluation, fixture = evaluate_levels(rows, models, fitted, parameters, levels)
    report = dict(
        metadata=metadata,
        models=models,
        allowed_drops=drops,
        selections=selections,
        evaluation=evaluation,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (artifact_dir / f"{run}.json", artifact),
        (artifact_dir / f"{run}-parity.json", dict(models=models, questions=fixture)),
        (result_dir / f"{run}-levels.json", report),
    ):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(markdown(report))
    return artifact, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["local-qwen", "gemini-flash-lite"]
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=DEFAULT_LEVELS,
        help="Unique name=allowed_accuracy_drop pairs",
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--pricing", choices=("list", "billed"), default="list")
    args = parser.parse_args()
    if len(args.models) < 2 or len(set(args.models)) != len(args.models):
        parser.error("Provide at least two distinct models, ordered by price")
    if Path(args.run).name != args.run or args.run in {".", ".."}:
        parser.error("--run must be a filename")
    try:
        levels = parse_levels(args.levels)
    except ValueError as exc:
        parser.error(str(exc))
    prices = train.price_table()
    rows = train.load_data(
        MANIFEST,
        ROOT / "docs/benchmarks/router/labels",
        args.models,
        args.pricing,
        prices,
    )
    export_router(
        rows,
        args.models,
        levels,
        args.run,
        ROOT / "src/llm_gateway/router_artifacts",
        ROOT / "docs/benchmarks/router/results",
        pricing=args.pricing,
        prices=prices,
    )


if __name__ == "__main__":
    main()
