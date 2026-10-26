"""Train and evaluate question-only price-ordered routing cascades offline."""

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llm_gateway.router_features import FEATURE_NAMES, extract_features
from scripts.router_data.common import (
    MANIFEST,
    ROOT,
    digest,
    read_labels,
    read_manifest,
)
from scripts.semantic_cache.embedding import MODEL, TASK_TYPE, EmbeddingClient

SEED = 42
TAUS = [i / 100 for i in range(101)]
C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
VARIANTS = (
    "handcrafted_lr",
    "handcrafted_hgb",
    "tfidf_lr",
    "embedding_lr",
    "tfidf_multiclass",
)


def load_data(manifest: Path, labels_dir: Path, models: list[str]) -> list[dict]:
    labels = [read_labels(labels_dir / f"{model}.jsonl") for model in models]
    rows = []
    for question in sorted(read_manifest(manifest), key=lambda q: q["id"]):
        key = question["id"]
        if not all(key in file and file[key]["error"] is None for file in labels):
            continue
        answers = [file[key] for file in labels]
        if any(answer["model"] != model for answer, model in zip(answers, models)):
            raise ValueError(f"Label model mismatch: {key}")
        # Explicit projection prevents prompt/category/dataset from becoming features.
        row = {field: question[field] for field in ("id", "question", "split")}
        row.update(
            correct=[int(answer["correct"]) for answer in answers],
            costs=[float(answer["cost_usd"]) for answer in answers],
            latencies=[float(answer["latency_ms"]) for answer in answers],
        )
        if row["split"] not in {"train", "test", "ood"}:
            raise ValueError(f"Unknown split: {key}")
        if not np.isfinite(row["costs"] + row["latencies"]).all():
            raise ValueError(f"Non-finite cost/latency: {key}")
        rows.append(row)
    if not rows:
        raise ValueError("No successful shared labels")
    return rows


def embedding_features(
    rows: list[dict],  # IDs locate cache entries; only question text is embedded.
    cache_dir: Path,  # Local reusable vectors, excluded from version control.
    client_factory=EmbeddingClient,  # Injectable so tests never call the provider.
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = None
    vectors = []
    cached = 0
    try:
        for n, row in enumerate(rows, start=1):
            # Hash identity for portable filenames; content/model/task changes invalidate.
            path = cache_dir / f"{digest(row['id'])}.npy"
            meta = path.with_suffix(".json")
            signature = dict(
                model=MODEL,
                task_type=TASK_TYPE,
                question_sha256=digest(row["question"]),
            )
            if (
                path.exists()
                and meta.exists()
                and json.loads(meta.read_text()) == signature
            ):
                vector = np.load(path, allow_pickle=False)
                cached += 1
            else:
                if client is None:
                    client = client_factory()
                vector = np.asarray(client.embed(row["question"]), dtype=float)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                    raise ValueError("Invalid embedding")
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, vector, allow_pickle=False)
                temporary.replace(path)
                meta.write_text(json.dumps(signature))
            vectors.append(vector)
            if n % 100 == 0:
                print(
                    f"embedded {n}/{len(rows)} ({cached} cached)",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if client is not None:
            client.close()
    return np.asarray(vectors)


def feature_input(rows: list[dict], variant: str, cache_dir: Path):
    if variant.startswith("handcrafted"):
        return np.asarray([extract_features(row["question"]) for row in rows])
    if variant.startswith("tfidf"):
        return np.asarray([row["question"] for row in rows])
    return embedding_features(rows, cache_dir)


def fit_classifier(x, y, variant: str):  # Only the current training fold is accepted.
    if len(np.unique(y)) == 1:
        if variant == "tfidf_multiclass":
            return None, {"constant_class": int(y[0])}
        return None, {"constant_probability": float(y[0])}
    if variant == "handcrafted_hgb":
        model = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False)
        model.fit(x, y)
        return model, {}
    # Keeping the vectorizer/scaler inside GridSearchCV fits it independently per fold.
    steps = (
        [("features", TfidfVectorizer(ngram_range=(1, 2)))]
        if variant.startswith("tfidf")
        else [("scale", StandardScaler())]
    )
    # lbfgs uses multinomial loss for targets with more than two classes.
    steps.append(
        (
            "classifier",
            LogisticRegression(solver="lbfgs", max_iter=2000, random_state=SEED),
        )
    )
    _, counts = np.unique(y, return_counts=True)
    if min(counts) < 5:
        raise ValueError(
            "Each nonconstant training fold needs at least five examples per class for 5-fold CV"
        )
    search = GridSearchCV(
        Pipeline(steps),
        {"classifier__C": C_VALUES},
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring=multiclass_cv_score
        if variant == "tfidf_multiclass"
        else "neg_brier_score",
        error_score="raise",
    )
    search.fit(x, y)
    return search.best_estimator_, {"C": search.best_params_["classifier__C"]}


def multiclass_cv_score(estimator, x, y) -> float:
    # Align with the estimator's observed classes, including non-contiguous indices.
    actual = np.asarray(y)[:, None] == estimator.classes_
    return -float(np.mean(np.sum((estimator.predict_proba(x) - actual) ** 2, axis=1)))


def probabilities(model, params: dict, x, n_classes: int | None = None):
    if n_classes is not None:
        result = np.zeros((len(x), n_classes))
        if model is None:
            result[:, params["constant_class"]] = 1
        else:
            # Absent classes retain zero probability and their original model index.
            result[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
        return result
    if model is None:
        return np.full(len(x), params["constant_probability"])
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def cascade(
    probabilities_, tau: float
) -> np.ndarray:  # Columns exclude the last/fallback model.
    accepted = np.asarray(probabilities_) >= tau
    return np.where(accepted.any(axis=1), accepted.argmax(axis=1), accepted.shape[1])


def summarize(rows: list[dict], choices: np.ndarray, models: list[str]) -> dict:
    indices = np.arange(len(rows))
    return dict(
        accuracy=float(
            np.asarray([r["correct"] for r in rows])[indices, choices].mean()
        ),
        total_cost=float(
            np.asarray([r["costs"] for r in rows])[indices, choices].sum()
        ),
        mean_latency_ms=float(
            np.asarray([r["latencies"] for r in rows])[indices, choices].mean()
        ),
        shares={model: float(np.mean(choices == i)) for i, model in enumerate(models)},
    )


def sweep(rows: list[dict], predictions: np.ndarray, models: list[str]) -> list[dict]:
    return [
        dict(tau=tau, **summarize(rows, cascade(predictions, tau), models))
        for tau in TAUS
    ]


def select_tau(
    train_rows: list[dict],  # Explicitly reject test/OOD rows at this boundary.
    oof: np.ndarray,  # Paired held-out probabilities, one column per non-final model.
    models: list[str],  # Caller-supplied increasing price order.
) -> dict:
    if not train_rows or any(row["split"] != "train" for row in train_rows):
        raise ValueError("Threshold selection accepts only train rows")
    target = np.mean([row["correct"][-1] for row in train_rows]) - 0.01
    candidates = [
        row
        for row in sweep(train_rows, oof, models)
        if row["accuracy"] >= target - 1e-12
    ]
    if not candidates:
        # tau=1 can still accept an exact probability of 1. Explicit all-last mode
        # guarantees the quality constraint even in this degenerate case.
        return dict(tau=None, all_last=True, target_accuracy=float(target))
    chosen = max(
        candidates,
        key=lambda row: (1 - row["shares"][models[-1]], -row["total_cost"], row["tau"]),
    )
    return dict(
        tau=chosen["tau"], all_last=False, target_accuracy=float(target), oof=chosen
    )


def gap_recovered(accuracy: float, cheapest: float, last: float) -> float | None:
    return (accuracy - cheapest) / (last - cheapest) if last != cheapest else None


def cheapest_correct(rows: list[dict]) -> np.ndarray:
    correct = np.asarray([row["correct"] for row in rows], dtype=bool)
    return np.where(correct.any(axis=1), correct.argmax(axis=1), correct.shape[1] - 1)


def upper_hull(points: list[dict]) -> list[dict]:
    # At equal cost only the most accurate point can belong to the upper envelope.
    unique = {}
    for point in points:
        cost = point["total_cost"]
        if cost not in unique or point["accuracy"] > unique[cost]["accuracy"]:
            unique[cost] = point
    hull = []
    for point in sorted(unique.values(), key=lambda p: p["total_cost"]):
        while len(hull) >= 2:
            a, b = hull[-2:]
            cross = (b["total_cost"] - a["total_cost"]) * (
                point["accuracy"] - b["accuracy"]
            ) - (b["accuracy"] - a["accuracy"]) * (
                point["total_cost"] - b["total_cost"]
            )
            if cross < 0:
                break
            hull.pop()
        hull.append(point)
    return hull


def hull_accuracy(hull: list[dict], cost: float) -> float | None:
    # No mixture of single-model policies exists outside this cost interval.
    if not hull or cost < hull[0]["total_cost"] or cost > hull[-1]["total_cost"]:
        return None
    return float(
        np.interp(cost, [p["total_cost"] for p in hull], [p["accuracy"] for p in hull])
    )


def selected_metrics(selected: dict, reference: dict, models: list[str]) -> dict:
    cheapest, last = (reference["single"][m] for m in (models[0], models[-1]))
    hull_acc = hull_accuracy(reference["hull"], selected["total_cost"])
    oracle_savings = last["total_cost"] - reference["oracle"]["total_cost"]
    return dict(
        selected,
        cost_reduction_vs_last=1 - selected["total_cost"] / last["total_cost"]
        if last["total_cost"]
        else None,
        gap_recovered=gap_recovered(
            selected["accuracy"], cheapest["accuracy"], last["accuracy"]
        ),
        accuracy_vs_hull=selected["accuracy"] - hull_acc
        if hull_acc is not None
        else None,
        oracle_savings_captured=(last["total_cost"] - selected["total_cost"])
        / oracle_savings
        if last["total_cost"] and oracle_savings
        else None,
    )


def baselines(rows: list[dict], models: list[str]) -> dict:
    single = {
        model: summarize(rows, np.full(len(rows), i), models)
        for i, model in enumerate(models)
    }
    # Correctness matrix is used only by this evaluation upper bound, never features.
    return dict(
        single=single,
        hull=upper_hull(list(single.values())),
        oracle=summarize(rows, cheapest_correct(rows), models),
    )


def cumulative_probabilities(predictions: np.ndarray) -> np.ndarray:
    # The existing cascade accepts the first cumulative mass >= tau; the last
    # model remains the fallback, including floating-point roundoff at tau=1.
    return np.cumsum(predictions, axis=1)[:, :-1]


def multiclass_metrics(actual: np.ndarray, predictions: np.ndarray) -> dict:
    one_hot = np.eye(predictions.shape[1])[actual]
    aucs = [
        roc_auc_score(one_hot[:, i], predictions[:, i])
        for i in range(predictions.shape[1])
        if len(np.unique(one_hot[:, i])) > 1
    ]
    return dict(
        roc_auc=float(np.mean(aucs)) if len(aucs) == predictions.shape[1] else None,
        brier=float(np.mean(np.sum((one_hot - predictions) ** 2, axis=1))),
    )


def train_variant(
    rows: list[dict],  # Shared successful labels with fixed manifest splits.
    models: list[str],  # Last model is the untrained fallback.
    variant: str,  # One feature/classifier combination, selected without evaluation data.
    cache_dir: Path,
) -> dict:
    train_rows = [row for row in rows if row["split"] == "train"]
    if len(train_rows) < 10:
        raise ValueError("At least ten shared train examples are required")
    x = feature_input(train_rows, variant, cache_dir)
    multiclass = variant == "tfidf_multiclass"
    n_classes = len(models) if multiclass else None
    y = (
        cheapest_correct(train_rows)[:, None]
        if multiclass
        else np.asarray([row["correct"][:-1] for row in train_rows])
    )
    oof = np.empty((len(train_rows), len(models) - 1), dtype=float)
    fitted, parameters = [], []
    # A common outer partition across models gives genuine paired cascade OOF
    # predictions. Inner five-fold CV chooses C without seeing the outer holdout.
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(x))
    for column in range(y.shape[1]):
        for fit, held in folds:
            classifier, params = fit_classifier(x[fit], y[fit, column], variant)
            predicted = probabilities(classifier, params, x[held], n_classes)
            if multiclass:
                oof[held] = cumulative_probabilities(predicted)
            else:
                oof[held, column] = predicted
        classifier, params = fit_classifier(x, y[:, column], variant)
        fitted.append(classifier)
        parameters.append(params)
    selection = select_tau(train_rows, oof, models)
    result = dict(
        selection=selection,
        classifiers=dict(
            zip(["multiclass"] if multiclass else models[:-1], parameters)
        ),
        evaluation={},
    )
    for split in ("test", "ood"):
        evaluation = [row for row in rows if row["split"] == split]
        if not evaluation:
            result["evaluation"][split] = None
            continue
        inputs = feature_input(evaluation, variant, cache_dir)
        if multiclass:
            class_probabilities = probabilities(
                fitted[0], parameters[0], inputs, n_classes
            )
            predictions = cumulative_probabilities(class_probabilities)
            metrics = {
                "multiclass": multiclass_metrics(
                    cheapest_correct(evaluation), class_probabilities
                )
            }
        else:
            predictions = np.column_stack(
                [probabilities(m, p, inputs) for m, p in zip(fitted, parameters)]
            )
            metrics = {}
            for index, model in enumerate(models[:-1]):
                actual = [row["correct"][index] for row in evaluation]
                metrics[model] = dict(
                    roc_auc=float(roc_auc_score(actual, predictions[:, index]))
                    if len(set(actual)) > 1
                    else None,
                    brier=float(brier_score_loss(actual, predictions[:, index])),
                )
        choices = (
            np.full(len(evaluation), len(models) - 1)
            if selection["all_last"]
            else cascade(predictions, selection["tau"])
        )
        selected = summarize(evaluation, choices, models)
        reference = baselines(evaluation, models)
        selected = selected_metrics(selected, reference, models)
        result["evaluation"][split] = dict(
            n=len(evaluation),
            classifiers=metrics,
            sweep=sweep(evaluation, predictions, models),
            selected=selected,
        )
    return result


def save_results(result: dict, out_dir: Path, run: str) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, plot_path = out_dir / f"{run}.json", out_dir / f"{run}.png"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, split in zip(axes, ("test", "ood")):
        axis.set(title=split, xlabel="Total serving cost (USD)", ylabel="Accuracy")
        reference = result["baselines"][split]
        if reference is None:
            axis.text(
                0.5, 0.5, "No shared labels", transform=axis.transAxes, ha="center"
            )
            continue
        for name, variant in result["variants"].items():
            evaluation = variant["evaluation"][split]
            points = evaluation["sweep"]
            (line,) = axis.plot(
                [p["total_cost"] for p in points],
                [p["accuracy"] for p in points],
                label=name,
            )
            selected = evaluation["selected"]
            axis.scatter(
                selected["total_cost"],
                selected["accuracy"],
                color=line.get_color(),
                marker="x",
            )
        for model, point in reference["single"].items():
            axis.scatter(point["total_cost"], point["accuracy"], label=model)
        hull = reference["hull"]
        axis.plot(
            [p["total_cost"] for p in hull],
            [p["accuracy"] for p in hull],
            "--",
            label="single-model upper hull",
        )
        oracle = reference["oracle"]
        axis.scatter(
            oracle["total_cost"], oracle["accuracy"], marker="*", s=100, label="oracle"
        )
        axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return json_path, plot_path


def log_results(result: dict, paths: tuple[Path, Path], run: str) -> None:
    # This offline workflow explicitly uses the requested local ./mlruns store.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("router-training")
    with mlflow.start_run(run_name=run):
        mlflow.log_params(
            {key: json.dumps(value) for key, value in result["params"].items()}
        )
        for variant, value in result["variants"].items():
            mlflow.log_param(f"{variant}.selection", json.dumps(value["selection"]))
            mlflow.log_param(f"{variant}.classifiers", json.dumps(value["classifiers"]))
            for split, evaluation in value["evaluation"].items():
                if evaluation is None:
                    continue
                for key, metric in evaluation["selected"].items():
                    if isinstance(metric, (float, int)):
                        mlflow.log_metric(f"{variant}.{split}.{key}", metric)
                for model, metrics in evaluation["classifiers"].items():
                    for key, metric in metrics.items():
                        if metric is not None:
                            mlflow.log_metric(
                                f"{variant}.{split}.{model}.{key}", metric
                            )
        for path in paths:
            mlflow.log_artifact(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["local-qwen", "gemini-flash-lite"],
        help="Gateway names ordered cheapest to most expensive",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--labels-dir", type=Path, default=ROOT / "docs/benchmarks/router/labels"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "docs/benchmarks/router/results"
    )
    parser.add_argument(
        "--embedding-cache", type=Path, default=ROOT / "data/router/embeddings"
    )
    parser.add_argument("--run", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    if len(args.models) < 2 or len(set(args.models)) != len(args.models):
        parser.error("Provide at least two distinct models, ordered by price")
    if Path(args.run).name != args.run or args.run in {".", ".."}:
        parser.error("--run must be a filename")
    rows = load_data(args.manifest, args.labels_dir, args.models)
    # Model order is supplied explicitly: token-dependent prices need not impose
    # a universal total ordering. Do not infer order using test/OOD realized costs.
    result = dict(
        params=dict(
            models=args.models,
            seed=SEED,
            cv_folds=5,
            c_values=C_VALUES,
            feature_names=FEATURE_NAMES,
            embedding_model=MODEL,
            embedding_task=TASK_TYPE,
            quality_tolerance=0.01,
            train_count=sum(r["split"] == "train" for r in rows),
            data_sha256=hashlib.sha256(
                json.dumps(rows, sort_keys=True).encode()
            ).hexdigest(),
        ),
        baselines={},
        variants={},
    )
    for split in ("test", "ood"):
        evaluation = [row for row in rows if row["split"] == split]
        result["baselines"][split] = (
            baselines(evaluation, args.models) if evaluation else None
        )
    for variant in VARIANTS:
        started = perf_counter()
        print(f"variant {variant} started (0.0s)", file=sys.stderr, flush=True)
        result["variants"][variant] = train_variant(
            rows, args.models, variant, args.embedding_cache
        )
        print(
            f"variant {variant} finished ({perf_counter() - started:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
    paths = save_results(result, args.out_dir, args.run)
    log_results(result, paths, args.run)
    print(f"Wrote {paths[0]} and {paths[1]}")


if __name__ == "__main__":
    main()
