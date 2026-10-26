"""Offline synthetic tests only: no label corpus training or provider calls."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_gateway.router_features import FEATURE_NAMES, extract_features

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.router_train import train  # noqa: E402


def toy_rows(split="test"):
    return [
        dict(
            id=str(i),
            question=f"Compute {i} plus 2?",
            split=split,
            correct=correct,
            costs=[0, 1, 3],
            latencies=[10, 20, 40],
        )
        for i, correct in enumerate(([1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]))
    ]


def test_features_deterministic_question_only(tmp_path):
    text = "Calculate 12 + 3. How many? Explain why and which."
    a = {
        "question": text,
        "prompt": "ANSWER law",
        "category": "law",
        "dataset": "gsm8k",
    }
    b = dict(a, prompt="completely different", category="history", dataset="mmlupro")
    assert (
        train.feature_input([a], "handcrafted_lr", tmp_path).tolist()
        == train.feature_input([b], "handcrafted_lr", tmp_path).tolist()
    )
    assert train.feature_input([a], "tfidf_lr", tmp_path).tolist() == [text]
    features = dict(zip(FEATURE_NAMES, extract_features(text)))
    assert features["characters"] == len(text)
    assert features["numbers"] == 2
    assert features["digits"] == 3
    assert features["math_operators"] == 1
    assert features["question_marks"] == 1
    assert features["how_many"] == 1
    assert features["calculate"] == 1
    assert extract_features(text) == extract_features(text)
    assert len(extract_features("")) == len(FEATURE_NAMES) == 17
    assert not any(extract_features(""))


def test_cascade_three_models():
    p = np.array([[0.8, 0.9], [0.3, 0.7], [0.2, 0.4], [0.5, 0.1]])
    assert train.cascade(p, 0.5).tolist() == [0, 1, 2, 0]


def test_baselines_and_gap_math():
    result = train.baselines(toy_rows(), ["cheap", "middle", "last"])
    assert result["single"]["cheap"]["accuracy"] == 0.25
    assert result["single"]["last"]["accuracy"] == 0.5
    oracle = result["oracle"]
    assert oracle["accuracy"] == 0.75
    assert oracle["total_cost"] == 7
    assert oracle["mean_latency_ms"] == 27.5
    assert train.hull_accuracy(result["hull"], 6) == 0.375
    selected = train.selected_metrics(
        dict(accuracy=0.5, total_cost=6), result, ["cheap", "middle", "last"]
    )
    assert selected["accuracy_vs_hull"] == 0.125
    assert selected["cost_reduction_vs_last"] == 0.5
    assert selected["oracle_savings_captured"] == 1.2
    assert train.gap_recovered(0.375, 0.25, 0.5) == 0.5
    assert train.gap_recovered(1, 0.5, 0.5) is None


def test_tau_selection_train_only_maximizes_cheap_share():
    rows = toy_rows("train")
    oof = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.2], [0.1, 0.1]])
    result = train.select_tau(rows, oof, ["cheap", "middle", "last"])
    assert result["oof"]["accuracy"] >= 0.49
    assert 1 - result["oof"]["shares"]["last"] == 0.5
    assert result["tau"] == 0.8
    assert result["oof"]["accuracy"] == 0.75
    for split in ("test", "ood"):
        with pytest.raises(ValueError, match="only train"):
            train.select_tau(toy_rows(split), oof, ["cheap", "middle", "last"])
    bad = [dict(row, correct=[0, 0, 1]) for row in rows]
    assert train.select_tau(bad, np.ones((4, 2)), ["cheap", "middle", "last"])[
        "all_last"
    ]


@pytest.mark.parametrize("variant", ["tfidf_lr", "tfidf_multiclass"])
def test_nested_folds_and_selection_never_receive_evaluation(
    monkeypatch, tmp_path, variant
):
    rows = [
        dict(
            id=str(i),
            question=f"trainword {i}",
            split="train",
            correct=[i % 2, 1],
            costs=[0, 1],
            latencies=[1, 2],
        )
        for i in range(50)
    ]
    rows += [
        dict(rows[0], id=split, question=f"{split}secret", split=split)
        for split in ("test", "ood")
    ]
    fits = []

    class Predictor:
        classes_ = [0, 1]

        def __init__(self, fitted):
            self.fitted = set(fitted)

        def predict_proba(self, x):
            # Outer validation observations were never used in this fit.
            assert not self.fitted.intersection(x)
            return np.tile([0.5, 0.5], (len(x), 1))

    def fit(x, y, variant):
        assert all(text.startswith("trainword") for text in x)
        fits.append(len(x))
        return Predictor(x), {"C": 1}

    original = train.select_tau

    def select(selected, predictions, models):
        assert {row["split"] for row in selected} == {"train"}
        assert len(selected) == 50
        return original(selected, predictions, models)

    monkeypatch.setattr(train, "fit_classifier", fit)
    monkeypatch.setattr(train, "select_tau", select)
    result = train.train_variant(rows, ["a", "b"], variant, tmp_path)
    assert fits == [40] * 5 + [50]
    assert set(result["evaluation"]) == {"test", "ood"}
    assert len(result["evaluation"]["test"]["sweep"]) == 101
    classifier = "multiclass" if variant == "tfidf_multiclass" else "a"
    assert result["evaluation"]["test"]["classifiers"][classifier]["roc_auc"] is None


@pytest.mark.parametrize("variant", train.VARIANTS)
def test_classifier_on_synthetic_features(variant, monkeypatch):
    monkeypatch.setattr(train, "C_VALUES", [0.1, 1])
    y = np.array([0, 1] * 20)
    x = (
        np.array(["plain wording" if i == 0 else "complex wording" for i in y])
        if variant.startswith("tfidf")
        else np.column_stack([y, y * 2])
    )
    model, params = train.fit_classifier(x, y, variant)
    n_classes = 2 if variant == "tfidf_multiclass" else None
    p = train.probabilities(model, params, x, n_classes)
    assert p.shape == ((40, 2) if n_classes else (40,))
    assert np.isfinite(p).all()
    if variant.startswith("tfidf"):
        assert "unseen" not in model.named_steps["features"].vocabulary_
        assert "plain wording" in model.named_steps["features"].vocabulary_
        train.probabilities(model, params, np.array(["unseen evaluation"]))
        assert "unseen" not in model.named_steps["features"].vocabulary_
    constant, params = train.fit_classifier(x, np.ones(40), variant)
    expected = [[0.0, 1.0]] * 40 if n_classes else [1.0] * 40
    assert train.probabilities(constant, params, x, n_classes).tolist() == expected


def test_embedding_cache_question_only_invalidation_and_close(tmp_path):
    client = Mock()
    client.embed.return_value = [0.1, 0.2]
    factory = Mock(return_value=client)
    rows = [
        {"id": "unsafe/id", "question": "Raw question", "prompt": "Never embed this"}
    ]
    first = train.embedding_features(rows, tmp_path, factory)
    second = train.embedding_features(rows, tmp_path, factory)
    assert np.array_equal(first, second)
    factory.assert_called_once()
    client.embed.assert_called_once_with("Raw question")
    client.close.assert_called_once()
    train.embedding_features([dict(rows[0], question="Changed")], tmp_path, factory)
    assert client.embed.call_count == 2
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_shared_latest_successful_labels(tmp_path):
    manifest = tmp_path / "questions.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(dict(row, prompt="secret", category="law", dataset="fake"))
            for row in toy_rows()
        )
    )
    for model in ("a", "b"):
        labels = [
            dict(
                id=str(i),
                model=model,
                error=None,
                correct=True,
                cost_usd="0.01",
                latency_ms=2,
            )
            for i in range(3)
        ]
        if model == "a":
            labels.append(dict(labels[1], error="failed"))
        (tmp_path / f"{model}.jsonl").write_text("\n".join(map(json.dumps, labels)))
    rows = train.load_data(manifest, tmp_path, ["a", "b"], pricing="billed")
    assert [row["id"] for row in rows] == ["0", "2"]
    assert not {"prompt", "category", "dataset"}.intersection(rows[0])


def test_artifacts_and_local_mlflow(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    # No classifier training: this fixture checks the output and logging path.
    rows = toy_rows()
    models = ["a", "b", "c"]
    reference = train.baselines(rows, models)
    point = dict(reference["single"]["a"], gap_recovered=0, cost_reduction_vs_last=1)
    evaluation = dict(
        sweep=[dict(tau=0, **point)],
        selected=point,
        classifiers={"a": {"roc_auc": None, "brier": 0.25}},
    )
    result = dict(
        params={"models": models},
        baselines={"test": reference, "ood": None},
        variants={
            "fixture": dict(
                selection={"tau": 0},
                classifiers={},
                evaluation={"test": evaluation, "ood": None},
            )
        },
    )
    paths = train.save_results(result, tmp_path / "results", "fixture")
    assert json.loads(paths[0].read_text())["params"]["models"] == models
    assert paths[1].read_bytes().startswith(b"\x89PNG")
    train.log_results(result, paths, "fixture")
    experiment = mlflow.get_experiment_by_name("router-training")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.fixture.test.accuracy"] == 0.25


def test_cli_with_synthetic_three_model_corpus(tmp_path, monkeypatch, capsys):
    """Exercise all variants and output wiring without real training data or HTTP."""
    manifest = tmp_path / "questions.jsonl"
    questions = [
        dict(
            id=str(i),
            question=f"Calculate item {i} plus two?",
            split="train" if i < 60 else "test" if i < 70 else "ood",
        )
        for i in range(80)
    ]
    manifest.write_text("\n".join(map(json.dumps, questions)))
    models = ["cheap", "middle", "last"]
    for index, model in enumerate(models):
        rows = [
            dict(
                id=str(i),
                model=model,
                error=None,
                correct=(i + index) % 2 == 0,
                cost_usd=str(index),
                latency_ms=index + 1,
            )
            for i in range(80)
        ]
        (tmp_path / f"{model}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "C_VALUES", [1])
    monkeypatch.setattr(
        train,
        "embedding_features",
        lambda rows, _: np.array([[len(r["question"]), 1] for r in rows]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--pricing",
            "billed",
            "--models",
            *models,
            "--manifest",
            str(manifest),
            "--labels-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "output"),
            "--run",
            "synthetic",
        ],
    )
    clock = iter(t for i in range(len(train.VARIANTS)) for t in (2 * i, 2 * i + 1.5))
    monkeypatch.setattr(train, "perf_counter", lambda: next(clock))
    train.main()
    progress = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("variant ")
    ]
    assert progress == [
        f"variant {variant} {state} ({seconds}s)"
        for variant in train.VARIANTS
        for state, seconds in (("started", "0.0"), ("finished", "1.5"))
    ]
    output = json.loads((tmp_path / "output/synthetic.json").read_text())
    assert output["params"]["pricing"] == "billed"
    assert output["params"]["price_table"] == train.price_table()
    assert set(output["variants"]) == set(train.VARIANTS)
    for name, variant in output["variants"].items():
        assert set(variant["classifiers"]) == (
            {"multiclass"} if name == "tfidf_multiclass" else {"cheap", "middle"}
        )
        for split in ("test", "ood"):
            evaluation = variant["evaluation"][split]
            assert evaluation["n"] == 10
            assert "accuracy_vs_hull" in evaluation["selected"]
            assert "oracle_savings_captured" in evaluation["selected"]
            assert len(evaluation["sweep"]) == 101
            assert sum(evaluation["selected"]["shares"].values()) == pytest.approx(1)


def test_embedding_progress_counts_cached_questions_every_100(tmp_path, capsys):
    client = Mock()
    client.embed.return_value = [0.1, 0.2]
    factory = Mock(return_value=client)
    rows = [dict(id=str(i), question=f"Question {i}") for i in range(250)]
    train.embedding_features(rows[:100], tmp_path, factory)
    capsys.readouterr()
    train.embedding_features(rows, tmp_path, factory)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.splitlines() == [
        "embedded 100/250 (100 cached)",
        "embedded 200/250 (100 cached)",
    ]
    assert client.embed.call_count == 250
    train.embedding_features(rows, tmp_path, factory)
    assert capsys.readouterr().err.splitlines() == [
        "embedded 100/250 (100 cached)",
        "embedded 200/250 (200 cached)",
    ]
    assert factory.call_count == 2


def test_upper_hull_uses_middle_model_and_interpolates():
    points = [
        dict(total_cost=x, accuracy=y)
        for x, y in [(0, 0.2), (2, 0.8), (3, 0.7), (4, 0.9), (2, 0.4)]
    ]
    hull = train.upper_hull(list(reversed(points)))
    assert hull == [points[0], points[1], points[3]]
    assert train.hull_accuracy(hull, 1) == pytest.approx(0.5)
    assert train.hull_accuracy(hull, 3) == pytest.approx(0.85)
    assert train.hull_accuracy(hull, 2) == 0.8
    assert train.hull_accuracy(hull, -1) is None
    assert train.hull_accuracy(hull, 5) is None
    two = train.upper_hull([points[0], points[3]])
    for share in train.TAUS:
        assert train.hull_accuracy(two, 4 * share) == pytest.approx(
            0.2 * (1 - share) + 0.9 * share
        )
    assert train.upper_hull(
        [dict(total_cost=1, accuracy=0.2), dict(total_cost=1, accuracy=0.5)]
    ) == [dict(total_cost=1, accuracy=0.5)]
    # Collinear and dominated interior points do not affect the envelope.
    assert (
        len(train.upper_hull([dict(total_cost=i, accuracy=i / 10) for i in range(3)]))
        == 2
    )
    assert train.hull_accuracy([dict(total_cost=1, accuracy=0.5)], 1) == 0.5


def test_selected_metrics_zero_denominators_and_outside_hull():
    models = ["a", "b", "c"]
    rows = [dict(row, costs=[0, 0, 0]) for row in toy_rows()]
    result = train.selected_metrics(
        dict(accuracy=0.5, total_cost=0), train.baselines(rows, models), models
    )
    assert result["cost_reduction_vs_last"] is None
    assert result["oracle_savings_captured"] is None
    assert result["accuracy_vs_hull"] == 0
    rows = [dict(row, correct=[0, 0, 1]) for row in toy_rows()]
    result = train.selected_metrics(
        dict(accuracy=0.5, total_cost=20), train.baselines(rows, models), models
    )
    assert result["oracle_savings_captured"] is None
    assert result["accuracy_vs_hull"] is None


def test_multiclass_targets_and_cumulative_routing():
    assert train.cheapest_correct(toy_rows()).tolist() == [0, 1, 2, 2]
    rows = [dict(row, correct=[1, 1, 1]) for row in toy_rows()]
    assert train.cheapest_correct(rows).tolist() == [0] * 4
    p = np.array([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    cumulative = train.cumulative_probabilities(p)
    assert train.cascade(cumulative, 0.6).tolist() == [1, 0, 2]
    assert train.cascade(cumulative, 0).tolist() == [0, 0, 0]
    assert train.cascade(cumulative, 1).tolist() == [2, 2, 2]
    assert train.cascade(cumulative, 0.2).tolist() == [0, 0, 1]


def test_multiclass_three_classes_and_absent_class_alignment(monkeypatch):
    monkeypatch.setattr(train, "C_VALUES", [0.1, 1])
    y = np.tile([0, 1, 2], 20)
    x = np.array([f"classword{target} example" for target in y])
    model, params = train.fit_classifier(x, y, "tfidf_multiclass")
    p = train.probabilities(model, params, x, 3)
    assert p.shape == (60, 3)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.array_equal(p.argmax(axis=1), y)
    assert train.multiclass_metrics(y, p)["roc_auc"] == 1
    assert train.multiclass_cv_score(model, x, y) == pytest.approx(
        -train.multiclass_metrics(y, p)["brier"]
    )
    # A fold can omit an entire model class; do not shift later model indices.
    present = y != 1
    model, params = train.fit_classifier(x[present], y[present], "tfidf_multiclass")
    p = train.probabilities(model, params, x, 3)
    assert np.all(p[:, 1] == 0)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.array_equal(p[present].argmax(axis=1), y[present])
    constant, params = train.fit_classifier(x, np.full(60, 2), "tfidf_multiclass")
    p = train.probabilities(constant, params, x, 3)
    assert p.tolist() == [[0, 0, 1]] * 60
    assert train.multiclass_metrics(np.full(60, 2), p) == {
        "roc_auc": None,
        "brier": 0.0,
    }


def test_multiclass_metric_math():
    y = np.array([0, 1, 2])
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.1, 0.2, 0.7]])
    metrics = train.multiclass_metrics(y, p)
    assert metrics["roc_auc"] == 1
    assert metrics["brier"] == pytest.approx((0.38 + 0.26 + 0.14) / 3)


@pytest.mark.parametrize("pricing,expected", [("list", 16.5), ("billed", 0.01)])
def test_pricing_loads_costs_without_changing_labels(tmp_path, pricing, expected):
    manifest = tmp_path / "questions.jsonl"
    manifest.write_text(json.dumps(dict(id="q", question="why?", split="train")) + "\n")
    label = dict(
        id="q",
        model="gemini-flash",
        error=None,
        correct=True,
        cost_usd="0.01",
        prompt_tokens=1_000_000,
        completion_tokens=2_000_000,
        latency_ms=10,
    )
    path = tmp_path / "gemini-flash.jsonl"
    original = json.dumps(label) + "\n"
    path.write_text(original)
    rows = train.load_data(manifest, tmp_path, ["gemini-flash"], pricing)
    assert rows[0]["costs"] == [expected]
    assert path.read_text() == original
    assert train.load_data(manifest, tmp_path, ["gemini-flash"])[0]["costs"] == [16.5]


def test_list_prices_require_model_and_token_counts():
    with pytest.raises(ValueError, match="No list price"):
        train.evaluation_cost({}, "unknown", "list", train.price_table())
    with pytest.raises(KeyError, match="prompt_tokens"):
        train.evaluation_cost({}, "gemini-flash", "list", train.price_table())
