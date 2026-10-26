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
    mix = result["random_mix"][50]
    assert mix["accuracy"] == 0.375
    assert mix["total_cost"] == 6
    assert mix["mean_latency_ms"] == 25
    assert mix["shares"] == {"cheap": 0.5, "middle": 0, "last": 0.5}
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


def test_nested_folds_and_selection_never_receive_evaluation(monkeypatch, tmp_path):
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
    result = train.train_variant(rows, ["a", "b"], "tfidf_lr", tmp_path)
    assert fits == [40] * 5 + [50]
    assert set(result["evaluation"]) == {"test", "ood"}
    assert len(result["evaluation"]["test"]["sweep"]) == 101
    assert result["evaluation"]["test"]["classifiers"]["a"]["roc_auc"] is None


@pytest.mark.parametrize("variant", train.VARIANTS)
def test_classifier_on_synthetic_features(variant, monkeypatch):
    monkeypatch.setattr(train, "C_VALUES", [0.1, 1])
    y = np.array([0, 1] * 20)
    x = (
        np.array(["plain wording" if i == 0 else "complex wording" for i in y])
        if variant == "tfidf_lr"
        else np.column_stack([y, y * 2])
    )
    model, params = train.fit_classifier(x, y, variant)
    p = train.probabilities(model, params, x)
    assert p.shape == (40,)
    assert np.isfinite(p).all()
    if variant == "tfidf_lr":
        assert "unseen" not in model.named_steps["features"].vocabulary_
        assert "plain wording" in model.named_steps["features"].vocabulary_
        train.probabilities(model, params, np.array(["unseen evaluation"]))
        assert "unseen" not in model.named_steps["features"].vocabulary_
    constant, params = train.fit_classifier(x, np.ones(40), variant)
    assert train.probabilities(constant, params, x).tolist() == [1.0] * 40


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
    rows = train.load_data(manifest, tmp_path, ["a", "b"])
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


def test_cli_with_synthetic_three_model_corpus(tmp_path, monkeypatch):
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
    train.main()
    output = json.loads((tmp_path / "output/synthetic.json").read_text())
    assert set(output["variants"]) == set(train.VARIANTS)
    for variant in output["variants"].values():
        assert set(variant["classifiers"]) == {"cheap", "middle"}
        for split in ("test", "ood"):
            evaluation = variant["evaluation"][split]
            assert evaluation["n"] == 10
            assert len(evaluation["sweep"]) == 101
            assert sum(evaluation["selected"]["shares"].values()) == pytest.approx(1)
