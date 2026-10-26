"""Synthetic exporter tests; optional router group only, no provider calls."""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from llm_gateway.router import Router  # noqa: E402
from scripts.router_train import export, train  # noqa: E402


def synthetic_rows():
    samples = [
        "Calculate blue blue blue widgets 12 + 4?",
        "Explain why green widgets differ from blue.",
        "Which red widgets? WHY?",
        "",
        "x ! ?",
        "UNKNOWN zygomorphic",
        "CAFÉ naïve blue_blue 123",
        "calculate\nBLUE widgets\twidgets",
        "blue blue green red",
    ]
    rows = []
    for split, count in (("train", 90), ("test", 200), ("ood", 12)):
        for i in range(count):
            text = samples[i % (3 if split == "train" else len(samples))]
            rows.append(
                dict(
                    id=f"{split}:{i:03}",
                    question=text,
                    split=split,
                    correct=[int(i % 3 == 0), int(i % 3 < 2), 1],
                    costs=[0, 1, 3],
                    latencies=[10, 20, 40],
                )
            )
    return rows


def test_export_synthetic_artifact_and_report(tmp_path, monkeypatch, capsys):
    rows = synthetic_rows()
    models = ["cheap", "middle", "last"]
    monkeypatch.setattr(train, "C_VALUES", [0.1, 1])
    fit_original, select_original = train.fit_cascade, train.select_tau
    fitting = []
    tolerances = []

    def fit(selected, names, variant, cache_dir):
        assert {row["split"] for row in selected} == {"train"}
        assert names == models and variant == "tfidf_lr"
        fitting.append(1)
        return fit_original(selected, names, variant, cache_dir)

    def select(selected, oof, names, allowed_drop):
        assert {row["split"] for row in selected} == {"train"}
        assert len(selected) == 90 and oof.shape == (90, 2)
        tolerances.append(allowed_drop)
        return select_original(selected, oof, names, allowed_drop)

    monkeypatch.setattr(train, "fit_cascade", fit)
    monkeypatch.setattr(train, "select_tau", select)
    artifact, report = export.export_router(
        rows,
        models,
        export.parse_levels(export.DEFAULT_LEVELS),
        "synthetic",
        tmp_path,
        tmp_path,
        pricing="billed",
        prices=train.price_table(),
    )
    assert report["metadata"]["pricing"] == "billed"
    assert report["metadata"]["price_table"] == train.price_table()
    assert fitting == [1]
    assert tolerances == [0.01, 0.03, 0.08]
    assert artifact["metadata"]["feature"] == "question text"
    assert len(artifact["metadata"]["data_sha256"]) == 64
    assert set(artifact["classifiers"]) == {"cheap", "middle"}
    fixture = json.loads((tmp_path / "synthetic-parity.json").read_text())
    assert len(fixture["questions"]) == 200
    assert all(q["id"].startswith("test:") for q in fixture["questions"])
    router = Router(tmp_path / "synthetic.json")
    for question in fixture["questions"]:
        decision = router.route(question["question"])
        assert list(decision.probabilities.values()) == pytest.approx(
            question["probabilities"], abs=1e-9, rel=0
        )
    for split in ("test", "ood"):
        for name, metrics in report["evaluation"][split].items():
            assert metrics["tau"] == artifact["levels"][name]
            assert metrics["accuracy_retained"] == metrics["accuracy"]
            assert metrics["last_model_accuracy"] == 1
            assert metrics["mean_latency_vs_last"] == metrics["mean_latency_ms"] / 40
            assert sum(metrics["shares"].values()) == pytest.approx(1)
            assert (
                "accuracy_vs_hull" in metrics and "oracle_savings_captured" in metrics
            )
    assert json.loads((tmp_path / "synthetic-levels.json").read_text()) == report
    printed = capsys.readouterr().out
    assert printed.startswith("| Level | Split | Tau |")
    assert "| high | test |" in printed and "| economy | ood |" in printed


def test_fixed_levels_score_each_split_once_and_report_math():
    rows = [
        dict(row, split=split, id=f"{split}:{row['question']}")
        for split in ("test", "ood")
        for row in [
            dict(question="one", correct=[1, 1], costs=[1, 3], latencies=[10, 30]),
            dict(question="two", correct=[0, 1], costs=[1, 3], latencies=[10, 30]),
        ]
    ]

    class Predictor:
        classes_ = [0, 1]
        calls = 0

        def predict_proba(self, x):
            self.calls += 1
            return np.array([[0.1, 0.9], [0.8, 0.2]])

    model = Predictor()
    result, _ = export.evaluate_levels(
        rows, ["a", "b"], [model], [{}], {"high": 0.8, "economy": 0.1, "fallback": None}
    )
    assert model.calls == 2
    high = result["test"]["high"]
    assert high["accuracy"] == high["accuracy_retained"] == 1
    assert high["cost_reduction_vs_last"] == pytest.approx(1 / 3)
    assert high["mean_latency_vs_last"] == pytest.approx(2 / 3)
    assert high["accuracy_vs_hull"] == 0.25
    assert high["oracle_savings_captured"] == 1
    assert high["shares"] == {"a": 0.5, "b": 0.5}
    assert result["ood"]["economy"]["accuracy_retained"] == 0.5
    assert result["test"]["fallback"]["shares"] == {"a": 0.0, "b": 1.0}
    zero = [dict(row, correct=[0, 0], latencies=[0, 0]) for row in rows]
    result, _ = export.evaluate_levels(zero, ["a", "b"], [model], [{}], {"high": 0.8})
    assert result["test"]["high"]["accuracy_retained"] is None
    assert result["test"]["high"]["mean_latency_vs_last"] is None


def test_allowed_drop_controls_train_threshold():
    rows = [
        dict(split="train", correct=[int(i < 80), 1], costs=[0, 1], latencies=[1, 2])
        for i in range(100)
    ]
    oof = np.array([[0.9] if i < 80 else [0.4] for i in range(100)])
    assert train.select_tau(rows, oof, ["a", "b"], allowed_drop=0.01)["tau"] == 0.9
    assert train.select_tau(rows, oof, ["a", "b"], allowed_drop=0.3)["tau"] == 0.4
    with pytest.raises(ValueError, match="only train"):
        train.select_tau(
            [dict(row, split="test") for row in rows], oof, ["a", "b"], allowed_drop=0.3
        )


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"lowercase": False},
        {"sublinear_tf": True},
        {"norm": None},
        {"norm": "l1"},
        {"use_idf": False},
        {"smooth_idf": False},
        {"token_pattern": r"(?u)\b(\w+)\b", "ngram_range": (1, 3)},
    ],
)
def test_vectorizer_settings_parity(settings, tmp_path):
    texts = ["Blue blue widgets", "red green café", "BLUE green", "red red widgets"]
    fitted = Pipeline(
        [
            ("features", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression()),
        ]
    )
    fitted.named_steps["features"].set_params(**settings)
    fitted.fit(texts, [1, 0, 1, 0])
    data = dict(
        version=1,
        models=["a", "b"],
        levels={"balanced": 0.5},
        classifiers={"a": export.serialize_classifier(fitted, {})},
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    router = Router(path)
    samples = texts + ["", "? x", "café BLUE BLUE unknown", "widgets\nRED"]
    for text, expected in zip(samples, fitted.predict_proba(samples)[:, 1]):
        assert router.route(text).probabilities["a"] == pytest.approx(
            expected, abs=1e-9, rel=0
        )


@pytest.mark.parametrize("constant", [0, 1])
def test_constant_classifier_export(constant, tmp_path):
    artifact = dict(
        version=1,
        models=["a", "b"],
        levels={"balanced": 0.5},
        classifiers={
            "a": export.serialize_classifier(None, {"constant_probability": constant})
        },
    )
    path = tmp_path / "constant.json"
    path.write_text(json.dumps(artifact))
    assert Router(path).route("any question").probabilities == {"a": constant}


@pytest.mark.parametrize(
    "levels",
    [
        ["high=nan"],
        ["high=2"],
        ["high=-1"],
        ["high"],
        ["=0.1"],
        ["high=0.1", "high=0.2"],
    ],
)
def test_invalid_levels(levels):
    with pytest.raises(ValueError):
        export.parse_levels(levels)


@pytest.mark.parametrize("pricing", ["list", "billed"])
def test_cli_passes_models_levels_and_output_paths(monkeypatch, tmp_path, pricing):
    from unittest.mock import Mock

    loader = Mock(return_value=synthetic_rows())
    writer = Mock()
    monkeypatch.setattr(export, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_data", loader)
    monkeypatch.setattr(export, "export_router", writer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export.py",
            *(["--pricing", pricing] if pricing == "billed" else []),
            "--models",
            "cheap",
            "middle",
            "last",
            "--levels",
            "high=0",
            "custom=0.2",
            "--run",
            "fixture",
        ],
    )
    export.main()
    loader.assert_called_once_with(
        export.MANIFEST,
        tmp_path / "docs/benchmarks/router/labels",
        ["cheap", "middle", "last"],
        pricing,
        train.price_table(),
    )
    assert writer.call_args.kwargs == {
        "pricing": pricing,
        "prices": train.price_table(),
    }
    assert writer.call_args.args[1:] == (
        ["cheap", "middle", "last"],
        {"high": 0.0, "custom": 0.2},
        "fixture",
        tmp_path / "src/llm_gateway/router_artifacts",
        tmp_path / "docs/benchmarks/router/results",
    )
