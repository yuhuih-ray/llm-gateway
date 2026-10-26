"""Pure-Python serving tests: no NumPy, sklearn, or router extras required."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_gateway.router import Router

FIXTURES = Path(__file__).parent / "fixtures/router"


def test_probabilities_match_sklearn_parity_fixture():
    router = Router(FIXTURES / "synthetic.json")
    fixture = json.loads((FIXTURES / "synthetic-parity.json").read_text())
    assert len(fixture["questions"]) == 200
    assert list(router.models) == fixture["models"]
    for question in fixture["questions"]:
        decision = router.route(question["question"])
        assert list(decision.probabilities.values()) == pytest.approx(
            question["probabilities"], abs=1e-9, rel=0
        )


def test_levels_and_load_once(tmp_path):
    artifact = json.loads((FIXTURES / "synthetic.json").read_text())
    artifact["levels"] = {
        "high": 0.9,
        "balanced": 0.5,
        "economy": 0.1,
        "fallback": None,
        "tie": 0.5,
    }
    for model, probability in zip(artifact["models"], [0.5, 0.8]):
        classifier = artifact["classifiers"][model]
        classifier["coefficients"] = [0.0] * len(classifier["coefficients"])
        classifier["intercept"] = math.log(probability / (1 - probability))
    path = tmp_path / "router.json"
    path.write_text(json.dumps(artifact))
    router = Router(path)
    path.unlink()  # Routing never reads or reloads the artifact.
    expected = {
        "high": "last",
        "balanced": "cheap",
        "economy": "cheap",
        "fallback": "last",
        "tie": "cheap",
    }
    for level, model in expected.items():
        decision = router.route("Any question?", level)
        assert decision.model == model
        assert decision.tau == artifact["levels"][level]
        assert decision.probabilities == pytest.approx({"cheap": 0.5, "middle": 0.8})
    router.levels["middle"] = 0.7
    assert router.route("", "middle").model == "middle"
    with pytest.raises(ValueError, match="Unknown router level"):
        router.route("question", "unknown")


def test_missing_artifact_is_clear(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="Router artifact not found:.*missing.json.*export.py"
    ):
        Router(tmp_path / "missing.json")


def test_runtime_runs_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from llm_gateway.router import Router
router = Router(sys.argv[1])
assert router.route('blue widgets').model in router.models
assert not any(name.split('.')[0] in {'numpy', 'sklearn'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(FIXTURES / "synthetic.json")],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("intercept,expected", [(1000, 1), (-1000, 0)])
def test_sigmoid_does_not_overflow(tmp_path, intercept, expected):
    artifact = json.loads((FIXTURES / "synthetic.json").read_text())
    for classifier in artifact["classifiers"].values():
        classifier["coefficients"] = [0.0] * len(classifier["coefficients"])
        classifier["intercept"] = intercept
    path = tmp_path / "extreme.json"
    path.write_text(json.dumps(artifact))
    assert set(Router(path).route("widgets").probabilities.values()) == {expected}
