import asyncio
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from llm_gateway import main

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.router_monitor import drift, send_batch


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_psi_hand_computed_and_boundary_bins():
    reference, current = [0, 0.2, 0.3, 1], [0.1, 0.6, 0.7, 1]
    assert drift.psi(reference, current, [0, 0.5, 1]) == pytest.approx(math.log(3))
    assert drift.proportions([0, 0.5, 1], [0, 0.5, 1]) == [1 / 3, 2 / 3]
    assert drift.psi([0], [1], [0, 0.5, 1]) == pytest.approx(
        2 * (1 - 1e-4) * math.log(1e4)
    )
    assert drift.psi(reference, reference, drift.PROBABILITY_BINS) == 0


def test_compare_and_insufficient_observations():
    reference = [dict(oov_ratio=0.01, probabilities={"local": 0.9})] * 50
    current = [dict(oov_ratio=0.9, probabilities={"local": 0.9})] * 50
    result = drift.compare(reference, current, ["local"], 50)
    assert result["oov_ratio"]["status"] == "drift"
    assert result["local"]["status"] == "stable"
    assert result["oov_ratio"]["current_mean"] == 0.9
    with pytest.raises(ValueError, match="Insufficient requests"):
        drift.compare(reference, current[:49], ["local"], 50)
    with pytest.raises(ValueError, match="Insufficient observations"):
        drift.compare(reference, [dict(oov_ratio=None)] * 50, ["local"], 50)


def test_reference_reads_only_test_question(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(dict(id=str(i), split=split, question="raw", prompt="forbidden"))
            for i, split in enumerate(["test", "train", "ood"])
        )
    )
    router = Mock()
    router.route.return_value.oov_ratio = 0.2
    router.route.return_value.probabilities = {"local": 0.5}
    assert drift.reference_rows(router, path) == [
        dict(oov_ratio=0.2, probabilities={"local": 0.5})
    ]
    router.route.assert_called_once_with("raw")


@pytest.mark.anyio
async def test_send_batch_with_mock_transport(capsys):
    inflight = peak = 0
    seen = []

    async def handler(request):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        seen.append(request)
        await asyncio.sleep(0.001)
        inflight -= 1
        return httpx.Response(200, headers={"X-Gateway-Model": "local-qwen"})

    async with httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handler)
    ) as client:
        result = await send_batch.send_batch(
            client, [dict(question="raw", prompt="forbidden")] * 5, "high", 2
        )
    assert peak == 2
    assert result["shares"] == {"local-qwen": 1}
    assert datetime.fromisoformat(result["start"]) <= datetime.fromisoformat(
        result["end"]
    )
    for request in seen:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["messages"] == [dict(role="user", content="raw")]
        assert payload["model"] == "auto" and payload["max_tokens"] == 64
        assert request.headers["X-Gateway-Quality"] == "high"
        assert "X-Gateway-Cache" not in request.headers
    assert "start:" in capsys.readouterr().out


@pytest.mark.anyio
async def test_preload_failure_does_not_stop_startup(monkeypatch, tmp_path, caplog):
    artifact = tmp_path / "invalid.json"
    artifact.write_text("invalid")
    settings = main.get_settings().model_copy(update={"router_artifact": artifact})
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    loader = Mock(side_effect=ValueError("invalid artifact"))
    monkeypatch.setattr(main, "load_router", loader)
    async with main.app.router.lifespan_context(main.app):
        assert main.app.state.accounting is not None
    loader.assert_called_once_with(artifact)
    assert "router_preload_failed" in caplog.text


@pytest.mark.anyio
async def test_preload_runs_off_event_loop(monkeypatch):
    import threading

    thread_id = threading.get_ident()
    called = []

    def load(path):
        called.append((path, threading.get_ident()))

    monkeypatch.setattr(main, "load_router", load)
    async with main.app.router.lifespan_context(main.app):
        pass
    assert called and called[0][1] != thread_id


def test_sample_is_reproducible_and_respects_split():
    path = Path("docs/benchmarks/router/questions.jsonl")
    first = send_batch.sample_questions(path, "ood", 5, 42)
    assert first == send_batch.sample_questions(path, "ood", 5, 42)
    assert all(row["split"] == "ood" for row in first)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "score,baseline,status,exit_code",
    [
        (0.099, 0, "stable", 0),
        (0.1, 0, "watch", 0),
        (0.2, 0, "drift", 1),
        (0.1, 0.1, "stable", 0),
        (0.2, 0.2, "stable", 0),
        (0.3, 0.4, "stable", 0),
    ],
)
async def test_report_json_status_and_exit_code(
    monkeypatch, tmp_path, capsys, score, baseline, status, exit_code
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    router = Mock(run_name="run", classifiers={"local": None})
    rows = [dict(oov_ratio=0.1, probabilities={"local": 0.8})] * 50
    monkeypatch.setattr(drift, "Router", lambda _: router)
    monkeypatch.setattr(drift, "reference_rows", lambda *_: rows)
    monkeypatch.setattr(drift, "current_rows", AsyncMock(return_value=rows))
    monkeypatch.setattr(drift, "get_sessionmaker", Mock())
    engine = Mock(dispose=AsyncMock())
    monkeypatch.setattr(drift, "get_engine", lambda: engine)
    monkeypatch.setattr(drift, "psi", lambda *_: score)
    monkeypatch.setattr(drift, "null_psi_p95", lambda *_: baseline)
    args = SimpleNamespace(
        artifact=tmp_path / "artifact",
        manifest=tmp_path / "manifest",
        since=drift.timestamp("2026-10-26T00:00:00Z"),
        until=drift.timestamp("2026-10-27T00:00:00Z"),
        min_requests=50,
        out=tmp_path / "output/report.json",
    )
    assert await drift.run(args) == exit_code
    report = json.loads(args.out.read_text())
    assert report["artifact_run"] == "run"
    assert report["features"]["local"]["status"] == status
    assert report["features"]["local"]["null_psi_p95"] == baseline
    assert "Null PSI p95" in capsys.readouterr().out
    engine.dispose.assert_awaited_once()


def test_bootstrap_seed_sample_size_and_percentile(monkeypatch):
    samples = []

    def score(reference, current, edges):
        samples.append(current)
        return len(samples)

    monkeypatch.setattr(drift, "psi", score)
    assert drift.null_psi_p95([0.1, 0.2, 0.3], 7, [0, 1]) == 475
    assert len(samples) == 500
    assert all(
        len(sample) == 7 and set(sample) <= {0.1, 0.2, 0.3} for sample in samples
    )
    first = samples.copy()
    samples.clear()
    drift.null_psi_p95([0.1, 0.2, 0.3], 7, [0, 1])
    assert samples == first


def test_bootstrap_small_window_has_higher_null_psi():
    reference = [i / 10 + 0.05 for i in range(10)] * 10
    small = drift.null_psi_p95(reference, 10, drift.PROBABILITY_BINS)
    large = drift.null_psi_p95(reference, 300, drift.PROBABILITY_BINS)
    assert small > large
    assert drift.null_psi_p95([0.5] * 10, 300, drift.PROBABILITY_BINS) == 0


def test_default_minimum_is_300(monkeypatch):
    from unittest.mock import AsyncMock

    runner = AsyncMock(return_value=0)
    monkeypatch.setattr(drift, "run", runner)
    monkeypatch.setattr(sys, "argv", ["drift.py", "--since", "2026-01-01T00:00:00Z"])
    assert drift.main() == 0
    assert runner.call_args.args[0].min_requests == 300
