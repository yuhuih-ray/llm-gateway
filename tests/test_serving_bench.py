import asyncio
import importlib.util
import math
from pathlib import Path

import httpx
import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts/serving_bench" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_script = script("gsm8k_eval")
compare = script("compare")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Work: 20. #### $1,234.00.", "1234.00"),
        ("#### 1\nRevised #### -2.5", "-2.5"),
        ("First 10 then 25 dollars.", "25"),
        ("Cost is $1,234.50.", "1234.50"),
        ("Result: -.5", "-.5"),
        ("No answer", None),
        ("#### not numeric", None),
        ("#### NaN", None),
        ("#### Infinity", None),
        ("####", None),
        ("#### 18 dollars", "18"),
        ("**#### 18**", "18"),
        ("#### $1,234.50 per day", "1234.50"),
        ("#### none", None),
        ("#### 18 dollars for 2 people", "18"),
    ],
)
def test_extract(text, expected):
    assert eval_script.extract_answer(text) == expected


def test_gold_requires_marker_and_decimal_comparison():
    assert eval_script.extract_answer("42", gold=True) is None
    assert eval_script.extract_answer("#### 18 dollars", gold=True) is None
    gold = eval_script.extract_answer("Work 99 #### 1,000", gold=True)
    pred = eval_script.extract_answer("#### $1000.00.")
    assert eval_script.correct_answer(gold, pred)
    assert not eval_script.correct_answer(gold, "1001")
    assert not eval_script.correct_answer(None, None)
    assert not eval_script.correct_answer(gold, None)


def test_paired_statistics():
    # Shared: four A-only wins, one B-only win, two both right, three both wrong.
    a = dict(enumerate([True] * 4 + [False] + [True] * 2 + [False] * 3))
    b = dict(enumerate([False] * 4 + [True] * 3 + [False] * 3))
    a[100] = True  # Unshared ids must not enter any denominator.
    b[200] = False
    result = compare.statistics(a, b)
    assert result["n"] == 10
    assert result["accuracy_a"] == 0.6
    assert result["accuracy_b"] == 0.3
    assert (result["n_ab"], result["n_ba"]) == (4, 1)
    assert result["difference"] == 0.3
    se = math.sqrt(4.1) / 10
    assert result["ci_low"] == pytest.approx(0.3 - 1.96 * se)
    assert result["ci_high"] == pytest.approx(0.3 + 1.96 * se)
    assert result["p"] == 0.375  # 2 * (C(5,0) + C(5,1)) / 2**5.


def test_agreement_and_no_shared_ids():
    result = compare.statistics({0: True, 1: False}, {0: True, 1: False})
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0
    assert result["p"] == 1
    with pytest.raises(ValueError, match="no shared"):
        compare.statistics({0: True}, {1: True})


def test_large_exact_binomial_does_not_overflow():
    a = dict.fromkeys(range(1319), True)
    b = dict.fromkeys(a, False)
    assert 0 <= compare.statistics(a, b)["p"] < 0.001
    a = {i: i < 659 for i in a}
    b = {i: not value for i, value in a.items()}
    assert compare.statistics(a, b)["p"] == 1


def test_load_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"id": 0, "correct": true}\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        compare.load(path)


@pytest.mark.parametrize(
    "failure,calls", [(503, 3), (429, 1), (400, 1), ("connect", 3), ("read", 1)]
)
def test_retry_policy(monkeypatch, failure, calls):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(eval_script.asyncio, "sleep", AsyncMock())
    attempts = 0

    def handle(request):
        nonlocal attempts
        attempts += 1
        if failure == "connect":
            raise httpx.ConnectError("refused")
        if failure == "read":
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(failure)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            with pytest.raises(httpx.HTTPError):
                await eval_script.completion(client, {})

    asyncio.run(run())
    assert attempts == calls


def test_evaluation_offline_cache_payload_and_outputs(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace

    downloads, active, peak = 0, 0, 0
    data = "".join(
        json.dumps({"question": str(i), "answer": f"#### {i}"}) + "\n" for i in range(3)
    )

    async def handle(request):
        nonlocal downloads, active, peak
        if request.method == "GET":
            downloads += 1
            assert "authorization" not in request.headers
            return httpx.Response(200, text=data)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == 1024
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        question, prompt = payload["messages"][0]["content"].split("\n")
        assert (
            prompt
            == 'Please reason step by step, and put your final answer after "####".'
        )
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": f"#### {question}"},
                        "finish_reason": "length" if question == "2" else "stop",
                    }
                ],
                "usage": {"completion_tokens": 10},
            },
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(eval_script, "DATA_PATH", tmp_path / "data/test.jsonl")
    monkeypatch.setattr(
        eval_script.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(handle), **kwargs
        ),
    )
    args = Namespace(
        base_url="http://test/v1",
        api_key="test-key",
        model="test-model",
        label="sample",
        concurrency=2,
        limit=None,
        out_dir=tmp_path / "results",
    )
    asyncio.run(eval_script.evaluate(args))
    asyncio.run(eval_script.evaluate(args))
    rows = [
        json.loads(line)
        for line in (args.out_dir / "sample.jsonl").read_text().splitlines()
    ]
    assert downloads == 1
    assert peak == 2
    assert [row["id"] for row in rows] == [0, 1, 2]
    assert all(row["correct"] for row in rows)
    assert all(
        set(row)
        == {
            "id",
            "gold",
            "pred",
            "correct",
            "finish_reason",
            "completion_tokens",
            "error",
        }
        for row in rows
    )
    output = capsys.readouterr().out
    assert "Accuracy: 100.00%" in output
    assert "finish_reason=length: 1" in output
    assert "Mean completion_tokens: 10.00" in output


def test_evaluation_error_and_unparsed_counts(monkeypatch, tmp_path, capsys):
    import json
    from argparse import Namespace
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        eval_script,
        "dataset",
        AsyncMock(
            return_value=[{"question": str(i), "answer": "#### 18"} for i in range(3)]
        ),
    )
    monkeypatch.setattr(
        eval_script,
        "completion",
        AsyncMock(
            side_effect=[
                httpx.ReadTimeout("private upstream details"),
                {
                    "choices": [
                        {"message": {"content": "#### none"}, "finish_reason": "stop"}
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": "**#### 18**"}, "finish_reason": "stop"}
                    ]
                },
            ]
        ),
    )
    args = Namespace(
        base_url="http://test/v1",
        api_key=None,
        model="test",
        label="errors",
        concurrency=1,
        limit=None,
        out_dir=tmp_path,
    )
    asyncio.run(eval_script.evaluate(args))
    rows = [
        json.loads(line)
        for line in (tmp_path / "errors.jsonl").read_text().splitlines()
    ]
    assert [row["error"] for row in rows] == ["ReadTimeout", None, None]
    assert [row["pred"] for row in rows] == [None, None, "18"]
    assert [row["correct"] for row in rows] == [False, False, True]
    captured = capsys.readouterr()
    assert "Failed requests: 1" in captured.out
    assert "Successful requests with no parsed answer: 1" in captured.out
    assert "Accuracy: 33.33%" in captured.out
    assert "ReadTimeout" in captured.err
    assert "private upstream details" not in captured.err
