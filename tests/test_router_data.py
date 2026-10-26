import asyncio
import hashlib
import importlib.util
import json
from argparse import Namespace
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts/router_data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"router_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = script("build_manifest")
label = script("label")
report = script("report")
common = script("common")


@pytest.fixture(autouse=True)
def isolated_responses(monkeypatch, tmp_path):
    monkeypatch.setattr(label, "RESPONSES", tmp_path / "responses")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ANSWER: A", "A"),
        ("Reasoning\nANSWER: (j)", "J"),
        ("ANSWER: B\nCorrection: ANSWER: (C)", "C"),
        ("The answer is (d)", "D"),
        ("answer is A, but the answer is F", "F"),
        ("ANSWER: B then answer is C", "B"),
        ("The choice is C", None),
        ("ANSWER: K", None),
        ("", None),
    ],
)
def test_mmlupro_extraction(text, expected):
    assert label.extract_mmlupro(text) == expected


def test_split_is_stable_and_holds_out_categories():
    ids = [f"mmlupro:{i}" for i in range(100)]
    before = {i: common.split_for(i, "math") for i in ids}
    after = {i: common.split_for(i, "math") for i in reversed(ids + ["mmlupro:100"])}
    assert all(before[i] == after[i] for i in ids)
    assert set(before.values()) == {"train", "test"}
    for i in ids:
        expected = (
            "test"
            if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 100 < 20
            else "train"
        )
        assert before[i] == expected
        assert common.split_for(i, "law") == common.split_for(i, "history") == "ood"


def test_manifest_stratification_prompts_and_order():
    gsm = [{"question": "What is 2 + 2?", "answer": "Calculation #### 4"}]
    mmlu = [
        dict(
            question_id=i,
            question=f"Question {i}",
            options=["one", "two"],
            answer="B",
            category="math" if i < 1500 else "law" if i < 2400 else "history",
        )
        for i in range(3000)
    ]
    result = builder.build(gsm, mmlu)
    assert builder.build(gsm, list(reversed(mmlu))) == result
    assert Counter(q["category"] for q in result) == {
        "gsm8k": 1,
        "math": 1000,
        "law": 600,
        "history": 400,
    }
    assert [q["id"] for q in result] == sorted(q["id"] for q in result)
    assert result[0]["gold"] == "4"
    assert result[0]["question"] == "What is 2 + 2?"
    assert (
        result[0]["prompt"]
        == 'What is 2 + 2?\nPlease reason step by step, and put your final answer after "####".'
    )
    sample = next(q for q in result if q["dataset"] == "mmlupro")
    assert (
        sample["prompt"]
        == f'Question {sample["id"].split(":")[1]}\nA. one\nB. two\nThink step by step, then end with a final line "ANSWER: <letter>".'
    )
    assert sample["question"] == f"Question {sample['id'].split(':')[1]}"
    assert "A. one" not in sample["question"]
    assert "Think step by step" not in sample["question"]
    assert "Please reason" not in result[0]["question"]
    for category in ("math", "law", "history"):
        expected = sorted(
            [f"mmlupro:{q['question_id']}" for q in mmlu if q["category"] == category],
            key=common.digest,
        )
        actual = {q["id"] for q in result if q["category"] == category}
        assert actual == set(expected[: len(actual)])


def test_committed_manifest_is_complete_and_consistent():
    rows = common.read_manifest()
    assert Counter(row["dataset"] for row in rows) == {"gsm8k": 1319, "mmlupro": 2000}
    assert [q["id"] for q in rows] == sorted(q["id"] for q in rows)
    assert all(q["split"] == common.split_for(q["id"], q["category"]) for q in rows)
    assert all(
        q["gold"] is not None and q["prompt"].startswith(q["question"] + "\n")
        for q in rows
    )


def test_resume_last_row_wins_and_errors_are_retried(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [
        dict(id="a", error="HTTP500"),
        dict(id="b", error=None),
        dict(id="a", error=None),
        dict(id="b", error="ReadTimeout"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    latest = common.read_labels(path)
    assert latest == {"a": rows[2], "b": rows[3]}
    assert common.pending_questions(
        [{"id": "a"}, {"id": "b"}, {"id": "c"}], latest
    ) == [{"id": "b"}, {"id": "c"}]
    assert common.read_labels(tmp_path / "absent") == {}


def test_pilot_selects_per_dataset_by_hash_before_resume():
    rows = [
        dict(id=f"{name}:{i}", dataset=name)
        for name in ("gsm8k", "mmlupro")
        for i in range(20)
    ]
    selected = common.select_questions(list(reversed(rows)), 3)
    assert Counter(row["dataset"] for row in selected) == {"gsm8k": 3, "mmlupro": 3}
    assert selected == common.select_questions(rows, 3)
    latest = {row["id"]: {"error": None} for row in selected}
    assert common.pending_questions(common.select_questions(rows, 3), latest) == []


def question(dataset="gsm8k"):
    return {
        "id": f"{dataset}:0",
        "dataset": dataset,
        "prompt": "exact user prompt",
        "gold": "18" if dataset == "gsm8k" else "B",
    }


def response_payload(content="#### 18 dollars"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


@pytest.mark.parametrize("actual", ["gemini-flash-lite", "local-qwen", None])
def test_model_identity_and_registry_pricing(monkeypatch, actual):
    model = "gemini-flash-lite"
    monkeypatch.setitem(
        label.MODELS,
        model,
        replace(
            label.MODELS[model], input_price=Decimal("2"), output_price=Decimal("3")
        ),
    )
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/chat/completions"
        assert "x-gateway-fallback" not in request.headers
        assert "x-gateway-cache" not in request.headers
        assert "cache-control" not in request.headers
        assert json.loads(request.content) == {
            "model": model,
            "messages": [{"role": "user", "content": "exact user prompt"}],
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
        }
        headers = {"X-Request-ID": "request-test"}
        if actual is not None:
            headers["X-Gateway-Model"] = actual
        return httpx.Response(200, headers=headers, json=response_payload())

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await label.label_question(client, model, question())

    result = asyncio.run(run())
    assert calls == 1
    assert result["model"] == model
    assert result["request_id"] == "request-test"
    assert result["latency_ms"] >= 0
    if actual != model:
        assert not (label.RESPONSES / f"{model}.jsonl").exists()
        assert result["error"] == "ModelMismatchError"
        assert not result["correct"]
        assert result["pred"] is result["cost_usd"] is None
    else:
        assert result["error"] is None and result["correct"]
        assert result["pred"] == "18"
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 20
        assert result["finish_reason"] == "stop"
        assert Decimal(result["cost_usd"]) == Decimal("0.00026000")


@pytest.mark.parametrize(
    "failure,calls,error",
    [
        (429, 21, "HTTP429"),
        (500, 3, "HTTP500"),
        (503, 3, "HTTP503"),
        (400, 1, "HTTP400"),
        ("connect", 3, "ConnectError"),
        ("connect_timeout", 3, "ConnectTimeout"),
        ("read", 1, "ReadTimeout"),
    ],
)
def test_retry_budgets_and_sanitized_errors(monkeypatch, failure, calls, error):
    sleep = AsyncMock()
    monkeypatch.setattr(label.asyncio, "sleep", sleep)
    attempts = 0

    def handle(request):
        nonlocal attempts
        attempts += 1
        if isinstance(failure, str):
            raise {
                "connect": httpx.ConnectError,
                "connect_timeout": httpx.ConnectTimeout,
                "read": httpx.ReadTimeout,
            }[failure]("private error")
        return httpx.Response(
            failure,
            headers={"Retry-After": "0.125", "X-Request-ID": "failed-request"},
            json={"error": "private error"},
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await label.label_question(client, "local-qwen", question())

    result = asyncio.run(run())
    assert attempts == calls
    assert result["error"] == error
    assert not result["correct"]
    if failure == 429:
        assert [call.args[0] for call in sleep.await_args_list] == [0.125] * 20
    elif calls == 3:
        assert [call.args[0] for call in sleep.await_args_list] == [0.5, 1.0]


def test_429_defaults_then_connect_retry_can_recover(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(label.asyncio, "sleep", sleep)
    responses = iter(
        [
            httpx.Response(429),
            httpx.ConnectError("refused"),
            httpx.Response(
                200,
                headers={"X-Gateway-Model": "local-qwen"},
                json=response_payload("ANSWER: (B)"),
            ),
        ]
    )

    def handle(request):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await label.label_question(client, "local-qwen", question("mmlupro"))

    row = asyncio.run(run())
    assert row["error"] is None and row["correct"]
    assert Decimal(row["cost_usd"]) == 0
    assert [call.args[0] for call in sleep.await_args_list] == [1, 0.5]


def test_collect_appends_resumes_and_prints_summary(monkeypatch, tmp_path, capsys):
    rows = [dict(question(), id=f"gsm8k:{i}") for i in range(3)]
    monkeypatch.setattr(label, "read_manifest", lambda: rows)
    old = dict(id="gsm8k:0", model="local-qwen", error=None, correct=True, cost_usd="0")
    error = dict(id="gsm8k:1", model="local-qwen", error="HTTP500")
    path = tmp_path / "local-qwen.jsonl"
    path.write_text(json.dumps(old) + "\n" + json.dumps(error) + "\n")
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"X-Gateway-Model": "local-qwen"}, json=response_payload()
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        label.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    args = Namespace(
        model="local-qwen",
        per_dataset=None,
        concurrency=2,
        api_key="test-key",
        base_url="http://test/v1",
        out_dir=tmp_path,
    )
    asyncio.run(label.collect(args))
    asyncio.run(label.collect(args))
    assert calls == 2
    assert len(path.read_text().splitlines()) == 4
    assert all(row["error"] is None for row in common.read_labels(path).values())
    out = capsys.readouterr().out
    assert "Done: 2; skipped: 1; errors: 0" in out
    assert "Done: 0; skipped: 3; errors: 0" in out
    assert "accuracy 100.00% (3 successful labels)" in out
    assert "Total cost_usd: 0" in out


def test_report_paired_filter_cost_and_latency():
    questions = [{"id": str(i)} for i in range(5)]
    a = {
        str(i): dict(
            correct=i in (0, 2), error=None, cost_usd="0.1", latency_ms=10 * (i + 1)
        )
        for i in range(5)
    }
    b = {
        str(i): dict(
            correct=i in (1, 2), error=None, cost_usd="0.2", latency_ms=20 * (i + 1)
        )
        for i in range(4)
    }
    b["3"]["error"] = "HTTP500"
    result = report.statistics(questions, a, b)
    assert result == dict(
        n=3,
        a_wrong_b_right=1,
        a_right_b_wrong=1,
        accuracy_a=2 / 3,
        accuracy_b=2 / 3,
        latency_a=20,
        latency_b=40,
        cost_a=Decimal("0.3"),
        cost_b=Decimal("0.6"),
    )
    assert "66.67%" in report.table("Dataset", {"gsm8k": questions}, a, b)
    a["0"]["cost_usd"] = None
    assert report.statistics(questions, a, b)["cost_a"] is None
    assert report.statistics([], a, b)["n"] == 0


def test_completion_saved_before_grading(monkeypatch):
    content = "Raw reasoning\n**#### 18**"

    def grade_after_save(question, text):
        saved = common.read_labels(label.RESPONSES / "local-qwen.jsonl")
        assert saved[question["id"]] == {
            "id": question["id"],
            "request_id": "saved-request",
            "content": content,
        }
        assert text == content
        raise RuntimeError("grading bug")

    monkeypatch.setattr(label, "grade", grade_after_save)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={
                        "X-Gateway-Model": "local-qwen",
                        "X-Request-ID": "saved-request",
                    },
                    json=response_payload(content),
                )
            ),
        ) as client:
            return await label.label_question(client, "local-qwen", question())

    row = asyncio.run(run())
    assert row["error"] == "RuntimeError"
    assert row["prompt_tokens"] == 100
    assert row["completion_tokens"] == 20
    assert row["request_id"] == "saved-request"


def test_regrade_cli_uses_latest_response_offline_and_keeps_metadata(
    monkeypatch, tmp_path, capsys
):
    from unittest.mock import Mock

    questions = [question(), question("mmlupro")]
    monkeypatch.setattr(label, "read_manifest", lambda: questions)
    metadata = dict(
        model="local-qwen",
        finish_reason="stop",
        prompt_tokens=123,
        completion_tokens=45,
        cost_usd="0.00012345",
        latency_ms=678.9,
        error=None,
    )
    rows = [
        dict(
            metadata,
            id=q["id"],
            request_id=f"request-{q['id']}",
            pred=None,
            correct=False,
        )
        for q in questions
    ]
    unparsed = dict(rows[0], id="gsm8k:99", error="HTTP500", request_id="no-response")
    path = tmp_path / "local-qwen.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [*rows, unparsed]))
    label.RESPONSES.mkdir(parents=True)
    responses = [
        dict(id=rows[0]["id"], request_id="older-request", content="#### 0"),
        dict(
            id=rows[0]["id"],
            request_id=rows[0]["request_id"],
            content="#### 18 dollars",
        ),
        dict(id=rows[1]["id"], request_id=rows[1]["request_id"], content="ANSWER: (B)"),
    ]
    (label.RESPONSES / "local-qwen.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses)
    )
    forbidden_client = Mock(
        side_effect=AssertionError("Regrade must not create an HTTP client")
    )
    monkeypatch.setattr(label.httpx, "AsyncClient", forbidden_client)
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.setattr(
        label.sys,
        "argv",
        [
            "label.py",
            "--model",
            "local-qwen",
            "--out-dir",
            str(tmp_path),
            "--regrade",
            "--per-dataset",
            "1",
        ],
    )
    label.main()
    result = common.read_labels(path)
    assert len(result) == 3
    for original, expected in zip(rows, ["18", "B"]):
        updated = result[original["id"]]
        assert updated == {**original, "pred": expected, "correct": True}
    assert result[unparsed["id"]] == unparsed
    assert "Regraded: 2; labels changed: 2" in capsys.readouterr().out
    label.main()
    assert "labels changed: 0" in capsys.readouterr().out
    forbidden_client.assert_not_called()


def test_regrade_matches_response_request_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(label, "read_manifest", lambda: [question()])
    row = dict(
        id="gsm8k:0",
        model="local-qwen",
        request_id="completed",
        pred=None,
        correct=False,
        error=None,
        prompt_tokens=100,
        completion_tokens=20,
        cost_usd="0.01",
        latency_ms=10,
        finish_reason="stop",
    )
    failed = {**row, "request_id": "later-failed", "error": "HTTP500", "cost_usd": None}
    path = tmp_path / "local-qwen.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(failed) + "\n")
    label.RESPONSES.mkdir(parents=True)
    raw = label.RESPONSES / "local-qwen.jsonl"
    raw.write_text(
        json.dumps(dict(id="gsm8k:0", request_id="completed", content="#### 18")) + "\n"
    )
    args = Namespace(model="local-qwen", out_dir=tmp_path)
    label.regrade(args)
    assert common.read_labels(path)["gsm8k:0"] == {**row, "pred": "18", "correct": True}
    original = path.read_text()
    raw.write_text(
        json.dumps(dict(id="gsm8k:0", request_id="missing-metadata", content="#### 18"))
        + "\n"
    )
    with pytest.raises(ValueError, match="No label metadata"):
        label.regrade(args)
    assert path.read_text() == original
