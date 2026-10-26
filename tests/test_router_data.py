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
        ("ANSWER: **C**", "C"),
        ("ANSWER: (D).", "D"),
        ("ANSWER: Because", None),
        ("the answer is clearly wrong", None),
        ("the answer is a gas", None),
        ("the answer is (B)", "B"),
        ("AnSwEr : __ J __", "J"),
        ("ANSWER: AB", None),
        ("ANSWER: (BC)", None),
        ("answer is **E**", "E"),
        ("Reasoning\nANSWER: (j)", None),
        ("ANSWER: B\nCorrection: ANSWER: (C)", "C"),
        ("The answer is (d)", None),
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
        (503, 21, "HTTP503"),
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
    assert result["attempts"] == calls
    assert result["error"] == error
    assert not result["correct"]
    if failure in (429, 503):
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
        dataset=None,
        max_tokens=1024,
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


def test_regrade_matches_response_request_metadata(monkeypatch, tmp_path, capsys):
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
    label.regrade(args)
    assert path.read_text() == original
    assert (
        "skipped 1 saved responses without matching label metadata"
        in capsys.readouterr().err
    )


def test_final_attempt_latency_excludes_backoff_and_grading(monkeypatch):
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr(label, "time", SimpleNamespace(perf_counter=lambda: clock[0]))

    async def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(label.asyncio, "sleep", sleep)
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "20"}),
            httpx.Response(503),
            httpx.Response(
                200, headers={"X-Gateway-Model": "local-qwen"}, json=response_payload()
            ),
        ]
    )

    def handle(request):
        clock[0] += 0.025
        return next(responses)

    original_grade = label.grade

    def slow_grade(question, content):
        clock[0] += 5
        return original_grade(question, content)

    monkeypatch.setattr(label, "grade", slow_grade)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await label.label_question(client, "local-qwen", question())

    row = asyncio.run(run())
    assert row["error"] is None
    assert row["latency_ms"] == 25
    assert row["attempts"] == 3
    assert clock[0] > 25


def test_final_failed_attempt_timing(monkeypatch):
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr(label, "time", SimpleNamespace(perf_counter=lambda: clock[0]))

    async def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(label.asyncio, "sleep", sleep)

    def handle(request):
        clock[0] += 0.01
        raise httpx.ConnectError("refused")

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await label.label_question(client, "local-qwen", question())

    row = asyncio.run(run())
    assert row["latency_ms"] == 10
    assert row["attempts"] == 3
    assert row["error"] == "ConnectError"


def test_budget_exhaustion_stops_waiting_questions_but_finishes_inflight(
    monkeypatch, tmp_path, capsys
):
    rows = [dict(question(), id=f"gsm8k:{i}", prompt=str(i)) for i in range(5)]
    monkeypatch.setattr(label, "read_manifest", lambda: rows)
    started = []
    client_type = httpx.AsyncClient
    path = tmp_path / "local-qwen.jsonl"

    async def run():
        second_started, finish_second = asyncio.Event(), asyncio.Event()

        async def handle(request):
            index = int(json.loads(request.content)["messages"][0]["content"])
            started.append(index)
            if index == 0:
                await second_started.wait()
                return httpx.Response(
                    429,
                    headers={"Retry-After": "100", "X-Request-ID": "budget-request"},
                    json={"error": {"code": "insufficient_quota"}},
                )
            if index == 1:
                second_started.set()
                await finish_second.wait()
            return httpx.Response(
                200, headers={"X-Gateway-Model": "local-qwen"}, json=response_payload()
            )

        monkeypatch.setattr(
            label.httpx,
            "AsyncClient",
            lambda **kwargs: client_type(
                transport=httpx.MockTransport(handle), **kwargs
            ),
        )
        args = Namespace(
            model="local-qwen",
            per_dataset=None,
            dataset=None,
            max_tokens=1024,
            concurrency=2,
            api_key="test-key",
            base_url="http://test/v1",
            out_dir=tmp_path,
        )
        async with asyncio.timeout(2):
            task = asyncio.create_task(label.collect(args))
            try:
                while not path.exists() or not path.read_text():
                    await asyncio.sleep(0.001)
                finish_second.set()
                await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert started == [0, 1]
    result = common.read_labels(path)
    assert set(result) == {"gsm8k:0", "gsm8k:1"}
    assert result["gsm8k:0"]["error"] == "budget_exceeded"
    assert result["gsm8k:0"]["attempts"] == 1
    assert result["gsm8k:0"]["request_id"] == "budget-request"
    assert result["gsm8k:1"]["error"] is None
    output = capsys.readouterr().out
    assert "Not started: 3 (budget exceeded)" in output
    assert "Done: 1; skipped: 0; errors: 1" in output


def test_regrade_skips_orphans_and_still_grades_other_ids(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(label, "read_manifest", lambda: [question()])
    row = dict(
        id="gsm8k:0",
        model="local-qwen",
        request_id="kept",
        pred=None,
        correct=False,
        error=None,
        attempts=2,
        latency_ms=5,
        cost_usd="0",
        prompt_tokens=10,
        completion_tokens=3,
    )
    path = tmp_path / "local-qwen.jsonl"
    path.write_text(json.dumps(row) + "\n")
    label.RESPONSES.mkdir(parents=True)
    responses = [
        dict(id="gsm8k:99", request_id="orphan", content="#### 9"),
        dict(id="gsm8k:0", request_id="kept", content="#### 18"),
    ]
    (label.RESPONSES / "local-qwen.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses)
    )
    label.regrade(Namespace(model="local-qwen", out_dir=tmp_path))
    assert common.read_labels(path) == {
        "gsm8k:0": {**row, "pred": "18", "correct": True}
    }
    output = capsys.readouterr()
    assert "Regraded: 1; labels changed: 1" in output.out
    assert "Warning: skipped 1" in output.err


@pytest.mark.parametrize("count", [0, 25, 51])
def test_collection_progress_intervals_and_final(monkeypatch, tmp_path, capsys, count):
    from types import SimpleNamespace

    rows = [dict(question(), id=f"gsm8k:{i}") for i in range(count + 1)]
    monkeypatch.setattr(label, "read_manifest", lambda: rows)
    # This already successful label must not enter the pending total or accuracy.
    old = dict(
        id=f"gsm8k:{count}", model="local-qwen", error=None, correct=False, cost_usd="0"
    )
    (tmp_path / "local-qwen.jsonl").write_text(json.dumps(old) + "\n")
    clock = [0.0]
    monkeypatch.setattr(label, "time", SimpleNamespace(perf_counter=lambda: clock[0]))

    async def complete(client, model, question, *, max_tokens):
        clock[0] += 2
        index = int(question["id"].split(":")[1])
        return dict(
            id=question["id"],
            model=model,
            error="HTTP500" if index == 0 else None,
            correct=index != 0,
            cost_usd="0",
        )

    monkeypatch.setattr(label, "label_question", complete)
    args = Namespace(
        model="local-qwen",
        per_dataset=None,
        dataset=None,
        max_tokens=1024,
        concurrency=1,
        api_key="test-key",
        base_url="http://test/v1",
        out_dir=tmp_path,
    )
    asyncio.run(label.collect(args))
    output = capsys.readouterr()
    lines = output.err.splitlines()
    if count == 0:
        assert lines == ["[0/0] errors=0 correct=0.00% rate=0.00 eta=00:00"]
    elif count == 25:
        assert lines == ["[25/25] errors=1 correct=100.00% rate=0.50 eta=00:00"]
    else:
        assert lines == [
            "[25/51] errors=1 correct=100.00% rate=0.50 eta=00:52",
            "[50/51] errors=1 correct=100.00% rate=0.50 eta=00:02",
            "[51/51] errors=1 correct=100.00% rate=0.50 eta=00:00",
        ]
    assert "rate=" not in output.out


@pytest.mark.parametrize("first", [429, 503])
def test_429_and_503_share_retry_after_budget(monkeypatch, first):
    sleep = AsyncMock()
    monkeypatch.setattr(label.asyncio, "sleep", sleep)
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        status = first if calls % 2 else 503 if first == 429 else 429
        return httpx.Response(status, headers={"Retry-After": "2.5"})

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await label.label_question(client, "local-qwen", question())

    row = asyncio.run(run())
    assert calls == row["attempts"] == 21
    assert row["error"] == f"HTTP{first}"
    assert [call.args[0] for call in sleep.await_args_list] == [2.5] * 20


def test_503_without_retry_after_keeps_three_attempts(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(label.asyncio, "sleep", sleep)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/",
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
        ) as client:
            return await label.label_question(client, "local-qwen", question())

    row = asyncio.run(run())
    assert row["error"] == "HTTP503"
    assert row["attempts"] == 3
    assert [call.args[0] for call in sleep.await_args_list] == [0.5, 1.0]


@pytest.mark.parametrize("reset", [False, True])
def test_consecutive_errors_stop_and_success_resets(
    monkeypatch, tmp_path, capsys, reset
):
    questions = [dict(question(), id=f"gsm8k:{i}", prompt=str(i)) for i in range(25)]
    monkeypatch.setattr(label, "read_manifest", lambda: questions)
    calls = []

    def handle(request):
        index = int(json.loads(request.content)["messages"][0]["content"])
        calls.append(index)
        if reset and index == 9:
            # A successful question resets the streak even if its answer is wrong.
            return httpx.Response(
                200,
                headers={"X-Gateway-Model": "local-qwen"},
                json=response_payload("#### 0"),
            )
        return httpx.Response(400 if index % 2 == 0 else 401)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        label.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    args = Namespace(
        model="local-qwen",
        per_dataset=None,
        dataset=None,
        max_tokens=1024,
        concurrency=1,
        api_key="test-key",
        base_url="http://test/v1",
        out_dir=tmp_path,
    )
    asyncio.run(label.collect(args))
    count = 20 if reset else 10
    assert calls == list(range(count))
    assert len(common.read_labels(tmp_path / "local-qwen.jsonl")) == count
    output = capsys.readouterr().out
    assert output.count("Stopped: 10 consecutive errors") == 1
    assert f"Not started: {25 - count} (10 consecutive errors)" in output
    expected = {"HTTP400": 10, "HTTP401": 9} if reset else {"HTTP400": 5, "HTTP401": 5}
    assert f"Error counts: {json.dumps(expected, sort_keys=True)}" in output


def test_error_streak_stop_allows_inflight_question_to_finish(
    monkeypatch, tmp_path, capsys
):
    questions = [dict(question(), id=f"gsm8k:{i}", prompt=str(i)) for i in range(15)]
    monkeypatch.setattr(label, "read_manifest", lambda: questions)
    client = httpx.AsyncClient
    calls = []
    path = tmp_path / "local-qwen.jsonl"

    async def run():
        slow_started, finish_slow = asyncio.Event(), asyncio.Event()

        async def handle(request):
            index = int(json.loads(request.content)["messages"][0]["content"])
            calls.append(index)
            if index == 0:
                slow_started.set()
                await finish_slow.wait()
                return httpx.Response(
                    200,
                    headers={"X-Gateway-Model": "local-qwen"},
                    json=response_payload(),
                )
            await slow_started.wait()
            return httpx.Response(400)

        monkeypatch.setattr(
            label.httpx,
            "AsyncClient",
            lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
        )
        args = Namespace(
            model="local-qwen",
            per_dataset=None,
            dataset=None,
            max_tokens=1024,
            concurrency=2,
            api_key="test-key",
            base_url="http://test/v1",
            out_dir=tmp_path,
        )
        async with asyncio.timeout(2):
            task = asyncio.create_task(label.collect(args))
            try:
                while len(common.read_labels(path)) < 10:
                    await asyncio.sleep(0.001)
                finish_slow.set()
                await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert calls == list(range(11))
    result = common.read_labels(path)
    assert len(result) == 11
    assert result["gsm8k:0"]["error"] is None
    output = capsys.readouterr().out
    assert "Stopped: 10 consecutive errors" in output
    assert 'Error counts: {"HTTP400": 10}' in output
    assert "Not started: 4 (10 consecutive errors)" in output


@pytest.mark.parametrize("status", [200, 400])
def test_custom_max_tokens_sent_and_recorded(status):
    def handle(request):
        assert json.loads(request.content)["max_tokens"] == 2048
        return httpx.Response(
            status,
            headers={"X-Gateway-Model": "local-qwen"},
            json=response_payload(),
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await label.label_question(
                client, "local-qwen", question(), max_tokens=2048
            )

    row = asyncio.run(run())
    assert row["max_tokens"] == 2048
    assert row["error"] == (None if status == 200 else "HTTP400")


@pytest.mark.parametrize("datasets", [[], ["mmlupro"], ["gsm8k", "mmlupro"]])
def test_cli_filters_datasets_before_sampling_and_passes_max_tokens(
    monkeypatch, tmp_path, datasets
):
    rows = [
        dict(question(dataset), id=f"{dataset}:{i}", prompt=f"{dataset}:{i}")
        for dataset in ("gsm8k", "mmlupro", "other")
        for i in range(4)
    ]
    monkeypatch.setattr(label, "read_manifest", lambda: rows)
    original_select = label.select_questions
    selected_datasets = set(datasets or ["gsm8k", "mmlupro", "other"])

    def select(filtered, limit):
        assert {q["dataset"] for q in filtered} == selected_datasets
        assert limit == 2
        return original_select(filtered, limit)

    monkeypatch.setattr(label, "select_questions", select)
    calls = []
    max_tokens = 2048 if datasets else 1024

    def handle(request):
        payload = json.loads(request.content)
        assert payload["max_tokens"] == max_tokens
        calls.append(payload["messages"][0]["content"])
        return httpx.Response(
            200, headers={"X-Gateway-Model": "local-qwen"}, json=response_payload()
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        label.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    argv = [
        "label.py",
        "--model",
        "local-qwen",
        "--api-key",
        "test-key",
        "--base-url",
        "http://test/v1",
        "--out-dir",
        str(tmp_path),
        "--per-dataset",
        "2",
    ]
    for dataset in datasets:
        argv.extend(["--dataset", dataset])
    if datasets:
        argv.extend(["--max-tokens", str(max_tokens)])
    monkeypatch.setattr(label.sys, "argv", argv)
    label.main()
    expected = original_select(
        [q for q in rows if q["dataset"] in selected_datasets], 2
    )
    assert set(calls) == {q["id"] for q in expected}
    saved = common.read_labels(tmp_path / "local-qwen.jsonl")
    assert set(saved) == set(calls)
    assert all(row["max_tokens"] == max_tokens for row in saved.values())


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_max_tokens(monkeypatch, value):
    monkeypatch.setattr(
        label.sys, "argv", ["label.py", "--model", "local-qwen", "--max-tokens", value]
    )
    with pytest.raises(SystemExit) as exc:
        label.main()
    assert exc.value.code == 2
