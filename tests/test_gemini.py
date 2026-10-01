from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import errors, types

from llm_gateway.errors import GatewayError
from llm_gateway.gemini import GeminiProvider, map_finish_reason
from llm_gateway.retries import with_retries
from llm_gateway.schemas import ChatCompletionRequest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_translation():
    generate = AsyncMock(
        return_value=types.GenerateContentResponse(
            response_id="answer",
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model", parts=[types.Part(text="Hello")]
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=5,
                candidates_token_count=2,
                total_token_count=10,
                thoughts_token_count=3,
            ),
        )
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    request = ChatCompletionRequest(
        model="gemini-flash",
        messages=[
            {"role": "system", "content": "First"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey"},
            {"role": "system", "content": "Second"},
        ],
        max_tokens=12,
        temperature=0.3,
    )
    response = await GeminiProvider(
        "verified-id", client=client, thinking_level=types.ThinkingLevel.LOW
    ).complete(request)
    args = generate.call_args.kwargs
    assert args["model"] == "verified-id"
    assert args["config"].system_instruction == "First\nSecond"
    assert args["config"].max_output_tokens == 12
    assert args["config"].temperature == 0.3
    assert args["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert [content.role for content in args["contents"]] == ["user", "model"]
    assert [content.parts[0].text for content in args["contents"]] == ["Hi", "Hey"]
    assert response.model == "gemini-flash"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Hello"
    assert response.usage.model_dump() == {
        "prompt_tokens": 5,
        "completion_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": 10,
    }


@pytest.mark.anyio
async def test_retry_then_success():
    operation = AsyncMock(
        side_effect=[errors.ServerError(503, {"message": "private"}), "ok"]
    )
    with (
        patch("llm_gateway.retries.asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch("llm_gateway.retries.random.uniform", return_value=0.25) as uniform,
    ):
        assert await with_retries(operation, 30) == "ok"
    assert operation.await_count == 2
    uniform.assert_called_once_with(0, 0.5)
    sleep.assert_awaited_once_with(0.25)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status,expected", [(400, 400), (401, 502), (403, 502), (404, 502)]
)
async def test_no_retry_on_client_error(
    status, expected, caplog
):  # Upstream status and public mapping.
    operation = AsyncMock(
        side_effect=errors.ClientError(status, {"message": "private"})
    )
    with pytest.raises(GatewayError) as error:
        await with_retries(operation, 30)
    assert error.value.status == expected
    assert "private" not in error.value.message
    assert operation.await_count == 1
    if status in (401, 403):
        assert "provider credentials are invalid" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure,expected",
    [
        (errors.ServerError(503, {}), 502),
        (TimeoutError(), 504),
        (httpx.ConnectError("private"), 502),
        (errors.ClientError(429, {}), 502),
    ],
)
async def test_exhaustion(failure, expected):  # Retryable failure and terminal status.
    operation = AsyncMock(side_effect=failure)
    with (
        patch("llm_gateway.retries.asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch(
            "llm_gateway.retries.random.uniform", side_effect=lambda low, high: high
        ) as uniform,
    ):
        with pytest.raises(GatewayError) as error:
            await with_retries(operation, 30)
    assert error.value.status == expected
    assert operation.await_count == 3
    assert [call.args for call in uniform.call_args_list] == [(0, 0.5), (0, 1.0)]
    assert [call.args for call in sleep.await_args_list] == [(0.5,), (1.0,)]


@pytest.mark.anyio
async def test_per_attempt_timeout():
    import asyncio

    async def operation():
        await asyncio.sleep(0.1)

    with patch("llm_gateway.retries.random.uniform", return_value=0):
        with pytest.raises(GatewayError) as error:
            await with_retries(operation, 0.001)
    assert error.value.status == 504


@pytest.mark.parametrize(
    "reason,expected",
    [
        (types.FinishReason.STOP, "stop"),
        (types.FinishReason.MAX_TOKENS, "length"),
        (types.FinishReason.SAFETY, "content_filter"),
        (types.FinishReason.RECITATION, "content_filter"),
        (types.FinishReason.BLOCKLIST, "content_filter"),
        (types.FinishReason.PROHIBITED_CONTENT, "content_filter"),
        (types.FinishReason.SPII, "content_filter"),
    ],
)
def test_finish_reason(reason, expected):  # SDK reason and gateway mapping.
    assert map_finish_reason(reason) == expected


def test_unknown_finish_reason():
    with pytest.raises(GatewayError):
        map_finish_reason(types.FinishReason.OTHER)
