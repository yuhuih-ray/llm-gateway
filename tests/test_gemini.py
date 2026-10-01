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
        (httpx.ConnectTimeout("private"), 504),
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


@pytest.mark.anyio
async def test_shared_client_and_lifespan_shutdown():
    from unittest.mock import Mock

    from pydantic import SecretStr

    from llm_gateway import gemini
    from llm_gateway.config import Settings
    from llm_gateway.main import app

    generate = AsyncMock(
        return_value=types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(parts=[types.Part(text="Hello")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        )
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate), aclose=AsyncMock()
    )
    factory = Mock(return_value=SimpleNamespace(aio=client))
    request = ChatCompletionRequest(
        model="gemini-flash", messages=[{"role": "user", "content": "Hi"}]
    )
    with (
        patch.object(gemini, "_shared_client", None),
        patch.object(gemini.genai, "Client", factory),
        patch.object(
            gemini,
            "get_settings",
            return_value=Settings(gemini_api_key=SecretStr("test")),
        ),
    ):
        async with app.router.lifespan_context(app):
            factory.assert_not_called()
            await GeminiProvider("model-a").complete(request)
            first = gemini.get_gemini_client()
            await GeminiProvider("model-b").complete(request)
            assert first is gemini.get_gemini_client() is client
            assert generate.await_count == 2
            factory.assert_called_once()
            assert factory.call_args.kwargs["http_options"].retry_options.attempts == 1
            client.aclose.assert_not_awaited()
        client.aclose.assert_awaited_once()
        assert gemini._shared_client is None


@pytest.mark.anyio
async def test_gemini_stream_thoughts_and_latest_usage():
    from llm_gateway.providers import FinalEvent, TextEvent
    from llm_gateway.schemas import CompletionTokensDetails, Usage

    closed = AsyncMock()

    async def upstream():
        try:
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            parts=[
                                types.Part(text="hidden", thought=True),
                                types.Part(text="Hello"),
                            ]
                        )
                    )
                ],
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=3, candidates_token_count=1, total_token_count=4
                ),
            )
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text=" world")]),
                        finish_reason=types.FinishReason.STOP,
                    )
                ]
            )
            yield types.GenerateContentResponse(
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=3,
                    candidates_token_count=2,
                    thoughts_token_count=4,
                    total_token_count=9,
                )
            )
        finally:
            await closed()

    generate = AsyncMock(return_value=upstream())
    provider = GeminiProvider(
        "test-model",
        client=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=generate)
        ),
    )
    result = [
        event
        async for event in provider.stream(
            ChatCompletionRequest(
                model="gemini-flash",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=99,
            )
        )
    ]
    assert result == [
        TextEvent("Hello"),
        TextEvent(" world"),
        FinalEvent(
            "stop",
            Usage(
                prompt_tokens=3,
                completion_tokens=6,
                total_tokens=9,
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=4),
            ),
        ),
    ]
    assert generate.call_args.kwargs["config"].max_output_tokens == 99
    closed.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [TimeoutError(), httpx.ReadTimeout("private")])
async def test_read_timeout_never_retries(failure):  # Deadline or SDK read timeout.
    operation = AsyncMock(side_effect=failure)
    with patch("llm_gateway.retries.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(GatewayError) as error:
            await with_retries(operation, 30)
    assert error.value.status == 504
    operation.assert_awaited_once()
    sleep.assert_not_awaited()


def test_measured_stream_reasoning_usage():
    from llm_gateway.gemini import map_usage

    usage = map_usage(
        types.GenerateContentResponseUsageMetadata(
            prompt_token_count=69,
            candidates_token_count=1318,
            thoughts_token_count=726,
            total_token_count=2113,
        )
    )
    assert usage.completion_tokens_details.reasoning_tokens == 726
    assert usage.completion_tokens == 2044
    assert usage.total_tokens == 2113
