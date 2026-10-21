import pytest

from llm_gateway.config import get_settings
from llm_gateway.providers import FinalEvent, TextEvent
from llm_gateway.registry import apply_token_limits
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.vllm import VllmProvider, close_vllm_client

pytestmark = [
    pytest.mark.live_vllm,
    pytest.mark.anyio,
    pytest.mark.skipif(
        not get_settings().vllm_base_url, reason="VLLM_BASE_URL is not set"
    ),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("stream", [False, True])
async def test_live_vllm(stream):
    request = apply_token_limits(
        ChatCompletionRequest(
            model="local-qwen",
            max_tokens=32,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 + 2? Reply with only the number.",
                }
            ],
        )
    )
    provider = VllmProvider("qwen2.5-7b-awq")
    try:
        if stream:
            events = [event async for event in provider.stream(request)]
            assert any(isinstance(event, TextEvent) and event.text for event in events)
            assert isinstance(events[-1], FinalEvent)
            assert events[-1].finish_reason in ("stop", "length")
            assert events[-1].usage.total_tokens > 0
        else:
            response = await provider.complete(request)
            assert response.model == "local-qwen"
            assert response.choices[0].message.content
            assert response.choices[0].finish_reason in ("stop", "length")
            assert response.usage.total_tokens > 0
    finally:
        await close_vllm_client()
