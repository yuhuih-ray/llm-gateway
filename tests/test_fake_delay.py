from unittest.mock import AsyncMock, call

import pytest
from pydantic import ValidationError

from llm_gateway.config import Settings
from llm_gateway.providers import FakeProvider
from llm_gateway.schemas import ChatCompletionRequest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("delay_ms", [0, 2000])
async def test_fake_delay_once_per_completion(monkeypatch, stream, delay_ms):
    monkeypatch.setattr(
        "llm_gateway.providers.get_settings",
        lambda: Settings(_env_file=None, fake_delay_ms=delay_ms),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("llm_gateway.providers.asyncio.sleep", sleep)
    provider = FakeProvider()
    request = ChatCompletionRequest(
        model="fake", messages=[{"role": "user", "content": "hi"}]
    )
    if stream:
        events = [event async for event in provider.stream(request)]
        assert len(events) == 4
    else:
        assert (await provider.complete(request)).choices[
            0
        ].message.content == "Hello from FakeProvider."
    expected = ([call(2.0)] if delay_ms else []) + ([call(0.001)] * 3 if stream else [])
    assert sleep.await_args_list == expected


def test_delay_setting_defaults_and_validation():
    assert Settings(_env_file=None).fake_delay_ms == 0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fake_delay_ms=-1)
