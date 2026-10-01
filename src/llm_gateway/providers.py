import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Protocol

from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Usage,
)


@dataclass(frozen=True)
class TextEvent:
    text: str


@dataclass(frozen=True)
class FinalEvent:
    finish_reason: Literal["stop", "length", "content_filter"]
    usage: Usage | None = None


StreamEvent = TextEvent | FinalEvent


class Provider(Protocol):
    def stream(
        self,
        request: ChatCompletionRequest,  # Validated input with final token limit.
    ) -> AsyncGenerator[StreamEvent, None]: ...

    async def complete(
        self,
        request: ChatCompletionRequest,  # Validated chat input.
    ) -> ChatCompletionResponse: ...


class FakeProvider:
    def __init__(
        self,
        fail_after: int | None = None,  # Number of text events before a test failure.
        delay: float = 0.001,  # Simulate incremental upstream delivery.
    ) -> None:
        self.fail_after = fail_after
        self.delay = delay

    async def stream(
        self,
        request: ChatCompletionRequest,  # Same deterministic response as complete().
    ) -> AsyncGenerator[StreamEvent, None]:
        chunks = ["Hello ", "from ", "FakeProvider."]
        for index, chunk in enumerate(chunks):
            if self.fail_after == index:
                raise ConnectionError("Fake provider failure")
            await asyncio.sleep(self.delay)
            yield TextEvent(chunk)
        if self.fail_after == len(chunks):
            raise ConnectionError("Fake provider failure")
        yield FinalEvent("stop", Usage())

    async def complete(
        self,
        request: ChatCompletionRequest,  # Validated chat input.
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-fake",
            model=request.model,
            choices=[
                Choice(message=AssistantMessage(content="Hello from FakeProvider."))
            ],
        )
