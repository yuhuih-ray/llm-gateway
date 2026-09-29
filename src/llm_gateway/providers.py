from typing import Protocol

from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
)


class Provider(Protocol):
    async def complete(
        self,
        request: ChatCompletionRequest,  # Validated chat input.
    ) -> ChatCompletionResponse: ...


class FakeProvider:
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
