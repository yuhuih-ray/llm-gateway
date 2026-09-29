from typing import Protocol

from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
)


class Provider(Protocol):
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...


class FakeProvider:
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="chatcmpl-fake",
            model=request.model,
            choices=[
                Choice(message=AssistantMessage(content="Hello from FakeProvider."))
            ],
        )
