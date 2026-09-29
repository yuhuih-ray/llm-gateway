from typing import Annotated

from fastapi import Depends, FastAPI

from llm_gateway.providers import FakeProvider, Provider
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

app = FastAPI()


def get_provider() -> Provider:
    return FakeProvider()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completion(
    request: ChatCompletionRequest,
    provider: Annotated[Provider, Depends(get_provider)],
) -> ChatCompletionResponse:
    return provider.complete(request)
