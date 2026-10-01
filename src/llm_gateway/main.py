from typing import Annotated

from fastapi import Depends, FastAPI

from llm_gateway.auth import (
    AuthContext,
    AuthenticationError,
    authenticate,
    authentication_error_handler,
)
from llm_gateway.providers import FakeProvider, Provider
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse

app = FastAPI()
app.add_exception_handler(AuthenticationError, authentication_error_handler)


def get_provider() -> Provider:
    return FakeProvider()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    auth: Annotated[
        AuthContext, Depends(authenticate)
    ],  # Authenticated tenant and key.
    request: ChatCompletionRequest,  # Validated chat input.
    provider: Annotated[
        Provider, Depends(get_provider)
    ],  # Injected completion provider.
) -> ChatCompletionResponse:
    return await provider.complete(request)
