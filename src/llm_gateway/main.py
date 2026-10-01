from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI

from llm_gateway.auth import (
    AuthContext,
    AuthenticationError,
    authenticate,
    authentication_error_handler,
)
from llm_gateway.errors import GatewayError, gateway_error_handler
from llm_gateway.gemini import close_gemini_client
from llm_gateway.providers import Provider
from llm_gateway.registry import get_provider
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # Application lifecycle.
    try:
        yield
    finally:
        await close_gemini_client()


app = FastAPI(lifespan=lifespan)
app.add_exception_handler(AuthenticationError, authentication_error_handler)
app.add_exception_handler(GatewayError, gateway_error_handler)


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
