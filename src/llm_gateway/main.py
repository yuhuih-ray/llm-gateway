from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from llm_gateway.admin import router as admin_router
from llm_gateway.auth import (
    AuthContext,
    AuthenticationError,
    authenticate,
    authentication_error_handler,
)
from llm_gateway.errors import GatewayError, gateway_error_handler
from llm_gateway.gemini import close_gemini_client
from llm_gateway.providers import Provider
from llm_gateway.registry import apply_token_limits, get_provider
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse
from llm_gateway.streaming import stream_response
from llm_gateway.usage import UsageMiddleware, UsageTracker, create_usage_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # Application lifecycle.
    app.state.usage_pool_last_attempt = monotonic()
    app.state.usage_pool = await create_usage_pool()
    try:
        yield
    finally:
        try:
            await close_gemini_client()
        finally:
            if app.state.usage_pool is not None:
                await app.state.usage_pool.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(admin_router)
app.add_middleware(UsageMiddleware)
app.add_exception_handler(AuthenticationError, authentication_error_handler)
app.add_exception_handler(GatewayError, gateway_error_handler)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    http_request: Request,  # Per-request accounting state, independent of DB sessions.
    auth: Annotated[
        AuthContext, Depends(authenticate)
    ],  # Authenticated tenant and key.
    request: ChatCompletionRequest,  # Validated chat input.
    provider: Annotated[
        Provider, Depends(get_provider)
    ],  # Injected completion provider.
) -> ChatCompletionResponse | StreamingResponse:
    request = apply_token_limits(request)
    tracker = UsageTracker(
        http_request.scope["state"],
        auth,
        request.model,
        http_request.app.state,
    )
    http_request.state.usage_tracker = tracker
    try:
        if request.stream:
            return await stream_response(provider, request, tracker)
        response = await provider.complete(request)
        tracker.success(response.usage)
        return response
    except Exception as exc:
        tracker.failure(exc)
        raise
