import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text

from llm_gateway.admin import router as admin_router
from llm_gateway.auth import (
    AuthContext,
    AuthenticationError,
    authenticate,
    authentication_error_handler,
)
from llm_gateway.budgets import budget_exceeded
from llm_gateway.cache import (
    CachedAnswer,
    cache_key,
    cacheable,
    create_cache_client,
    read_cache,
    write_cache,
)
from llm_gateway.db import get_sessionmaker
from llm_gateway.errors import GatewayError, gateway_error_handler
from llm_gateway.gemini import close_gemini_client
from llm_gateway.input_limits import BodySizeMiddleware
from llm_gateway.providers import Provider
from llm_gateway.ratelimit import rate_limiter, rejected, request_buckets
from llm_gateway.registry import MODELS, apply_token_limits, get_provider
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse
from llm_gateway.streaming import stream_response
from llm_gateway.usage import UsageMiddleware, UsageTracker, create_usage_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # Application lifecycle.
    app.state.usage_pool_last_attempt = monotonic()
    app.state.usage_pool = await create_usage_pool()
    app.state.cache_client = create_cache_client()
    try:
        yield
    finally:
        try:
            await close_gemini_client()
        finally:
            try:
                await app.state.cache_client.aclose()
            finally:
                if app.state.usage_pool is not None:
                    await app.state.usage_pool.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(admin_router)
app.add_middleware(BodySizeMiddleware)
app.add_middleware(UsageMiddleware)
app.add_exception_handler(AuthenticationError, authentication_error_handler)
app.add_exception_handler(GatewayError, gateway_error_handler)


logger = logging.getLogger(__name__)


@app.get("/health")
@app.get("/health/live")
async def health() -> dict[str, str]:
    # Liveness asks "is this process alive". Dependency checks here would cause
    # restart storms during a DB outage. A blocked event loop cannot answer,
    # which is exactly the failure that restarting the process can fix.
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:  # Lifespan-owned queue pool.
    # Postgres is required for authentication; the usage queue is best-effort
    # (failed enqueues log the record). Never probe Gemini/providers: failures
    # are external and per-model, and every health probe would spend quota.
    # Short deadlines prevent hung probes holding pool connections and piling up.
    postgres, usage_queue = "unavailable", "unavailable"
    try:
        async with asyncio.timeout(1.0):
            async with get_sessionmaker()() as session:
                await session.execute(text("SELECT 1"))
        postgres = "ok"
    except Exception:
        logger.warning("Readiness postgres unavailable", exc_info=True)

    pool = getattr(request.app.state, "usage_pool", None)
    try:
        async with asyncio.timeout(0.5):
            if pool is None:
                logger.warning("Readiness usage queue unavailable: pool missing")
            elif await pool.ping():
                usage_queue = "ok"
            else:
                logger.warning("Readiness usage queue unavailable: ping returned false")
    except Exception:
        logger.warning("Readiness usage queue unavailable", exc_info=True)
    return JSONResponse(
        status_code=200 if postgres == "ok" else 503,
        content={
            "status": "ready" if postgres == "ok" else "not_ready",
            "checks": {"postgres": postgres, "usage_queue": usage_queue},
        },
    )


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
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    request = apply_token_limits(request)
    decision = await rate_limiter.check(
        getattr(http_request.app.state, "usage_pool", None),
        request_buckets(auth.tenant_id, MODELS[request.model].provider),
    )
    if not decision.allowed:
        return rejected(decision)
    cache = getattr(http_request.app.state, "cache_client", None)
    key = (
        cache_key(auth.tenant_id, request)
        if cacheable(request, ",".join(http_request.headers.getlist("cache-control")))
        else None
    )
    answer = await read_cache(cache, key) if key else None
    http_request.state.cache_status = "HIT" if answer else "MISS"
    # Exhausted tenants can still receive free cached answers after request limits.
    if answer is None and await budget_exceeded(
        getattr(http_request.app.state, "usage_pool", None),
        auth.tenant_id,
        auth.monthly_budget_usd,
    ):
        raise GatewayError(
            429, "Monthly budget exceeded", "insufficient_quota", "insufficient_quota"
        )
    tracker = UsageTracker(
        http_request.scope["state"],
        auth,
        request.model,
        http_request.app.state,
        limiter=rate_limiter,
    )
    http_request.state.usage_tracker = tracker
    if answer is not None:
        tracker.record.cache_hit = True
        tracker.record.upstream_model = answer.upstream_model
        provider = answer

    async def store(response: ChatCompletionResponse) -> None:
        if key and answer is None:
            await write_cache(
                cache,
                key,
                CachedAnswer(
                    response=response,
                    upstream_model=MODELS[request.model].upstream_model,
                ),
            )

    try:
        if request.stream:
            return await stream_response(
                provider, request, tracker, store if key and answer is None else None
            )
        response = await provider.complete(request)
        tracker.success(response.usage)
        await store(response)
        return response
    except Exception as exc:
        tracker.failure(exc)
        raise
