import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from arq.constants import default_queue_name
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text

from llm_gateway.accounting import AccountingBuffer
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
from llm_gateway.circuit import CircuitBreaker
from llm_gateway.concurrency import acquire_provider
from llm_gateway.db import get_sessionmaker
from llm_gateway.deadline import check_deadline, commit, remaining
from llm_gateway.errors import GatewayError, gateway_error_handler
from llm_gateway.gemini import close_gemini_client
from llm_gateway.input_limits import BodySizeMiddleware
from llm_gateway.providers import Provider
from llm_gateway.ratelimit import (
    provider_buckets,
    rate_limiter,
    rejected,
    request_buckets,
)
from llm_gateway.registry import (
    MODELS,
    apply_token_limits,
    create_provider,
    get_provider,
)
from llm_gateway.schemas import ChatCompletionRequest, ChatCompletionResponse
from llm_gateway.streaming import stream_response
from llm_gateway.timing import ServerTimingMiddleware, measure
from llm_gateway.usage import UsageMiddleware, UsageTracker, create_usage_pool
from llm_gateway.vllm import close_vllm_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # Application lifecycle.
    app.state.provider_semaphores = {}
    app.state.circuits = {}
    app.state.queue_warning_at = float("-inf")
    app.state.usage_pool = await create_usage_pool()
    app.state.cache_client = create_cache_client()
    app.state.accounting = AccountingBuffer(app.state)
    app.state.accounting.start()
    try:
        yield
    finally:
        await app.state.accounting.close()
        try:
            await close_gemini_client()
        finally:
            try:
                try:
                    await close_vllm_client()
                finally:
                    await app.state.cache_client.aclose()
            finally:
                if app.state.usage_pool is not None:
                    await app.state.usage_pool.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(admin_router)
app.add_middleware(BodySizeMiddleware)
app.add_middleware(UsageMiddleware)
app.add_middleware(ServerTimingMiddleware)
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
    queue_metrics: dict[str, int | float | None] = {
        "depth": None,
        "oldest_job_age_seconds": None,
    }
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
                depth = await pool.zcard(default_queue_name)
                oldest = await pool.zrange(default_queue_name, 0, 0, withscores=True)
                # ARQ sorted-set scores are scheduled Unix milliseconds. Deferred
                # retries become old when due; never report a negative age.
                age = (
                    max(0.0, time.time() - float(oldest[0][1]) / 1000)
                    if oldest
                    else 0.0
                )
                queue_metrics = {"depth": depth, "oldest_job_age_seconds": age}
                usage_queue = "ok"
                now = time.monotonic()
                if age > 600 and now - request.app.state.queue_warning_at >= 60:
                    request.app.state.queue_warning_at = now
                    logger.warning(
                        "Usage queue backlog depth=%s oldest_job_age_seconds=%.1f",
                        depth,
                        age,
                    )
            else:
                logger.warning("Readiness usage queue unavailable: ping returned false")
    except Exception:
        logger.warning("Readiness usage queue unavailable", exc_info=True)
    return JSONResponse(
        status_code=200 if postgres == "ok" else 503,
        content={
            "status": "ready" if postgres == "ok" else "not_ready",
            "checks": {"postgres": postgres, "usage_queue": usage_queue},
            "usage_queue": queue_metrics,
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
    original = request
    # An explicitly named model must not silently change: callers may need local
    # data handling or fixed behavior. Phase 4 model="auto" will opt in by default.
    targets = [request.model]
    if http_request.headers.get("x-gateway-fallback", "").lower() == "on":
        targets.extend(
            name for name in MODELS[request.model].fallbacks if name != request.model
        )
    fallback_reason = None
    tracker = None
    cache = getattr(http_request.app.state, "cache_client", None)
    pool = getattr(http_request.app.state, "usage_pool", None)
    for index, model in enumerate(targets):
        check_deadline()
        request = apply_token_limits(original.model_copy(update={"model": model}))
        entry = MODELS[model]
        with measure(http_request, "ratelimit"):
            buckets = (
                request_buckets(auth.tenant_id, entry.provider)
                if index == 0
                else provider_buckets(entry.provider)
            )
            if buckets:
                decision = await rate_limiter.check(pool, buckets)
                if not decision.allowed:
                    return rejected(decision)
        with measure(http_request, "cache"):
            key = (
                cache_key(auth.tenant_id, request)
                if cacheable(
                    request,
                    ",".join(http_request.headers.getlist("cache-control")),
                    ",".join(http_request.headers.getlist("x-gateway-cache")),
                )
                else None
            )
            answer = await read_cache(cache, key) if key else None
        http_request.state.cache_status = "HIT" if answer else "MISS"
        # Exhausted tenants can still receive free cached answers after request limits.
        with measure(http_request, "budget"):
            if answer is None and await budget_exceeded(
                pool, auth.tenant_id, auth.monthly_budget_usd
            ):
                raise GatewayError(
                    429,
                    "Monthly budget exceeded",
                    "insufficient_quota",
                    "insufficient_quota",
                )
        breaker = http_request.app.state.circuits.setdefault(
            entry.provider, CircuitBreaker(entry.provider)
        )
        permit = None
        called = False
        committed = False
        try:
            if answer is None:
                permit = breaker.acquire()
                http_request.state.provider_slot = await acquire_provider(
                    http_request.app.state, entry.provider
                )
            if tracker is None:
                tracker = UsageTracker(
                    http_request.scope["state"],
                    auth,
                    original.model,
                    http_request.app.state,
                )
                http_request.state.usage_tracker = tracker
            tracker.record.model_selected = model
            tracker.record.upstream_model = (
                answer.upstream_model if answer else entry.upstream_model
            )
            tracker.record.cache_hit = answer is not None
            if index:
                tracker.record.router_info = {
                    "fallback_from": original.model,
                    "fallback_reason": fallback_reason or "upstream_error",
                }
            selected_provider = (
                answer
                if answer is not None
                else provider
                if index == 0
                else create_provider(model)
            )

            async def store(response: ChatCompletionResponse) -> None:
                # Each target has its own cache key: never reuse a cloud answer as local.
                if key and answer is None:
                    with measure(http_request, "cache"):
                        await write_cache(
                            cache,
                            key,
                            CachedAnswer(
                                response=response, upstream_model=entry.upstream_model
                            ),
                        )

            result: ChatCompletionResponse | StreamingResponse
            with measure(http_request, "provider"):
                called = answer is None
                if request.stream:
                    result = await stream_response(
                        selected_provider,
                        request,
                        tracker,
                        store if key and answer is None else None,
                    )
                else:
                    result = await selected_provider.complete(request)
            commit()
            committed = True
            if permit is not None:
                breaker.finish(permit, True)
                permit = None
            http_request.state.selected_model = model
            if index:
                http_request.state.fallback_from = original.model
            if isinstance(result, ChatCompletionResponse):
                result = result.model_copy(update={"model": model})
                tracker.success(result.usage)
                await store(result)
            return result
        except asyncio.CancelledError:
            if permit is not None:
                left = remaining()
                breaker.finish(
                    permit, False if called and left is not None and left <= 0 else None
                )
                permit = None
            raise
        except Exception as exc:
            if permit is not None:
                breaker.finish(
                    permit,
                    False
                    if called and isinstance(exc, GatewayError) and exc.breaker_failure
                    else None,
                )
                permit = None
            slot = http_request.scope["state"].pop("provider_slot", None)
            if slot is not None:
                slot.release()
            if tracker is not None:
                tracker.failure(exc)
            check_deadline()
            if (
                not committed
                and isinstance(exc, GatewayError)
                and exc.fallback_reason
                and index + 1 < len(targets)
            ):
                fallback_reason = exc.fallback_reason
                continue
            raise
        finally:
            if permit is not None:
                breaker.finish(permit, None)
    raise AssertionError("No routing target")
