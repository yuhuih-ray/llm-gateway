import asyncio

from starlette.datastructures import State

from llm_gateway.config import get_settings
from llm_gateway.deadline import check_deadline, limited_timeout
from llm_gateway.errors import GatewayError


async def acquire_provider(
    state: State,  # Lifespan state holds one semaphore per provider per process.
    provider: str,  # All registry models using this provider share its capacity.
    wait: bool = True,  # Auto routing escalates immediately when local capacity is full.
) -> asyncio.Semaphore | None:
    if provider != "vllm":
        return None
    # A rate limit cannot bound in-flight work when requests last seconds. vLLM
    # batches requests, but KV-cache memory caps concurrency. This is per process;
    # a global limit across gateway workers would need Redis.
    semaphore = state.provider_semaphores.setdefault(
        provider, asyncio.Semaphore(get_settings().vllm_max_concurrency)
    )
    if not wait:
        if semaphore.locked():
            raise GatewayError(
                503,
                "Upstream concurrency limit reached",
                "upstream_overloaded",
                headers={"Retry-After": "1"},
                fallback_reason="upstream_overloaded",
            )
        # In one event loop an unlocked acquire completes without yielding, so
        # no competing task can claim this slot between the check and acquisition.
        await semaphore.acquire()
        return semaphore
    try:
        async with asyncio.timeout(limited_timeout(2)):
            await semaphore.acquire()
    except TimeoutError:
        check_deadline()
        raise GatewayError(
            503,
            "Upstream concurrency limit reached",
            "upstream_overloaded",
            headers={"Retry-After": "1"},
            fallback_reason="upstream_overloaded",
        ) from None
    return semaphore
