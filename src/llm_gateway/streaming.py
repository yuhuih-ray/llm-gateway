import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from uuid import uuid4

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from llm_gateway.config import get_settings
from llm_gateway.errors import GatewayError
from llm_gateway.providers import FinalEvent, Provider, StreamEvent, TextEvent
from llm_gateway.retries import with_retries
from llm_gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChunkChoice,
    Delta,
)

logger = logging.getLogger(__name__)


class SSEStream:
    def __init__(
        self,
        upstream: AsyncGenerator[StreamEvent, None],  # Owned provider iterator.
        first: StreamEvent,  # Already fetched before HTTP status is committed.
        request: ChatCompletionRequest,  # Gateway model and usage preference.
        timeout: float,  # Maximum idle wait for the next provider event.
    ) -> None:
        self.upstream = upstream
        self.first = first
        self.request = request
        self.timeout = timeout
        self.sent = 0
        self.closed = False
        self.finished = False
        self.id = f"chatcmpl-{uuid4()}"
        self.created = int(time.time())

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            with anyio.CancelScope(shield=True):
                await self.upstream.aclose()

    def encode(
        self,
        choice: ChunkChoice | None,  # None denotes a usage-only chunk.
        final: FinalEvent | None = None,  # Optional final usage payload.
    ) -> str:
        chunk = ChatCompletionChunk(
            id=self.id,
            created=self.created,
            model=self.request.model,
            choices=[choice] if choice else [],
            usage=final.usage if final else None,
        )
        payload = chunk.model_dump()
        for item in payload["choices"]:
            item["delta"] = {
                key: value for key, value in item["delta"].items() if value is not None
            }
        if final is None:
            del payload["usage"]
        self.sent += 1
        return f"data: {json.dumps(payload)}\n\n"

    async def body(self) -> AsyncGenerator[str, None]:
        try:
            yield self.encode(ChunkChoice(delta=Delta(role="assistant")))
            event = self.first
            while True:
                if isinstance(event, TextEvent):
                    yield self.encode(ChunkChoice(delta=Delta(content=event.text)))
                else:
                    yield self.encode(ChunkChoice(finish_reason=event.finish_reason))
                    if (
                        self.request.stream_options
                        and self.request.stream_options.include_usage
                    ):
                        yield self.encode(None, event)
                    self.finished = True
                    yield "data: [DONE]\n\n"
                    return
                async with asyncio.timeout(self.timeout):
                    event = await anext(self.upstream)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            logger.warning(
                "Upstream stream failed chunks_sent=%s reason=%s",
                self.sent,
                type(exc).__name__,
            )
            self.finished = True
            message = (
                "Upstream stream timed out"
                if isinstance(exc, TimeoutError)
                else "Upstream stream failed"
            )
            yield f"data: {json.dumps({'error': {'message': message, 'type': 'upstream_error'}})}\n\n"
        finally:
            if not self.finished:
                logger.info("Client disconnected chunks_sent=%s", self.sent)
            await self.close()


class GatewayStreamingResponse(StreamingResponse):
    def __init__(
        self, stream: SSEStream
    ) -> None:  # Own cleanup even before body starts.
        self.stream = stream
        self.event_body = stream.body()
        super().__init__(
            self.event_body,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Cancellation can occur while sending headers, before body() is entered.
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.event_body.aclose()
                if not self.stream.closed:
                    logger.info("Client disconnected chunks_sent=%s", self.stream.sent)
                await self.stream.close()


async def stream_response(
    provider: Provider,  # No SSE knowledge is required of the provider.
    request: ChatCompletionRequest,  # Already authenticated and token-limited.
) -> StreamingResponse:
    timeout = get_settings().gemini_timeout_seconds

    async def open_stream() -> tuple[AsyncGenerator[StreamEvent, None], StreamEvent]:
        upstream = provider.stream(request)
        try:
            first = await anext(upstream)
            return upstream, first
        except BaseException:
            with anyio.CancelScope(shield=True):
                await upstream.aclose()
            raise

    try:
        upstream, first = await with_retries(open_stream, timeout)
    except GatewayError:
        raise
    except Exception:
        raise GatewayError(502, "Upstream stream failed", "upstream_error") from None
    return GatewayStreamingResponse(SSEStream(upstream, first, request, timeout))
