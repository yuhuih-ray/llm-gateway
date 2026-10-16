from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.config import get_settings


class BodySizeMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = get_settings().max_request_bytes

        async def reject() -> None:
            await JSONResponse(
                {
                    "error": {
                        "message": "Request body too large",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
                status_code=413,
            )(scope, receive, send)

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    if int(value) > limit:
                        await reject()
                        return
                except ValueError:
                    pass  # Actual byte counting remains authoritative.
        # Buffer only up to the cap, before JSON parsing or endpoint dependencies.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                await reject()
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)
