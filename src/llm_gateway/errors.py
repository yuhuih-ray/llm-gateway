from fastapi import Request
from fastapi.responses import JSONResponse


class GatewayError(Exception):
    def __init__(
        self,
        status: int,  # HTTP response status.
        message: str,  # Sanitized public message.
        error_type: str,  # OpenAI error category.
        code: str | None = None,  # Optional machine-readable code.
        headers: dict[str, str] | None = None,  # Retry hints, never credentials.
    ) -> None:
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = headers


async def gateway_error_handler(
    request: Request,  # FastAPI handler interface.
    exc: Exception,  # Registered only for GatewayError.
) -> JSONResponse:
    assert isinstance(exc, GatewayError)
    error = {"message": exc.message, "type": exc.error_type}
    if exc.code is not None:
        error["code"] = exc.code
    return JSONResponse(
        status_code=exc.status, content={"error": error}, headers=exc.headers
    )
