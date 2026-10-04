from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    stream: bool = False
    stream_options: StreamOptions | None = None
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = 0


class Usage(BaseModel):
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    completion_tokens_details: CompletionTokensDetails = Field(
        default_factory=CompletionTokensDetails
    )


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    id: str
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
