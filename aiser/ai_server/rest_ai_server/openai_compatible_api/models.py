from enum import Enum
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field
import time


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    stream: Optional[bool] = False


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionStreamResponseChoice]]
    usage: Optional[dict] = None
