import json
import time
from typing import List, AsyncGenerator

from aiser.agent import Agent
from aiser.models import ChatMessage
from .models import (
    ChatMessage as OpenAIChatMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponseChoice,
    DeltaMessage,
)


class ChatService:
    def __init__(self, agent: Agent):
        self._agent = agent

    def _convert_to_internal_messages(self, messages: List[OpenAIChatMessage]) -> List[ChatMessage]:
        """Convert OpenAI messages to internal ChatMessage format."""
        return [ChatMessage(text_content=msg.content) for msg in messages]

    async def create_chat_completion(
        self, messages: List[OpenAIChatMessage], model: str
    ) -> ChatCompletionResponse:
        """Handle non-streaming chat completion."""
        internal_messages = self._convert_to_internal_messages(messages)
        response_gen = self._agent.reply(messages=internal_messages)
        
        # Get the full response by consuming the generator
        full_response = ""
        async for msg in response_gen:
            full_response += msg.text_content

        return ChatCompletionResponse(
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=OpenAIChatMessage(role="assistant", content=full_response),
                    finish_reason="stop",
                )
            ],
        )

    async def generate_stream(
        self, messages: List[OpenAIChatMessage], model: str
    ) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion."""
        internal_messages = self._convert_to_internal_messages(messages)
        response_gen = self._agent.reply(messages=internal_messages)

        # First chunk with role
        yield self._create_stream_chunk(
            model=model,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )

        async for msg in response_gen:
            yield self._create_stream_chunk(
                model=model,
                delta=DeltaMessage(content=msg.text_content),
                finish_reason=None,
            )

        # Final chunk
        yield self._create_stream_chunk(
            model=model,
            delta=DeltaMessage(),
            finish_reason="stop",
        )

    def _create_stream_chunk(
        self, model: str, delta: DeltaMessage, finish_reason: str | None
    ) -> str:
        """Create a single stream response chunk in OpenAI format."""
        response = ChatCompletionResponse(
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )
        return f"data: {json.dumps(response.model_dump(exclude_none=True))}\n\n"
