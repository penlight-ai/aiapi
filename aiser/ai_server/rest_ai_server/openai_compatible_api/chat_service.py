import json
import time
from typing import List, AsyncGenerator

from aiser.agent import ServerAgent, TokenUsage
from aiser.models import ChatMessage
from .models import (
    ChatMessage as OpenAIChatMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponseChoice,
    ContentItem,
    DeltaMessage,
    UsageInfo,
)


class ChatService:
    def __init__(self, agent: ServerAgent):
        self._agent = agent

    def _convert_to_internal_messages(self, messages: List[OpenAIChatMessage]) -> List[ChatMessage]:
        """Convert OpenAI messages to internal ChatMessage format."""
        # todo handle case where content is a list of ContentItem
        # return [ChatMessage(text_content=msg.content) for msg in messages]
        msgs = []
        for msg in messages:
            new_msg = ChatMessage(text_content='')
            if isinstance(msg.content, list):
                new_msg.text_content = '\n\n'.join([item.text for item in msg.content])
            elif isinstance(msg.content, str):
                new_msg.text_content = msg.content
            msgs.append(new_msg)
        return msgs


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

        # Final chunk with finish reason
        yield self._create_stream_chunk(
            model=model,
            delta=DeltaMessage(),
            finish_reason="stop",
        )
        
        # Get token usage from the latest reply if available
        token_usage = self._agent.get_latest_reply_token_usage()
        usage = UsageInfo(
            prompt_tokens=token_usage.prompt_tokens if token_usage else 0,
            completion_tokens=token_usage.completion_tokens if token_usage else 0,
            total_tokens=token_usage.total_tokens if token_usage else 0
        )
        
        # Usage info chunk - including a choice with empty content
        response = ChatCompletionResponse(
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None
                )
            ],
            usage=usage
        )
        yield f"data: {json.dumps(response.model_dump(exclude_none=True))}\n\n"

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
