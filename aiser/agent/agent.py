from abc import ABC, abstractmethod
import typing
from dataclasses import dataclass

from ..identifiable_entities import IdentifiableEntity
from ..models import ChatMessage

@dataclass
class TokenUsage:
    """Token usage information for the latest reply call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ServerAgent(IdentifiableEntity, ABC):
    def __init__(self, agent_id: str):
        super().__init__(entity_id=agent_id)

    @abstractmethod
    def reply(self, messages: typing.List[ChatMessage]) -> typing.AsyncGenerator[ChatMessage, None]:
        raise NotImplementedError
    
    def get_latest_reply_token_usage(self) -> typing.Optional[TokenUsage]:
        """Optional method that agents can implement to provide token usage information
        for the most recent reply() call.
        
        This method should return the token usage that has accumulated since the start
        of the latest reply() method call.
        
        Returns:
            TokenUsage if the agent implements token counting, None otherwise.
        """
        return None
