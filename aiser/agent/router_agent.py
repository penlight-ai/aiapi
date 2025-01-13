from aiser.agent import Agent, TokenUsage
from aiser.models import ChatMessage
import typing
import re


class RouterAgent(Agent):
    token_usage_for_last_reply: typing.Optional[TokenUsage] = None

    def __init__(self, agents: typing.List[Agent]):
        super().__init__(agent_id="router_agent")
        self.agents = agents
        self.current_agent_index = 0

    def get_latest_reply_token_usage(self) -> typing.Optional[TokenUsage]:
        agent = self.agents[self.current_agent_index]
        return agent.get_latest_reply_token_usage()

    async def reply(
        self, messages: typing.List[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        self.infer_agent_idx_from_latest_message_if_appropriate(
            message_content=messages[-1].text_content
        )
        agent = self.agents[self.current_agent_index]
        yield ChatMessage(
            text_content=f'#### Agent "{agent.get_id()}":\n',
        )
        async for chunk in agent.reply(messages):
            yield chunk

    def infer_agent_idx_from_latest_message_if_appropriate(self, message_content: str):
        match = re.search(r"@agent\s+([\w-]+)", message_content)
        if match:
            agent_name = match.group(1)
            for idx, agent in enumerate(self.agents):
                if agent.get_id() == agent_name:
                    self.current_agent_index = idx
                    break
