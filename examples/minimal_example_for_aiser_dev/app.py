from typing import List
import typing
import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from aiser.agent.agent import TokenUsage
from aiser import RestAiServer, KnowledgeBase, SemanticSearchResult, Agent
from aiser.models import ChatMessage
import asyncio
from dotenv import load_dotenv

load_dotenv()


class KnowledgeBaseExample(KnowledgeBase):
    def perform_semantic_search(self, query_text: str, desired_number_of_results: int) -> List[SemanticSearchResult]:
        result_example = SemanticSearchResult(
            content="This is an example of a semantic search result",
            score=0.5,
        )
        return [result_example for _ in range(desired_number_of_results)]


class AgentExample(Agent):
    def get_latest_reply_token_usage(self) -> typing.Optional[TokenUsage]:
        return TokenUsage(prompt_tokens=10, completion_tokens=20)
    
    async def reply(self, messages: typing.List[ChatMessage]) -> typing.AsyncGenerator[ChatMessage, None]:
        reply_message = "This is an example of a reply from an agent"
        for character in reply_message:
            yield ChatMessage(text_content=character)
            await asyncio.sleep(0.1)


if __name__ == '__main__':
    server = RestAiServer(
        agents=[
            AgentExample(
                agent_id='anthropic/claude-3.5-sonnet:beta'  # replace with your agent id

            ),
        ],
        knowledge_bases=[
            KnowledgeBaseExample(
                knowledge_base_id='85bc1c72-b8e0-4042-abcf-8eb2d478f207'  # replace with your knowledge base id
            ),
        ],
        port=int(os.getenv('PORT', 5000))
    )
    server.run()
