from typing import List
import typing
from aiser import RestAiServer, KnowledgeBase, SemanticSearchResult, ServerAgent
from aiser.models import ChatMessage
import asyncio


class KnowledgeBaseExample(KnowledgeBase):
    def perform_semantic_search(self, query_text: str, desired_number_of_results: int) -> List[SemanticSearchResult]:
        result_example = SemanticSearchResult(
            content="This is an example of a semantic search result",
            score=0.5,
        )
        return [result_example for _ in range(desired_number_of_results)]


class AgentExample(ServerAgent):
    async def reply(self, messages: typing.List[ChatMessage]) -> typing.AsyncGenerator[ChatMessage, None]:
        reply_message = "This is an example of a reply from an agent"
        for character in reply_message:
            yield ChatMessage(text_content=character)
            await asyncio.sleep(0.1)


if __name__ == '__main__':
    server = RestAiServer(
        agents=[
            AgentExample(
                agent_id='10209b93-2dd0-47a0-8eb2-33fb018a783b'  # replace with your agent id
            ),
        ],
        knowledge_bases=[
            KnowledgeBaseExample(
                knowledge_base_id='85bc1c72-b8e0-4042-abcf-8eb2d478f207'  # replace with your knowledge base id
            ),
        ],
        port=5000
    )
    server.run()
