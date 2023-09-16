import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from .ai_server import AiServer
from ..models.dtos import SemanticSearchRequest, AgentChatRequest, SemanticSearchResultDto, \
    SemanticSearchResultResponseDto, AgentChatResponse, ChatMessageDto
from ..models import ChatMessage
import typing


class SimpleAiServer(AiServer):
    def _get_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/version")
        async def version() -> str:
            return "0.1.0"

        @app.post("/knowledge-base/{kb_id}/semantic-search")
        async def knowledge_base(kb_id: str, request: SemanticSearchRequest) -> SemanticSearchResultResponseDto:
            for kb in self._knowledge_bases:
                if kb.id == kb_id:
                    results = kb.perform_semantic_search(
                        query_text=request.text,
                        desired_number_of_results=request.numResults
                    )
                    result_dto = SemanticSearchResultResponseDto(results=[
                        SemanticSearchResultDto(content=result.content, score=result.score)
                        for result in results
                    ])
                    return result_dto
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        async def convert_agent_message_gen_to_streaming_response(
                message_gen: typing.AsyncGenerator[ChatMessage, None]) -> typing.AsyncGenerator[str, None]:
            async for item in message_gen:
                message_dto = ChatMessageDto(textContent=item.text_content)
                yield AgentChatResponse(outputMessage=message_dto).model_dump_json(by_alias=True)

        @app.post("/agent/{agent_id}/chat")
        async def agent_chat(agent_id: str, request: AgentChatRequest) -> StreamingResponse:
            for agent in self._agents:
                if agent.id == agent_id:
                    response_generator = agent.reply(
                        input_message=ChatMessage(text_content=request.inputMessage.textContent)
                    )
                    response_generator = convert_agent_message_gen_to_streaming_response(message_gen=response_generator)
                    return StreamingResponse(
                        response_generator,
                        media_type="text/event-stream"
                    )
            raise HTTPException(status_code=404, detail="Agent not found")

        return app

    def run(self):
        uvicorn.run(app=self._get_app(), port=self._port)
