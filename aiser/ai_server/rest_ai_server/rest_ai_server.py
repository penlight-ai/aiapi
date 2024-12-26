import time
import typing

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, APIRouter, Request, status
from fastapi.responses import StreamingResponse
from aiser.ai_server.ai_server import AiServer
from aiser.ai_server.authentication import (
    AsymmetricJwtRestAuthenticator,
    NonFunctionalRestAuthenticator,
    RestAuthenticator
)
from aiser.config.ai_server_config import ServerEnvironment
from aiser.models.dtos import (
    SemanticSearchRequest,
    AgentChatRequest,
    SemanticSearchResultDto,
    SemanticSearchResultResponseDto,
    AgentChatResponse,
    ChatMessageDto,
    VersionInfo
)
from aiser.models import ChatMessage
from aiser.knowledge_base import KnowledgeBase
from aiser.agent import Agent
from aiser.config import AiServerConfig, AiApiSpecs
from aiser.utils import meets_minimum_version


class RestAiServer(AiServer):
    def __init__(
            self,
            complete_url: typing.Optional[str] = None,
            knowledge_bases: typing.Optional[typing.List[KnowledgeBase]] = None,
            agents: typing.Optional[typing.List[Agent]] = None,
            host: str = "127.0.0.1",
            port: int = 5000,
            workers: typing.Optional[int] = None,
            config: typing.Optional[AiServerConfig] = None,
            authenticator: typing.Optional[RestAuthenticator] = None
    ):
        super().__init__(
            complete_url=complete_url,
            knowledge_bases=knowledge_bases,
            agents=agents,
            host=host,
            port=port,
            config=config
        )
        self._workers = workers
        self._authenticator = authenticator or self._determine_authenticator_fallback()

    def _determine_authenticator_fallback(self) -> RestAuthenticator:
        if self._config.server_environment == ServerEnvironment.DEVELOPMENT:
            return NonFunctionalRestAuthenticator()
        return AsymmetricJwtRestAuthenticator(
            complete_server_url=self._config.complete_url,
            consumer=self._config.consumer,
        )

    def get_app(self) -> FastAPI:
        verify_token = self._authenticator.get_authentication_dependency(
            acceptable_subjects=self._get_list_of_identifiable_entity_ids()
        )

        def verify_meets_minimum_version(request: Request):
            min_version = request.headers.get("Min-Aiser-Version")
            if min_version is None:
                return
            if not meets_minimum_version(
                    current_version=self.get_aiser_version(),
                    min_version=min_version
            ):
                error_message = f"Minimum version required: {min_version}. Current version: {self.get_aiser_version()}"
                print(error_message)
                raise HTTPException(
                    status_code=status.HTTP_426_UPGRADE_REQUIRED,
                    detail=error_message
                )

        non_authenticated_router = APIRouter()
        authenticated_router = APIRouter(dependencies=[
            Depends(verify_token),
            Depends(verify_meets_minimum_version)
        ])

        @non_authenticated_router.get("/")
        async def read_root():
            return "ok"

        @authenticated_router.get("/version")
        async def version() -> VersionInfo:
            return VersionInfo(version=self.get_aiser_version())

        @authenticated_router.post("/knowledge-base/{kb_id}/semantic-search")
        async def knowledge_base(
                kb_id: str, request: SemanticSearchRequest,
        ) -> SemanticSearchResultResponseDto:
            for kb in self._knowledge_bases:
                if kb.accepts_id(kb_id):
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
                yield AgentChatResponse(outputMessage=message_dto).model_dump_json(by_alias=True) + "\n"

        @authenticated_router.post(AiApiSpecs.classic_agent_api_v1.path + "/{agent_id}/chat")
        async def agent_chat(
                agent_id: str,
                request: AgentChatRequest,
        ) -> StreamingResponse:
            for agent in self._agents:
                if agent.accepts_id(agent_id):
                    messages = [ChatMessage(text_content=messageDto.textContent) for messageDto in request.messages]
                    response_generator = agent.reply(messages=messages)
                    response_generator = convert_agent_message_gen_to_streaming_response(message_gen=response_generator)
                    return StreamingResponse(
                        response_generator,
                        media_type="text/event-stream"
                    )
            raise HTTPException(status_code=404, detail="Agent not found")

        app = FastAPI()
        app.include_router(authenticated_router)
        app.include_router(non_authenticated_router)

        return app

    def run(self):
        uvicorn.run(app=self.get_app(), port=self._port, host=self._host, workers=self._workers or 1)
