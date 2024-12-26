import time
import typing
import logging
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, APIRouter, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from .openai_compatible_api import ChatCompletionRequest, ChatService
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        self._chat_services: Dict[str, ChatService] = {}
        super().__init__(
            complete_url=complete_url,
            knowledge_bases=knowledge_bases,
            agents=agents,
            host=host,
            port=port,
            config=config
        )
        
        # Initialize chat services for each agent
        if agents:
            for agent in agents:
                self._chat_services[agent.get_id()] = ChatService(agent)
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
                logger.error(error_message)
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

        @authenticated_router.post(AiApiSpecs.openai_compatible_api_v1.path + "/chat/completions")
        async def openai_compatible_chat_completion(request: ChatCompletionRequest, raw_request: Request):
            """
            OpenAI-compatible chat completions endpoint supporting both streaming and non-streaming modes.
            Uses the existing agent abstraction under the hood.
            """
            # Log request details for debugging
            logger.debug("=== OpenAI Compatible API Request Debug Info ===")
            logger.debug(f"Method: {raw_request.method}")
            logger.debug(f"URL: {raw_request.url}")
            
            logger.debug("Headers:")
            for name, value in raw_request.headers.items():
                logger.debug(f"{name}: {value}")
            
            logger.debug("Request Body:")
            logger.debug(f"Model: {request.model}")
            logger.debug(f"Stream: {request.stream}")
            
            logger.debug("Messages:")
            for msg in request.messages:
                logger.debug(f"Role: {msg.role}")
                logger.debug(f"Content: {msg.content}")
                logger.debug("---")
            logger.debug("============================================")
            
            try:
                # Extract agent ID from model field (assuming model field contains agent ID)
                agent_id = request.model
                
                if agent_id not in self._chat_services:
                    raise HTTPException(status_code=404, detail="Agent not found")
            except Exception as e:
                logger.error(f"Validation error in request: {str(e)}")
                logger.error(f"Full request data: {request.model_dump_json()}")
                raise
            
            chat_service = self._chat_services[agent_id]
            
            if request.stream:
                return StreamingResponse(
                    chat_service.generate_stream(request.messages, request.model),
                    media_type="text/event-stream",
                )

            return await chat_service.create_chat_completion(request.messages, request.model)

        app = FastAPI(debug=True)
        
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            logger.info(f"Incoming request: {request.method} {request.url}")
            # log the request body
            body = await request.body()
            logger.info(f"Request body: {await request.body()}")
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                raise
        
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            logger.error(f"FastAPI validation error: {str(exc)}")
            logger.error("Validation error details:")
            for error in exc.errors():
                logger.error(f"Location: {error['loc']}")
                logger.error(f"Message: {error['msg']}")
                logger.error(f"Type: {error['type']}")
                logger.error("---")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder({"detail": exc.errors()}),
            )
        app.include_router(authenticated_router)
        app.include_router(non_authenticated_router)

        return app

    def run(self):
        uvicorn.run(app=self.get_app(), port=self._port, host=self._host, workers=self._workers or 1)
