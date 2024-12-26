from .ai_server_env_variables import get_ai_server_env_variables, AiServerEnvVariables
from ..ai_server_consumer import AiServerConsumer, get_default_consumer
import typing
from pydantic import BaseModel


class ServerEnvironment:
    DEVELOPMENT = 'Development'
    PRODUCTION = 'Production'

class AiApiSpec(BaseModel):
    name: str
    path: str

class AiApiSpecs:
    openai_compatible_api_v1: AiApiSpec = AiApiSpec(
        name='openai compatible api v1',
        path='/openai-compatible-api/v1'
    )

    classic_agent_api_v1: AiApiSpec = AiApiSpec(
        name='classic agent api v1',
        path='/agent'
    )



class AiServerConfig(BaseModel):
    consumer: AiServerConsumer
    complete_url: typing.Optional[str]
    server_environment: str


def make_ai_server_config(
        complete_url: typing.Optional[str],
        consumer: typing.Optional[AiServerConsumer] = None
) -> AiServerConfig:
    config_maker = AiServerConfigMaker(
        complete_url=complete_url,
        env_variables=get_ai_server_env_variables(),
        overriding_consumer=consumer
    )
    return config_maker.make_config()


class AiServerConfigMaker:
    def __init__(
            self,
            complete_url: typing.Optional[str],
            env_variables: AiServerEnvVariables,
            overriding_consumer: typing.Optional[AiServerConsumer] = None,
    ):
        self.complete_url = complete_url
        self.env_variables = env_variables
        self.overriding_consumer = overriding_consumer

    def make_config(self) -> AiServerConfig:
        final_consumer = self._determine_consumer()
        final_server_environment = self._determine_server_environment()

        return AiServerConfig(
            complete_url=self.complete_url,
            consumer=final_consumer,
            server_environment=final_server_environment
        )

    def _determine_consumer(
            self,
    ) -> AiServerConsumer:
        if self.overriding_consumer is not None:
            return self.overriding_consumer
        if self.env_variables.CONSUMER_PUBLIC_KEY_INFO_URL is not None:
            return AiServerConsumer(
                publicKeyInfoUrl=self.env_variables.CONSUMER_PUBLIC_KEY_INFO_URL
            )
        return get_default_consumer()

    def _determine_server_environment(
            self
    ) -> str:
        if self.env_variables.AISER_ENVIRONMENT == ServerEnvironment.DEVELOPMENT:
            return ServerEnvironment.DEVELOPMENT
        return ServerEnvironment.PRODUCTION
