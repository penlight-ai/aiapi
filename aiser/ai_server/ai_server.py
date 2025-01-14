from abc import ABC, abstractmethod
from typing import List, Optional

from ..knowledge_base import KnowledgeBase
from ..agent import ServerAgent
from ..config import AiServerConfig, make_ai_server_config
from ..version import __version__


class AiServer(ABC):
    _knowledge_bases: List[KnowledgeBase]
    _agents: List[ServerAgent]
    _port: int

    def __init__(
            self,
            host: str,
            port: int,
            complete_url: Optional[bool],
            knowledge_bases: Optional[List[KnowledgeBase]] = None,
            agents: Optional[List[ServerAgent]] = None,
            config: Optional[AiServerConfig] = None
    ):
        super().__init__()
        self._port = port
        self._host = host
        self._knowledge_bases = [] if knowledge_bases is None else knowledge_bases
        self._agents = [] if agents is None else agents
        self._config: AiServerConfig = config or make_ai_server_config(
            complete_url=complete_url
        )

    def get_aiser_version(self) -> str:
        return __version__

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def _get_list_of_identifiable_entity_ids(self) -> List[str]:
        agent_ids = [
            agent.get_id()
            for agent in self._agents
        ]
        knowledge_base_ids = [
            knowledge_base.get_id()
            for knowledge_base in self._knowledge_bases
        ]
        return agent_ids + knowledge_base_ids
