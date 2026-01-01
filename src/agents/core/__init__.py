"""Agent core module - autonomous intelligent actors."""

from .agent import Agent, AgentActor
from .messaging import Mailbox, Message, MessagePriority, MessageType
from .router import MessageRouter, get_message_router, init_message_router
from .state import AgentLifecycle, AgentMetrics, AgentState, Coordinates, AgentStatus
from .monitoring import SelfMonitoringModule
from .generation import GenerationModule
from .specialization import SpecializationModule
from .knowledge_transfer import KnowledgeTransferModule
from .modeling import ModelingModule
from .world_model import WorldModel, AgentModel, Concept
from .evolution import EvolutionModule
from .communication import CommunicationModule
from .adaptation import AdaptationModule

__all__ = [
    # Agent
    "Agent",
    "AgentActor",
    "AgentState",
    "AgentLifecycle",
    "AgentStatus",
    "AgentMetrics",
    "Coordinates",
    "SelfMonitoringModule",
    "GenerationModule",
    "SpecializationModule",
    "KnowledgeTransferModule",
    "ModelingModule",
    "WorldModel",
    "AgentModel",
    "Concept",
    "EvolutionModule",
    "CommunicationModule",
    "AdaptationModule",
    # Messaging
    "Message",
    "MessageType",
    "MessagePriority",
    "Mailbox",
    # Router
    "MessageRouter",
    "get_message_router",
    "init_message_router",
]
