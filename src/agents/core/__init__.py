"""Agent core module - autonomous intelligent actors."""

from .agent import Agent
from .messaging import Mailbox, Message, MessagePriority, MessageType
from .router import MessageRouter, get_message_router, init_message_router
from .state import AgentLifecycle, AgentMetrics, AgentState, Coordinates, AgentStatus

__all__ = [
    # Agent
    "Agent",
    "AgentState",
    "AgentLifecycle",
    "AgentStatus",
    "AgentMetrics",
    "Coordinates",
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
