"""Message router for inter-agent communication."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID

import ray

from src.agents.core.messaging import Mailbox, Message, MessagePriority, MessageType
from src.simulation.events import Event, EventBus, EventType

if TYPE_CHECKING:
    pass


@dataclass
class AgentHandle:
    """Handle to a registered agent."""

    agent_id: UUID
    actor_ref: ray.ObjectRef | None = None  # Ray actor reference
    mailbox: Mailbox = field(default_factory=lambda: Mailbox(UUID(int=0)))
    groups: set[str] = field(default_factory=set)
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        if self.mailbox.agent_id == UUID(int=0):
            self.mailbox = Mailbox(agent_id=self.agent_id)


class MessageRouter:
    """
    Central message router for agent-to-agent communication.

    Handles:
    - Direct messages between agents
    - Broadcast to all agents
    - Multicast to groups
    - Message queuing and delivery
    - Event integration
    """

    def __init__(self, event_bus: EventBus | None = None):
        self.agents: dict[UUID, AgentHandle] = {}
        self.groups: dict[str, set[UUID]] = defaultdict(set)
        self.event_bus = event_bus

        # Message handlers for different types
        self._type_handlers: dict[MessageType, list[Callable]] = defaultdict(list)

        # Delivery queue for async processing
        self._delivery_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._running = False

    # -------------------------------------------------------------------------
    # Agent Registration
    # -------------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: UUID,
        actor_ref: ray.ObjectRef | None = None,
        groups: list[str] | None = None,
        mailbox: Mailbox | None = None,
    ) -> None:
        """Register an agent with the router."""
        handle = AgentHandle(
            agent_id=agent_id,
            actor_ref=actor_ref,
            groups=set(groups or []),
            mailbox=mailbox or Mailbox(agent_id=agent_id),
        )
        self.agents[agent_id] = handle

        # Add to groups
        for group in handle.groups:
            self.groups[group].add(agent_id)

    def unregister_agent(self, agent_id: UUID) -> None:
        """Unregister an agent from the router."""
        if agent_id not in self.agents:
            return

        handle = self.agents[agent_id]

        # Remove from groups
        for group in handle.groups:
            self.groups[group].discard(agent_id)

        del self.agents[agent_id]

    def add_to_group(self, agent_id: UUID, group: str) -> bool:
        """Add an agent to a group."""
        if agent_id not in self.agents:
            return False

        self.agents[agent_id].groups.add(group)
        self.groups[group].add(agent_id)
        return True

    def remove_from_group(self, agent_id: UUID, group: str) -> bool:
        """Remove an agent from a group."""
        if agent_id not in self.agents:
            return False

        self.agents[agent_id].groups.discard(group)
        self.groups[group].discard(agent_id)
        return True

    def get_group_members(self, group: str) -> list[UUID]:
        """Get all agents in a group."""
        return list(self.groups.get(group, set()))

    # -------------------------------------------------------------------------
    # Message Sending
    # -------------------------------------------------------------------------

    def send(self, message: Message) -> bool:
        """
        Send a message to its recipient(s).

        Returns True if message was queued for delivery.
        """
        if message.type == MessageType.BROADCAST:
            return self._send_broadcast(message)
        elif message.type == MessageType.MULTICAST:
            return self._send_multicast(message)
        else:
            return self._send_direct(message)

    def _send_direct(self, message: Message) -> bool:
        """Send direct message to single recipient."""
        if not message.recipient_id:
            return False

        if message.recipient_id not in self.agents:
            return False

        handle = self.agents[message.recipient_id]
        handle.mailbox.receive(message)

        # Publish event
        self._publish_message_event(message)

        return True

    def _send_broadcast(self, message: Message) -> bool:
        """Send message to all agents."""
        sender_id = message.sender_id

        for agent_id, handle in self.agents.items():
            if agent_id != sender_id:  # Don't send to self
                # Create copy for each recipient
                msg_copy = Message(
                    type=message.type,
                    priority=message.priority,
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    subject=message.subject,
                    content=message.content,
                    conversation_id=message.conversation_id,
                    ttl=message.ttl,
                    metadata=message.metadata.copy(),
                )
                handle.mailbox.receive(msg_copy)

        self._publish_message_event(message)
        return True

    def _send_multicast(self, message: Message) -> bool:
        """Send message to multiple recipients or groups."""
        recipients = set(message.recipient_ids)

        # Resolve group names in metadata
        group_names = message.metadata.get("groups", [])
        for group in group_names:
            recipients.update(self.groups.get(group, set()))

        # Remove sender from recipients
        if message.sender_id:
            recipients.discard(message.sender_id)

        # Deliver to each recipient
        for agent_id in recipients:
            if agent_id in self.agents:
                msg_copy = Message(
                    type=message.type,
                    priority=message.priority,
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    subject=message.subject,
                    content=message.content,
                    conversation_id=message.conversation_id,
                    ttl=message.ttl,
                    metadata=message.metadata.copy(),
                )
                self.agents[agent_id].mailbox.receive(msg_copy)

        self._publish_message_event(message)
        return len(recipients) > 0

    def _publish_message_event(self, message: Message) -> None:
        """Publish message event to event bus."""
        if not self.event_bus:
            return

        # Map message type to event type
        event = Event(
            event_type=EventType.AGENT_ACTION,
            source_agent_id=message.sender_id,
            data={
                "action_type": "message",
                "message": message.to_dict(),
            },
            metadata={
                "message_type": message.type.value,
                "recipient_id": str(message.recipient_id) if message.recipient_id else None,
            },
        )
        self.event_bus.publish(event)

    # -------------------------------------------------------------------------
    # Message Receiving
    # -------------------------------------------------------------------------

    def get_messages(
        self,
        agent_id: UUID,
        unread_only: bool = True,
        limit: int = 50,
    ) -> list[Message]:
        """Get messages for an agent."""
        if agent_id not in self.agents:
            return []

        mailbox = self.agents[agent_id].mailbox

        if unread_only:
            return mailbox.get_unread(limit)
        return mailbox.inbox[-limit:]

    def get_messages_by_type(
        self,
        agent_id: UUID,
        msg_type: MessageType,
    ) -> list[Message]:
        """Get messages of a specific type."""
        if agent_id not in self.agents:
            return []

        return self.agents[agent_id].mailbox.get_by_type(msg_type)

    def mark_read(self, agent_id: UUID, message_id: UUID) -> bool:
        """Mark a message as read."""
        if agent_id not in self.agents:
            return False

        return self.agents[agent_id].mailbox.mark_read(message_id)

    def mark_all_read(self, agent_id: UUID) -> int:
        """Mark all messages as read for an agent."""
        if agent_id not in self.agents:
            return 0

        return self.agents[agent_id].mailbox.mark_all_read()

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def send_direct(
        self,
        sender_id: UUID,
        recipient_id: UUID,
        subject: str,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs,
    ) -> Message:
        """Helper to send a direct message."""
        msg = Message(
            type=MessageType.DIRECT,
            priority=priority,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            content=content,
            **kwargs,
        )
        self.send(msg)
        return msg

    def broadcast(
        self,
        sender_id: UUID,
        subject: str,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs,
    ) -> Message:
        """Helper to broadcast a message."""
        msg = Message(
            type=MessageType.BROADCAST,
            priority=priority,
            sender_id=sender_id,
            subject=subject,
            content=content,
            **kwargs,
        )
        self.send(msg)
        return msg

    def send_to_group(
        self,
        sender_id: UUID,
        group: str,
        subject: str,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs,
    ) -> Message:
        """Helper to send to a group."""
        msg = Message(
            type=MessageType.MULTICAST,
            priority=priority,
            sender_id=sender_id,
            subject=subject,
            content=content,
            metadata={"groups": [group]},
            **kwargs,
        )
        self.send(msg)
        return msg

    def request(
        self,
        sender_id: UUID,
        recipient_id: UUID,
        request_type: str,
        content: Any,
        **kwargs,
    ) -> Message:
        """Send a request message expecting a response."""
        msg = Message(
            type=MessageType.REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=request_type,
            content=content,
            **kwargs,
        )
        self.send(msg)
        return msg

    def respond(
        self,
        sender_id: UUID,
        original_message: Message,
        content: Any,
        success: bool = True,
        **kwargs,
    ) -> Message:
        """Send a response to a request."""
        msg = original_message.create_reply(
            sender_id=sender_id,
            content=content,
            msg_type=MessageType.RESPONSE,
        )
        msg.metadata["success"] = success
        msg.metadata.update(kwargs)
        self.send(msg)
        return msg

    # -------------------------------------------------------------------------
    # Knowledge Sharing
    # -------------------------------------------------------------------------

    def share_knowledge(
        self,
        sender_id: UUID,
        recipient_id: UUID | None,
        knowledge_type: str,
        content: Any,
        embedding: list[float] | None = None,
        **kwargs,
    ) -> Message:
        """Share knowledge with another agent or broadcast."""
        msg = Message(
            type=MessageType.SHARE_KNOWLEDGE
            if recipient_id
            else MessageType.BROADCAST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=f"knowledge:{knowledge_type}",
            content=content,
            embedding=embedding,
            **kwargs,
        )
        self.send(msg)
        return msg

    def query_knowledge(
        self,
        sender_id: UUID,
        recipient_id: UUID | None,
        query: str,
        query_embedding: list[float] | None = None,
        **kwargs,
    ) -> Message:
        """Query another agent for knowledge."""
        msg = Message(
            type=MessageType.QUERY_KNOWLEDGE,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject="knowledge_query",
            content={"query": query},
            embedding=query_embedding,
            **kwargs,
        )
        if recipient_id:
            self.send(msg)
        else:
            # Broadcast query
            msg.type = MessageType.BROADCAST
            self.send(msg)
        return msg

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_agent_stats(self, agent_id: UUID) -> dict[str, Any] | None:
        """Get message stats for an agent."""
        if agent_id not in self.agents:
            return None

        return self.agents[agent_id].mailbox.stats()

    def get_router_stats(self) -> dict[str, Any]:
        """Get overall router statistics."""
        total_messages = sum(
            len(h.mailbox.inbox) for h in self.agents.values()
        )
        total_unread = sum(
            sum(1 for m in h.mailbox.inbox if not m.read)
            for h in self.agents.values()
        )

        return {
            "registered_agents": len(self.agents),
            "groups": {g: len(members) for g, members in self.groups.items()},
            "total_messages": total_messages,
            "total_unread": total_unread,
        }


# Global router instance
_router: MessageRouter | None = None


def get_message_router() -> MessageRouter | None:
    """Get the global message router."""
    return _router


def init_message_router(event_bus: EventBus | None = None) -> MessageRouter:
    """Initialize the global message router."""
    global _router
    _router = MessageRouter(event_bus=event_bus)
    return _router
