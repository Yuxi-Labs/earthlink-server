"""Agent-to-agent messaging system."""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    # Direct communication
    DIRECT = "direct"  # Point-to-point message
    BROADCAST = "broadcast"  # To all agents
    MULTICAST = "multicast"  # To specific group

    # Coordination
    REQUEST = "request"  # Request something from another agent
    RESPONSE = "response"  # Response to a request
    ACK = "ack"  # Acknowledgment

    # Knowledge sharing
    SHARE_KNOWLEDGE = "share_knowledge"  # Share learned information
    QUERY_KNOWLEDGE = "query_knowledge"  # Ask for knowledge

    # Collaboration
    COLLABORATE_REQUEST = "collaborate_request"  # Propose collaboration
    COLLABORATE_ACCEPT = "collaborate_accept"
    COLLABORATE_REJECT = "collaborate_reject"
    COLLABORATE_UPDATE = "collaborate_update"  # Update on shared task

    # Goals
    GOAL_PROPOSAL = "goal_proposal"  # Propose a shared goal
    GOAL_ASSIGNMENT = "goal_assignment"  # Assign a goal to agent

    # Social
    GREETING = "greeting"
    STATUS_UPDATE = "status_update"


class MessagePriority(int, Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Message:
    """Inter-agent message."""

    id: UUID = field(default_factory=uuid4)
    type: MessageType = MessageType.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL

    # Routing
    sender_id: UUID | None = None
    recipient_id: UUID | None = None  # None for broadcast
    recipient_ids: list[UUID] = field(default_factory=list)  # For multicast

    # Content
    subject: str = ""
    content: Any = None
    embedding: list[float] | None = None  # Semantic embedding

    # Conversation tracking
    conversation_id: UUID | None = None  # Groups related messages
    reply_to: UUID | None = None  # Reference to previous message

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl: int | None = None  # Time-to-live in seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    delivered: bool = False
    read: bool = False
    delivered_at: datetime | None = None
    read_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "priority": self.priority.value,
            "sender_id": str(self.sender_id) if self.sender_id else None,
            "recipient_id": str(self.recipient_id) if self.recipient_id else None,
            "recipient_ids": [str(rid) for rid in self.recipient_ids],
            "subject": self.subject,
            "content": self.content,
            "conversation_id": str(self.conversation_id) if self.conversation_id else None,
            "reply_to": str(self.reply_to) if self.reply_to else None,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "metadata": self.metadata,
            "delivered": self.delivered,
            "read": self.read,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            type=MessageType(data.get("type", "direct")),
            priority=MessagePriority(data.get("priority", 1)),
            sender_id=UUID(data["sender_id"]) if data.get("sender_id") else None,
            recipient_id=UUID(data["recipient_id"]) if data.get("recipient_id") else None,
            recipient_ids=[UUID(rid) for rid in data.get("recipient_ids", [])],
            subject=data.get("subject", ""),
            content=data.get("content"),
            conversation_id=UUID(data["conversation_id"]) if data.get("conversation_id") else None,
            reply_to=UUID(data["reply_to"]) if data.get("reply_to") else None,
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {}),
        )

    def create_reply(
        self,
        sender_id: UUID,
        content: Any,
        msg_type: MessageType = MessageType.RESPONSE,
    ) -> "Message":
        """Create a reply to this message."""
        return Message(
            type=msg_type,
            sender_id=sender_id,
            recipient_id=self.sender_id,
            subject=f"Re: {self.subject}",
            content=content,
            conversation_id=self.conversation_id or self.id,
            reply_to=self.id,
        )


@dataclass
class Mailbox:
    """Agent's message mailbox."""

    agent_id: UUID
    inbox: list[Message] = field(default_factory=list)
    outbox: list[Message] = field(default_factory=list)
    max_inbox_size: int = 1000

    def receive(self, message: Message) -> bool:
        """Receive a message into inbox."""
        if len(self.inbox) >= self.max_inbox_size:
            # Remove oldest low-priority messages
            self._cleanup_inbox()

        message.delivered = True
        message.delivered_at = datetime.now(UTC)
        self.inbox.append(message)
        return True

    def get_unread(self, limit: int = 50) -> list[Message]:
        """Get unread messages, highest priority first."""
        unread = [m for m in self.inbox if not m.read]
        unread.sort(key=lambda m: (-m.priority.value, m.timestamp))
        return unread[:limit]

    def get_by_type(self, msg_type: MessageType) -> list[Message]:
        """Get messages by type."""
        return [m for m in self.inbox if m.type == msg_type]

    def get_conversation(self, conversation_id: UUID) -> list[Message]:
        """Get all messages in a conversation."""
        return [
            m for m in self.inbox
            if m.conversation_id == conversation_id or m.id == conversation_id
        ]

    def mark_read(self, message_id: UUID) -> bool:
        """Mark a message as read."""
        for msg in self.inbox:
            if msg.id == message_id:
                msg.read = True
                msg.read_at = datetime.now(UTC)
                return True
        return False

    def mark_all_read(self) -> int:
        """Mark all messages as read. Returns count."""
        count = 0
        for msg in self.inbox:
            if not msg.read:
                msg.read = True
                msg.read_at = datetime.now(UTC)
                count += 1
        return count

    def queue_outgoing(self, message: Message) -> None:
        """Queue a message for sending."""
        self.outbox.append(message)

    def get_outgoing(self) -> list[Message]:
        """Get and clear outgoing messages."""
        messages = self.outbox.copy()
        self.outbox.clear()
        return messages

    def _cleanup_inbox(self) -> None:
        """Remove old low-priority messages to make room."""
        # Keep high-priority and recent messages
        cutoff = datetime.now(UTC)
        kept = []
        for msg in self.inbox:
            if msg.priority >= MessagePriority.HIGH:
                kept.append(msg)
            elif msg.ttl is None:
                kept.append(msg)
            elif (cutoff - msg.timestamp).total_seconds() < msg.ttl:
                kept.append(msg)

        # If still too many, remove oldest read messages
        if len(kept) >= self.max_inbox_size:
            kept = [m for m in kept if not m.read] + [m for m in kept if m.read]
            kept = kept[: self.max_inbox_size - 100]

        self.inbox = kept

    def stats(self) -> dict[str, Any]:
        """Get mailbox statistics."""
        return {
            "inbox_count": len(self.inbox),
            "outbox_count": len(self.outbox),
            "unread_count": sum(1 for m in self.inbox if not m.read),
            "by_type": {
                t.value: sum(1 for m in self.inbox if m.type == t)
                for t in MessageType
                if any(m.type == t for m in self.inbox)
            },
        }
