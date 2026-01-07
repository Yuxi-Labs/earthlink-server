"""
Communication capability: structured messaging, negotiation, and broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import UUID

from .messaging import Message, MessagePriority, MessageType


@dataclass
class MessageResult:
    """Result of a send/broadcast/negotiation operation."""

    success: bool
    recipients: list[str]
    rationale: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "recipients": self.recipients,
            "rationale": self.rationale,
            "created_at": self.created_at.isoformat(),
        }


class CommunicationModule:
    """Send/receive messages with intents, negotiation, and broadcast."""

    def __init__(self, agent_id: UUID | None = None, mailbox: Any | None = None):
        from .messaging import Mailbox
        self.agent_id = agent_id
        self.mailbox = mailbox if mailbox is not None else Mailbox(agent_id=agent_id)

    async def send_message(
        self,
        recipient: UUID,
        content: dict,
        intent: MessageType = MessageType.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL,
        subject: str = "",
    ) -> MessageResult:
        """Send structured message with intent."""
        msg = Message(
            type=intent,
            priority=priority,
            sender_id=self.agent_id,
            recipient_id=recipient,
            subject=subject or intent.value,
            content=content,
        )
        self.mailbox.outbox.append(msg)
        rationale = f"Queued message to {recipient}"
        return MessageResult(success=True, recipients=[str(recipient)], rationale=rationale)

    async def negotiate(
        self,
        agents: list[UUID],
        negotiation_topic: dict,
    ) -> MessageResult:
        """Multi-agent negotiation protocol (simplified broadcast proposal)."""
        for aid in agents:
            await self.send_message(
                recipient=aid,
                content={"proposal": negotiation_topic},
                intent=MessageType.REQUEST,
                priority=MessagePriority.HIGH,
                subject="negotiation_proposal",
            )
        return MessageResult(
            success=True,
            recipients=[str(a) for a in agents],
            rationale="Broadcast negotiation proposal to agents.",
        )

    async def broadcast(
        self,
        content: dict,
        filter_condition: Callable[[Message], bool] | None = None,
    ) -> MessageResult:
        """Broadcast to all known agents (represented in mailbox router)."""
        recipients = []
        # The mailbox router will handle actual delivery; here we queue a broadcast.
        msg = Message(
            type=MessageType.BROADCAST,
            priority=MessagePriority.NORMAL,
            sender_id=self.agent_id,
            recipient_id=None,
            subject="broadcast",
            content=content,
        )
        if filter_condition is None or filter_condition(msg):
            self.mailbox.outbox.append(msg)
            recipients.append("broadcast")

        rationale = "Queued broadcast message"
        return MessageResult(success=True, recipients=recipients, rationale=rationale)
