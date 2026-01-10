"""Tests for messaging system."""

import pytest
from uuid import uuid4

from src.agents.core import Message, MessageType, MessagePriority, Mailbox, MessageRouter
from src.simulation.events import EventBus


def test_message_creation():
    """Test message creation."""
    sender = uuid4()
    recipient = uuid4()
    
    msg = Message(
        type=MessageType.DIRECT,
        priority=MessagePriority.HIGH,
        sender_id=sender,
        recipient_id=recipient,
        subject="Test",
        content="Hello",
    )
    
    assert msg.sender_id == sender
    assert msg.recipient_id == recipient
    assert msg.type == MessageType.DIRECT
    assert msg.priority == MessagePriority.HIGH


def test_mailbox():
    """Test mailbox functionality."""
    agent_id = uuid4()
    mailbox = Mailbox(agent_id=agent_id)
    
    # Receive message
    msg = Message(
        sender_id=uuid4(),
        recipient_id=agent_id,
        subject="Test",
        content="Hello",
    )
    
    mailbox.receive(msg)
    
    assert len(mailbox.inbox) == 1
    assert mailbox.inbox[0].id == msg.id
    
    # Get unread
    unread = mailbox.get_unread()
    assert len(unread) == 1


def test_mailbox_mark_read():
    """Test marking messages as read."""
    agent_id = uuid4()
    mailbox = Mailbox(agent_id=agent_id)
    
    msg = Message(
        sender_id=uuid4(),
        recipient_id=agent_id,
        subject="Test",
        content="Hello",
    )
    
    mailbox.receive(msg)
    
    # Mark as read
    mailbox.mark_read(msg.id)
    
    # Should have no unread
    unread = mailbox.get_unread()
    assert len(unread) == 0


def test_message_router():
    """Test message routing."""
    event_bus = EventBus()
    router = MessageRouter(event_bus=event_bus)
    
    agent_id = uuid4()
    mailbox = Mailbox(agent_id=agent_id)
    
    # Register agent with mailbox
    router.register_agent(agent_id, mailbox=mailbox)
    
    # Route message
    msg = Message(
        sender_id=uuid4(),
        recipient_id=agent_id,
        subject="Routed",
        content="Hello via router",
    )
    
    router.send(msg)
    
    # Should be in mailbox
    assert len(mailbox.inbox) == 1
    assert mailbox.inbox[0].subject == "Routed"


def test_broadcast_message():
    """Test broadcasting to multiple agents."""
    event_bus = EventBus()
    router = MessageRouter(event_bus=event_bus)
    
    # Create multiple agents
    mailboxes = []
    for _ in range(3):
        agent_id = uuid4()
        mailbox = Mailbox(agent_id=agent_id)
        router.register_agent(agent_id, mailbox=mailbox)
        mailboxes.append(mailbox)
    
    # Broadcast
    sender_id = uuid4()
    router.broadcast(
        sender_id=sender_id,
        subject="Broadcast",
        content="Hello everyone",
    )
    
    # All should receive
    for mailbox in mailboxes:
        assert len(mailbox.inbox) == 1
        assert mailbox.inbox[0].subject == "Broadcast"

