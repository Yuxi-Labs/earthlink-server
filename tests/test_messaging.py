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
    
    # Mark read
    mailbox.mark_read(msg.id)
    assert msg.read is True
    
    unread = mailbox.get_unread()
    assert len(unread) == 0


def test_message_router():
    """Test message router."""
    router = MessageRouter()
    
    agent1_id = uuid4()
    agent2_id = uuid4()
    
    # Register agents
    router.register_agent(agent1_id)
    router.register_agent(agent2_id)
    
    assert len(router.agents) == 2
    
    # Send direct message
    msg = router.send_direct(
        sender_id=agent1_id,
        recipient_id=agent2_id,
        subject="Test",
        content="Hello",
    )
    
    # Check agent2 received it
    messages = router.get_messages(agent2_id, unread_only=False)
    assert len(messages) == 1
    assert messages[0].sender_id == agent1_id


def test_message_router_broadcast():
    """Test broadcast messaging."""
    router = MessageRouter()
    
    agents = [uuid4() for _ in range(5)]
    for aid in agents:
        router.register_agent(aid)
    
    # Broadcast from first agent
    router.broadcast(
        sender_id=agents[0],
        subject="Broadcast",
        content="Hello all",
    )
    
    # All other agents should receive
    for i in range(1, 5):
        messages = router.get_messages(agents[i], unread_only=False)
        assert len(messages) == 1
        assert messages[0].subject == "Broadcast"


def test_message_router_groups():
    """Test group messaging."""
    router = MessageRouter()
    
    agent1 = uuid4()
    agent2 = uuid4()
    agent3 = uuid4()
    
    router.register_agent(agent1)
    router.register_agent(agent2)
    router.register_agent(agent3)
    
    # Add to group
    router.add_to_group(agent1, "researchers")
    router.add_to_group(agent2, "researchers")
    
    # Send to group
    router.send_to_group(
        sender_id=agent3,
        group="researchers",
        subject="Group message",
        content="Hello researchers",
    )
    
    # Only agent1 and agent2 should receive
    assert len(router.get_messages(agent1, unread_only=False)) == 1
    assert len(router.get_messages(agent2, unread_only=False)) == 1
    assert len(router.get_messages(agent3, unread_only=False)) == 0
