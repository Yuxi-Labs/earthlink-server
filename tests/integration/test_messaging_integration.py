"""Integration tests for messaging system."""

import pytest
from uuid import uuid4

from src.agents.core import MessageRouter, get_message_router, init_message_router
from src.simulation.events import EventBus, EventType


@pytest.mark.asyncio
async def test_message_router_with_event_bus():
    """Test message router integration with event bus."""
    event_bus = EventBus()
    router = init_message_router(event_bus)
    
    events_received = []
    
    def event_handler(event):
        events_received.append(event)
    
    event_bus.subscribe(EventType.MESSAGE_SENT, event_handler)
    
    # Register agents
    agent1 = uuid4()
    agent2 = uuid4()
    router.register_agent(agent1)
    router.register_agent(agent2)
    
    # Send message
    router.send_direct(
        sender_id=agent1,
        recipient_id=agent2,
        subject="Test",
        content="Hello",
    )
    
    # Event should be published
    assert len(events_received) > 0


@pytest.mark.asyncio
async def test_end_to_end_agent_communication():
    """Test end-to-end agent communication flow."""
    router = MessageRouter()
    
    # Create 3 agents
    agents = [uuid4() for _ in range(3)]
    for aid in agents:
        router.register_agent(aid)
    
    # Add to group
    router.add_to_group(agents[0], "team1")
    router.add_to_group(agents[1], "team1")
    
    # Agent 2 sends to group
    router.send_to_group(
        sender_id=agents[2],
        group="team1",
        subject="Team message",
        content="Hello team",
    )
    
    # Check agents in group received it
    assert len(router.get_messages(agents[0], unread_only=False)) == 1
    assert len(router.get_messages(agents[1], unread_only=False)) == 1
    assert len(router.get_messages(agents[2], unread_only=False)) == 0
    
    # Agent 0 replies
    original = router.get_messages(agents[0], unread_only=False)[0]
    router.respond(
        sender_id=agents[0],
        original_message=original,
        content="Got it!",
    )
    
    # Agent 2 should receive reply
    assert len(router.get_messages(agents[2], unread_only=False)) == 1
