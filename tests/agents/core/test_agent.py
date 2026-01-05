"""Tests for agent core functionality."""

import pytest
from uuid import uuid4

from src.agents.core import Agent, AgentState, MessageType


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization."""
    agent_id = uuid4()
    agent = Agent(agent_id=agent_id, name="TestAgent")
    
    assert agent.agent_id == agent_id
    assert agent.name == "TestAgent"
    assert agent.state.step == 0
    assert len(agent.state.observations) == 0


@pytest.mark.asyncio
async def test_agent_step():
    """Test agent step execution."""
    agent = Agent(agent_id=uuid4(), name="TestAgent")
    
    # Initial step
    await agent.step()
    
    assert agent.state.step == 1
    assert len(agent.state.observations) > 0


@pytest.mark.asyncio
async def test_agent_messaging():
    """Test agent message sending/receiving."""
    agent1 = Agent(agent_id=uuid4(), name="Agent1")
    agent2 = Agent(agent_id=uuid4(), name="Agent2")
    
    # Send message from agent1 to agent2
    msg = agent1.send_message(
        recipient_id=agent2.agent_id,
        subject="Test",
        content="Hello"
    )
    
    assert msg is not None
    assert msg.sender_id == agent1.agent_id
    assert msg.recipient_id == agent2.agent_id
    assert msg.subject == "Test"
    assert msg.content == "Hello"


def test_agent_state_serialization():
    """Test agent state to/from dict."""
    state = AgentState(
        step=10,
        position=[1.0, 2.0],
        velocity=[0.1, 0.2],
    )
    
    # To dict
    state_dict = state.to_dict()
    assert state_dict["step"] == 10
    assert state_dict["position"] == [1.0, 2.0]
    
    # From dict
    restored = AgentState.from_dict(state_dict)
    assert restored.step == 10
    assert restored.position == [1.0, 2.0]

