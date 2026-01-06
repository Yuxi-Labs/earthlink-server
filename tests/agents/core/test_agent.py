"""Tests for agent core functionality."""

import pytest
from uuid import uuid4

from src.agents.core import Agent, AgentState, MessageType


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization."""
    agent_id = uuid4()
    agent = Agent(agent_id=agent_id, name="TestAgent")
    
    assert agent.get_id() == agent_id
    assert agent.state.name == "TestAgent"
    assert agent.state.metrics.total_steps_executed == 0


@pytest.mark.asyncio
async def test_agent_step():
    """Test agent step execution."""
    agent = Agent(agent_id=uuid4(), name="TestAgent")
    
    # Initial step (step() is synchronous, not async)
    action = agent.step(observation={})
    
    assert agent.state.metrics.total_steps_executed == 1
    assert action is not None


@pytest.mark.asyncio
async def test_agent_messaging():
    """Test agent message sending/receiving."""
    agent1 = Agent(agent_id=uuid4(), name="Agent1")
    agent2 = Agent(agent_id=uuid4(), name="Agent2")
    
    # Send message from agent1 to agent2
    msg = agent1.send_message(
        recipient_id=agent2.get_id(),
        subject="Test",
        content="Hello"
    )
    
    assert msg is not None
    assert msg["sender_id"] == str(agent1.get_id())
    assert msg["recipient_id"] == str(agent2.get_id())
    assert msg["subject"] == "Test"
    assert msg["content"] == "Hello"


def test_agent_state_serialization():
    """Test agent state to/from dict."""
    state = AgentState()
    state.metrics.total_steps_executed = 10
    state.location.x = 1.0
    state.location.y = 2.0
    
    # To dict
    state_dict = state.to_dict()
    assert state_dict["metrics"]["total_steps_executed"] == 10
    assert state_dict["location"]["x"] == 1.0
    assert state_dict["location"]["y"] == 2.0

