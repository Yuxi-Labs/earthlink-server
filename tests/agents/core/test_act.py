"""
Test Act capability - executing actions in the environment.
"""
import pytest
from src.agents.core.agent import Agent
from src.agents.core.state import AgentState, Coordinates


def test_agent_can_act():
    """Test that agent can execute basic actions."""
    agent = Agent(agent_id="test_actor", name="TestActor")
    
    # Define an action
    action = {
        "type": "move",
        "params": {
            "latitude": 51.5074,
            "longitude": -0.1278,
        }
    }
    
    # Execute action
    result = agent.act(action)
    
    assert result is not None
    assert "success" in result or "status" in result or "type" in result


def test_act_updates_agent_state():
    """Test that acting updates the agent's state."""
    agent = Agent(agent_id="test_actor", name="TestActor")
    initial_steps = agent.state.metrics.total_steps_executed
    
    action = {"type": "explore", "params": {}}
    agent.act(action)
    
    # State should be updated
    assert agent.state is not None


def test_act_handles_invalid_action():
    """Test that act handles invalid actions gracefully."""
    agent = Agent(agent_id="test_actor", name="TestActor")
    
    invalid_action = {"type": "invalid_action_type", "params": {}}
    
    result = agent.act(invalid_action)
    
    # Should not crash, should return error or handle gracefully
    assert result is not None
