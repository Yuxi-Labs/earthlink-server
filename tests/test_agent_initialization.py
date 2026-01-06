"""Test agent initialization with individualized traits."""

import asyncio
from uuid import uuid4

import pytest

from src.agents.core.agent import Agent


class TestAgentInitialization:
    """Test that agents are created with unique, individualized traits."""

    def test_agent_with_specific_curiosity(self):
        """Test agent initializes with curiosity from config."""
        config = {"curiosity": 0.75}
        agent = Agent(agent_id=uuid4(), name="TestAgent1", config=config)
        
        assert agent.state.metrics.curiosity_score == 0.75
        assert agent.config["curiosity"] == 0.75

    def test_agent_without_curiosity_uses_default(self):
        """Test agent without curiosity config uses default value."""
        agent = Agent(agent_id=uuid4(), name="TestAgent2", config={})
        
        # Default from AgentMetrics dataclass is 0.5
        assert agent.state.metrics.curiosity_score == 0.5

    def test_multiple_agents_have_varied_curiosity(self):
        """Test that multiple agents can have different curiosity values."""
        agents = [
            Agent(agent_id=uuid4(), name=f"Agent{i}", config={"curiosity": i * 0.1})
            for i in range(3, 10)  # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        
        curiosity_values = [a.state.metrics.curiosity_score for a in agents]
        
        # All should be different
        assert len(set(curiosity_values)) == len(curiosity_values)
        
        # Should match config
        for i, agent in enumerate(agents):
            expected = (i + 3) * 0.1
            assert abs(agent.state.metrics.curiosity_score - expected) < 0.01

    def test_agent_state_serialization_includes_curiosity(self):
        """Test that get_state() includes curiosity_score."""
        config = {"curiosity": 0.62}
        agent = Agent(agent_id=uuid4(), name="TestAgent3", config=config)
        
        state = agent.get_state()
        
        assert "metrics" in state
        assert "curiosity_score" in state["metrics"]
        assert state["metrics"]["curiosity_score"] == 0.62

    def test_curiosity_persists_across_state_dict_conversion(self):
        """Test curiosity survives to_dict() conversion."""
        config = {"curiosity": 0.88}
        agent = Agent(agent_id=uuid4(), name="TestAgent4", config=config)
        
        state_dict = agent.state.to_dict()
        
        assert state_dict["metrics"]["curiosity_score"] == 0.88

    def test_agent_initialization_with_other_traits(self):
        """Test agents can have other individualized traits."""
        config = {
            "curiosity": 0.7,
            "risk_tolerance": 0.4,
            "social_preference": 0.6,
        }
        agent = Agent(agent_id=uuid4(), name="TestAgent5", config=config)
        
        assert agent.config["curiosity"] == 0.7
        assert agent.config["risk_tolerance"] == 0.4
        assert agent.config["social_preference"] == 0.6
        assert agent.state.metrics.curiosity_score == 0.7
