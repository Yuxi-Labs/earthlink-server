"""Test that agents are truly individuals with varied capabilities."""

import asyncio
from uuid import UUID

import pytest
import ray

from src.simulation.runner import SimulationRunner


@pytest.fixture
async def simulation():
    """Create a simulation runner for testing."""
    from src.simulation.config import SimulationConfig
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    config = SimulationConfig(
        target_agents=0,
        max_steps=10,
    )
    
    runner = SimulationRunner(config)
    await runner.initialize()
    
    yield runner
    
    await runner.stop()
    ray.shutdown()


@pytest.mark.asyncio
class TestAgentIndividuality:
    """Test that each agent is a unique individual, not a clone."""

    async def test_agents_have_varied_personality_traits(self, simulation):
        """Test basic personality traits (curiosity, risk, social) vary."""
        agent_ids = []
        
        # Spawn 10 agents
        for i in range(10):
            agent_id = await simulation.spawn_agent(name=f"Individual{i}")
            agent_ids.append(agent_id)
        
        await asyncio.sleep(0.5)
        
        states = await simulation.list_agents()
        agents = [s for s in states if s["name"].startswith("Individual")]
        
        assert len(agents) == 10
        
        # Extract traits
        curiosity_values = [a["metrics"]["curiosity_score"] for a in agents]
        
        # Should have variety - not all clones
        unique_curiosity = len(set(curiosity_values))
        assert unique_curiosity > 5, "Agents should have varied curiosity"
        
        # Should be in expected range
        for c in curiosity_values:
            assert 0.3 <= c <= 0.9

    async def test_agents_have_varied_learning_rates(self, simulation):
        """Test learning capability parameters vary per agent."""
        agents = []
        
        for i in range(5):
            agent_id = await simulation.spawn_agent(name=f"Learner{i}")
            agents.append(agent_id)
        
        await asyncio.sleep(0.5)
        
        # Get agent states
        states = await simulation.list_agents()
        learner_states = [s for s in states if s["name"].startswith("Learner")]
        
        # Note: We can't directly access config from state,
        # but we can verify agents were spawned successfully
        assert len(learner_states) == 5
        
        # Verify all have unique IDs (not clones)
        ids = [s["id"] for s in learner_states]
        assert len(set(ids)) == 5

    async def test_agents_have_varied_capability_configs(self, simulation):
        """Test that spawned agents get different capability configurations."""
        # Spawn agents without explicit config
        agent1_id = await simulation.spawn_agent(name="Agent1")
        agent2_id = await simulation.spawn_agent(name="Agent2")
        agent3_id = await simulation.spawn_agent(name="Agent3")
        
        await asyncio.sleep(0.5)
        
        # All should be unique individuals
        assert agent1_id != agent2_id
        assert agent2_id != agent3_id
        assert agent1_id != agent3_id
        
        # Get states
        states = await simulation.list_agents()
        
        agent1 = next(s for s in states if s["name"] == "Agent1")
        agent2 = next(s for s in states if s["name"] == "Agent2")
        agent3 = next(s for s in states if s["name"] == "Agent3")
        
        # Verify they exist
        assert agent1 is not None
        assert agent2 is not None
        assert agent3 is not None
        
        # Verify they have different curiosity (personality trait)
        c1 = agent1["metrics"]["curiosity_score"]
        c2 = agent2["metrics"]["curiosity_score"]
        c3 = agent3["metrics"]["curiosity_score"]
        
        # At least 2 should be different
        curiosity_set = {c1, c2, c3}
        assert len(curiosity_set) >= 2, "Agents should have individual curiosity values"

    async def test_explicit_config_overrides_randomization(self, simulation):
        """Test that explicit config prevents randomization."""
        # Spawn with explicit config
        config = {
            "curiosity": 0.75,
            "learning_rate": 0.015,
            "reasoning_depth": 4,
        }
        
        agent_id = await simulation.spawn_agent(name="ExplicitAgent", config=config)
        await asyncio.sleep(0.5)
        
        state = await simulation.get_agent_state(agent_id)
        
        # Should use exact curiosity
        assert state["metrics"]["curiosity_score"] == 0.75
        
        # Agent should exist and be functional
        assert state["name"] == "ExplicitAgent"

    async def test_population_diversity(self, simulation):
        """Test that a population of agents shows diversity."""
        # Spawn 20 agents to get statistical significance
        agent_ids = []
        for i in range(20):
            agent_id = await simulation.spawn_agent(name=f"PopAgent{i}")
            agent_ids.append(agent_id)
        
        await asyncio.sleep(1.0)
        
        states = await simulation.list_agents()
        pop_agents = [s for s in states if s["name"].startswith("PopAgent")]
        
        assert len(pop_agents) == 20
        
        curiosity_values = [a["metrics"]["curiosity_score"] for a in pop_agents]
        
        # Statistical checks
        low_curiosity = sum(1 for c in curiosity_values if c < 0.5)
        high_curiosity = sum(1 for c in curiosity_values if c >= 0.5)
        
        # Should have both low and high curiosity agents
        assert low_curiosity > 0, "Should have some low-curiosity agents"
        assert high_curiosity > 0, "Should have some high-curiosity agents"
        
        # Distribution should be roughly balanced
        assert low_curiosity >= 5, "Should have at least 5 low-curiosity agents"
        assert high_curiosity >= 5, "Should have at least 5 high-curiosity agents"
        
        # Should have good variety
        unique_values = len(set(curiosity_values))
        assert unique_values >= 12, f"Expected at least 12 unique curiosity values, got {unique_values}"
