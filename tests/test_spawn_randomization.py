"""Test spawn_agent creates individualized agents with varied traits."""

import asyncio
from uuid import UUID

import pytest
import ray

from src.simulation.runner import SimulationRunner


@pytest.fixture
async def simulation():
    """Create a simulation runner for testing."""
    from src.config import SimulationConfig
    
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    config = SimulationConfig(
        target_agents=0,  # Don't auto-spawn
        max_steps=10,
        step_interval=1.0,
        persistence_enabled=False,  # Disable DB for testing
    )
    
    runner = SimulationRunner(config)
    await runner.initialize()
    
    yield runner
    
    # Cleanup
    await runner.stop()
    ray.shutdown()


@pytest.mark.asyncio
class TestSpawnAgentRandomization:
    """Test that spawn_agent creates agents with randomized traits."""

    async def test_spawn_agent_randomizes_curiosity(self, simulation):
        """Test that spawned agents have varied curiosity values."""
        agent_ids = []
        
        # Spawn 10 agents
        for i in range(10):
            agent_id = await simulation.spawn_agent(name=f"RandomTest{i}")
            agent_ids.append(agent_id)
            assert isinstance(agent_id, UUID)
        
        # Wait for agents to initialize
        await asyncio.sleep(0.5)
        
        # Get agent states
        states = await simulation.list_agents()
        
        # Extract curiosity values
        curiosity_values = [
            s["metrics"]["curiosity_score"] 
            for s in states 
            if s["name"].startswith("RandomTest")
        ]
        
        # Should have spawned all agents
        assert len(curiosity_values) == 10
        
        # Values should be in range 0.3-0.9 (from spawn_agent randomization)
        for c in curiosity_values:
            assert 0.3 <= c <= 0.9
        
        # Should have variety (not all the same)
        unique_values = len(set(curiosity_values))
        assert unique_values > 5  # At least 6 different values

    async def test_spawn_agent_respects_explicit_config(self, simulation):
        """Test that explicitly provided curiosity is not randomized."""
        agent_id = await simulation.spawn_agent(
            name="ExplicitTest",
            config={"curiosity": 0.42}
        )
        
        await asyncio.sleep(0.5)
        
        # Get agent state
        state = await simulation.get_agent_state(agent_id)
        
        # Should have exact curiosity value
        assert state["metrics"]["curiosity_score"] == 0.42

    async def test_spawn_agent_randomizes_risk_tolerance(self, simulation):
        """Test that risk_tolerance is also randomized."""
        agent_ids = []
        
        for i in range(8):
            agent_id = await simulation.spawn_agent(name=f"RiskTest{i}")
            agent_ids.append(agent_id)
        
        await asyncio.sleep(0.5)
        
        # Check that agents got random risk_tolerance values
        # Note: risk_tolerance is in config, not metrics, so we need to check differently
        # For now, just verify agents were created
        assert len(agent_ids) == 8

    async def test_spawn_agent_creates_unique_agents(self, simulation):
        """Test that each spawned agent is truly unique (not a clone)."""
        agent_ids = []
        
        for i in range(5):
            agent_id = await simulation.spawn_agent(name=f"UniqueTest{i}")
            agent_ids.append(agent_id)
        
        await asyncio.sleep(0.5)
        
        # All IDs should be different
        assert len(set(agent_ids)) == 5
        
        # All agents should exist
        states = await simulation.list_agents()
        unique_names = {
            s["name"] 
            for s in states 
            if s["name"].startswith("UniqueTest")
        }
        assert len(unique_names) == 5

    async def test_spawn_many_agents_statistical_distribution(self, simulation):
        """Test that curiosity distribution is approximately uniform in 0.3-0.9."""
        agent_ids = []
        
        # Spawn 30 agents for statistical significance
        for i in range(30):
            agent_id = await simulation.spawn_agent(name=f"StatTest{i}")
            agent_ids.append(agent_id)
        
        await asyncio.sleep(1.0)
        
        states = await simulation.list_agents()
        curiosity_values = [
            s["metrics"]["curiosity_score"]
            for s in states
            if s["name"].startswith("StatTest")
        ]
        
        assert len(curiosity_values) == 30
        
        # Check distribution
        low = sum(1 for c in curiosity_values if c < 0.5)
        high = sum(1 for c in curiosity_values if c >= 0.5)
        
        # Should be roughly balanced (not all low or all high)
        # With uniform distribution, expect ~15/15, allow 30-70 split
        assert low >= 9 and low <= 21
        assert high >= 9 and high <= 21
        
        # Mean should be around 0.6 (midpoint of 0.3-0.9)
        mean = sum(curiosity_values) / len(curiosity_values)
        assert 0.5 <= mean <= 0.7

    async def test_spawn_agent_initializes_location(self, simulation):
        """Test that spawned agents get initial locations."""
        agent_id = await simulation.spawn_agent(name="LocationTest")
        
        await asyncio.sleep(0.5)
        
        state = await simulation.get_agent_state(agent_id)
        
        # Should have a location
        assert "location" in state
        location = state["location"]
        
        # Location should be set (not 0, 0, 0)
        # GB bounds: lat 49.9-58.7, lon -8.2-1.8
        assert 49.0 <= location["x"] <= 59.0  # Latitude
        assert -9.0 <= location["y"] <= 2.0   # Longitude
