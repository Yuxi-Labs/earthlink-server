"""Tests for simulation runner."""

import pytest

pytest.importorskip("ray")

from src.simulation import SimulationRunner, SimulationConfig, SimulationState


pytestmark = pytest.mark.asyncio


async def test_simulation_initialization(monkeypatch, ray_session):
    """Test simulation runner initialization."""
    async def _noop_load_persisted(self):
        return None

    monkeypatch.setattr(SimulationRunner, "_load_persisted_agents", _noop_load_persisted)

    config = SimulationConfig(steps_per_second=10.0)
    runner = SimulationRunner(config=config)
    
    # World should not exist before initialize
    assert runner.world is None
    
    await runner.initialize()
    
    assert runner.state == SimulationState.IDLE
    assert runner._total_steps == 0
    
    # World should be created and loaded
    assert runner.world is not None
    assert runner.world._is_loaded
    
    await runner.shutdown()


async def test_simulation_stats(monkeypatch, ray_session):
    """Test simulation statistics."""
    async def _noop_load_persisted(self):
        return None

    monkeypatch.setattr(SimulationRunner, "_load_persisted_agents", _noop_load_persisted)

    config = SimulationConfig(steps_per_second=10.0)
    runner = SimulationRunner(config=config)
    
    await runner.initialize()
    
    stats = runner.get_stats()
    
    assert "state" in stats
    assert "total_steps" in stats
    assert "num_agents" in stats
    assert "world" in stats
    assert stats["world"] == "Earth"
    
    await runner.shutdown()


async def test_world_is_earth(monkeypatch, ray_session):
    """Test that the simulation's world is Earth."""
    async def _noop_load_persisted(self):
        return None

    monkeypatch.setattr(SimulationRunner, "_load_persisted_agents", _noop_load_persisted)

    config = SimulationConfig()
    runner = SimulationRunner(config=config)
    
    await runner.initialize()
    
    # Verify the world is Earth
    assert runner.world is not None
    assert runner.world.id == "earth"
    assert runner.world.name == "Earth"
    assert runner.world.metadata.world_type == "spatial"
    
    await runner.shutdown()
