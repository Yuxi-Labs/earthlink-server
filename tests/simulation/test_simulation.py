"""Tests for simulation runner."""

import importlib.util
import sys

import pytest

try:
    from src.simulation import SimulationRunner, SimulationConfig, SimulationState
    _ray_present = importlib.util.find_spec("ray") is not None
except ImportError:
    SimulationRunner = None
    SimulationConfig = None
    SimulationState = None
    _ray_present = False

RAY_AVAILABLE = _ray_present and sys.version_info < (3, 13) and SimulationRunner is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available for current interpreter")
async def test_simulation_initialization(monkeypatch):
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


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available for current interpreter")
async def test_simulation_stats(monkeypatch):
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


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available for current interpreter")
async def test_world_is_earth(monkeypatch):
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
