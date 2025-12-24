"""Tests for simulation runner."""

import pytest

from src.simulation import SimulationRunner, SimulationConfig, SimulationState


@pytest.mark.asyncio
async def test_simulation_initialization():
    """Test simulation runner initialization."""
    config = SimulationConfig(steps_per_second=10.0)
    runner = SimulationRunner(config=config)
    
    await runner.initialize()
    
    assert runner.state == SimulationState.IDLE
    assert runner.total_steps == 0
    
    await runner.shutdown()


@pytest.mark.asyncio
async def test_simulation_start_stop():
    """Test starting and stopping simulation."""
    config = SimulationConfig(steps_per_second=10.0)
    runner = SimulationRunner(config=config)
    
    await runner.initialize()
    await runner.start()
    
    assert runner.state == SimulationState.RUNNING
    
    await runner.pause()
    assert runner.state == SimulationState.PAUSED
    
    await runner.shutdown()


@pytest.mark.asyncio
async def test_simulation_stats():
    """Test simulation statistics."""
    config = SimulationConfig(steps_per_second=10.0)
    runner = SimulationRunner(config=config)
    
    await runner.initialize()
    
    stats = runner.get_stats()
    
    assert "state" in stats
    assert "total_steps" in stats
    assert "num_agents" in stats
    assert "num_worlds" in stats
    
    await runner.shutdown()
