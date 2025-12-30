"""Autonomous spawning integration test."""

import asyncio
import importlib.util
import sys

import pytest

try:
    from src.simulation import SimulationRunner, SimulationConfig
    _ray_present = importlib.util.find_spec("ray") is not None
except ImportError:
    SimulationRunner = None
    SimulationConfig = None
    _ray_present = False

RAY_AVAILABLE = _ray_present and sys.version_info < (3, 13) and SimulationRunner is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available for current interpreter")
async def test_spawn_manager_reaches_target_population(monkeypatch):
    """Spawn manager should auto-create agents up to target and assign a world."""

    # Avoid DB access during test
    async def _noop_load_persisted(self):
        return None

    monkeypatch.setattr(SimulationRunner, "_load_persisted_agents", _noop_load_persisted)

    config = SimulationConfig(
        target_agents=2,
        max_agents=3,
        spawn_interval_seconds=0.1,
        steps_per_second=50.0,
        max_steps=30,
    )
    runner = SimulationRunner(config=config)

    await runner.initialize()

    # Ensure a default world exists for assignment
    runner.create_world(world_id="earth-home", name="Earth Home", world_type="earth")

    # Directly invoke spawn logic to avoid scheduler timing issues in CI
    await runner._maybe_spawn_agents("earth-home")

    # Population should meet or exceed target
    assert len(runner._agents) >= config.target_agents

    # Default world should exist and agents should be assigned
    default_world = runner.world_registry.get_world("earth-home")
    assert default_world is not None
    assert all(world_id for world_id in runner._agent_worlds.values())

    await runner.shutdown()
