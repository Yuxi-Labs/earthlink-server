"""Autonomous spawning integration test."""

import pytest

pytest.importorskip("ray")

from src.simulation import SimulationRunner, SimulationConfig


pytestmark = pytest.mark.asyncio


async def test_spawn_manager_reaches_target_population(monkeypatch, ray_session):
    """Spawn manager should auto-create agents up to target population."""

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

    # World should be created automatically on initialize
    assert runner.world is not None
    assert runner.world._is_loaded

    # Invoke spawn logic directly
    await runner._maybe_spawn_agents()

    # Population should meet or exceed target
    assert len(runner._agents) >= config.target_agents

    await runner.shutdown()


async def test_earth_created_on_init(monkeypatch, ray_session):
    """Simulation should create and load Earth on initialization."""

    async def _noop_load_persisted(self):
        return None

    monkeypatch.setattr(SimulationRunner, "_load_persisted_agents", _noop_load_persisted)

    config = SimulationConfig()
    runner = SimulationRunner(config=config)

    # World should not exist before initialize
    assert runner.world is None

    await runner.initialize()

    # Earth should be created and loaded
    assert runner.world is not None
    assert runner.world.name == "Earth"
    assert runner.world._is_loaded

    await runner.shutdown()

    # World should be unloaded after shutdown
    assert runner.world is None
