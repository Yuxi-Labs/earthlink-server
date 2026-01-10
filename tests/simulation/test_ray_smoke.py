"""Minimal Ray-backed smoke tests for TDD coverage."""

import pytest

from src.agents.core.agent import Agent

pytestmark = pytest.mark.asyncio


async def test_ray_actor_round_trip(ray_session):
    """Ray should run a simple actor call end-to-end."""
    ray = ray_session

    @ray.remote
    class Echo:
        async def ping(self, value: str) -> str:
            return value

    echo = Echo.remote()
    assert await echo.ping.remote("pong") == "pong"


async def test_agent_autonomous_step_runs(ray_session):
    """Agent Ray actor should initialize and execute one autonomous step."""
    ray = ray_session

    agent_ref = Agent.remote(name="RaySmokeAgent")
    await agent_ref.initialize_components.remote()
    await agent_ref.set_earthlink_position.remote(0.0, 0.0, 0.0)

    result = await agent_ref.autonomous_step.remote()
    assert result is not None
    assert isinstance(result, dict)
    assert "type" in result

    state = await agent_ref.get_state.remote()
    assert isinstance(state, dict)
    assert "metrics" in state
