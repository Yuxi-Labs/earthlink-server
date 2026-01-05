"""Multi-agent interaction and collaboration tests."""

import pytest
import ray
import asyncio
from uuid import uuid4

from src.agents.core.agent import Agent


@pytest.mark.asyncio
async def test_multi_agent_spawn():
    """Test spawning multiple agents simultaneously."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Spawn 3 agents
    agent_refs = []
    for i in range(3):
        agent_ref = Agent.remote(name=f"Agent{i+1}")
        await agent_ref.initialize_components.remote()
        agent_refs.append(agent_ref)
    
    # Verify all agents spawned
    states = await asyncio.gather(*[ref.get_state.remote() for ref in agent_refs])
    
    assert len(states) == 3
    for state in states:
        assert "id" in state
        assert "name" in state
        assert state["lifecycle"] in ["idle", "training", "IDLE", "TRAINING"]
    
    print(f"✓ Successfully spawned {len(agent_refs)} agents")


@pytest.mark.asyncio
async def test_multi_agent_different_locations():
    """Test multiple agents exploring different locations simultaneously."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Create 3 agents in different cities
    locations = [
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631},
        {"name": "Brisbane", "lat": -27.4698, "lon": 153.0251},
    ]
    
    agent_refs = []
    for i, loc in enumerate(locations):
        agent_ref = Agent.remote(name=f"Explorer_{loc['name']}")
        await agent_ref.initialize_components.remote()
        await agent_ref.set_earthlink_position.remote(loc["lat"], loc["lon"], 10.0)
        agent_refs.append(agent_ref)
    
    # Execute exploration steps in parallel
    step_tasks = [ref.autonomous_step.remote() for ref in agent_refs]
    results = await asyncio.gather(*step_tasks)
    
    # Verify all agents explored
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result is not None
        assert "type" in result


@pytest.mark.asyncio
async def test_multi_agent_scalability():
    """Test system can handle 10+ agents simultaneously."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    num_agents = 10
    
    # Spawn agents
    agent_refs = []
    for i in range(num_agents):
        agent_ref = Agent.remote(name=f"ScaleAgent{i+1}")
        await agent_ref.initialize_components.remote()
        await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
        agent_refs.append(agent_ref)
    
    # Run 1 exploration step for all agents in parallel
    tasks = [ref.autonomous_step.remote() for ref in agent_refs]
    results = await asyncio.gather(*tasks)
    
    # Verify all completed
    assert len(results) == num_agents
    
    successful = sum(1 for r in results if r is not None)
    
    # Get all states
    states = await asyncio.gather(*[ref.get_state.remote() for ref in agent_refs])
    
    active_count = sum(1 for s in states if s["metrics"]["total_steps"] >= 1)
    
    assert active_count >= num_agents * 0.8  # At least 80% success
