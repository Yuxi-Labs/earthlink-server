"""End-to-end simulation tests with agent autonomous exploration."""

import pytest
import ray

from src.agents.core.agent import Agent


@pytest.mark.asyncio
async def test_agent_autonomous_exploration_single_step():
    """Test agent autonomous exploration in a single step."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Create agent
    agent_ref = Agent.remote(name="Explorer1")
    await agent_ref.initialize_components.remote()
    
    # Spawn in Sydney
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
    
    # Run one autonomous step
    result = await agent_ref.autonomous_step.remote()
    
    # Verify step executed
    assert result is not None
    assert "type" in result
    assert result["type"] in ["knowledge_exploration", "spatial_exploration", "idle"]
    
    # Check agent state updated
    state = await agent_ref.get_state.remote()
    assert state["metrics"]["total_steps"] >= 1


@pytest.mark.asyncio
async def test_agent_autonomous_exploration_multiple_steps():
    """Test agent running multiple autonomous exploration steps."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    agent_ref = Agent.remote(name="Explorer2")
    await agent_ref.initialize_components.remote()
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
    
    # Run 5 autonomous steps
    steps_taken = 0
    action_types = []
    
    for i in range(5):
        result = await agent_ref.autonomous_step.remote()
        steps_taken += 1
        action_types.append(result["type"])
    
    # Verify agent explored
    assert steps_taken == 5
    assert len(action_types) == 5
    
    # Check state
    state = await agent_ref.get_state.remote()
    assert state["metrics"]["total_steps"] >= 5


@pytest.mark.asyncio
async def test_agent_curiosity_driven_behavior():
    """Test agent's curiosity module drives exploration."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    agent_ref = Agent.remote(name="CuriousAgent")
    await agent_ref.initialize_components.remote()
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
    
    # Track curiosity-driven actions
    explore_actions = 0
    
    for i in range(10):
        result = await agent_ref.autonomous_step.remote()
        
        if result["type"] in ["knowledge_exploration", "spatial_exploration"]:
            explore_actions += 1
    
    # Agent should be curious and explore
    assert explore_actions > 0, "Agent should take exploration actions"
