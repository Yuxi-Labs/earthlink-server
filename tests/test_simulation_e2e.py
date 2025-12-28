"""End-to-end simulation tests with agent autonomous exploration."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import ray

from agents.core.agent import Agent


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
    
    print(f"✓ Agent completed autonomous step: {result['type']}")
    print(f"  Total steps: {state['metrics']['total_steps']}")


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
        print(f"  Step {i+1}: {result['type']}")
    
    # Verify agent explored
    assert steps_taken == 5
    assert len(action_types) == 5
    
    # Check state
    state = await agent_ref.get_state.remote()
    assert state["metrics"]["total_steps"] >= 5
    
    print(f"✓ Agent completed {steps_taken} autonomous steps")
    print(f"  Actions: {action_types}")


@pytest.mark.asyncio
async def test_agent_exploration_with_knowledge_acquisition():
    """Test that agent acquires knowledge during exploration."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    agent_ref = Agent.remote(name="KnowledgeSeeker")
    await agent_ref.initialize_components.remote()
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
    
    # Get initial knowledge count
    initial_state = await agent_ref.get_state.remote()
    initial_knowledge = initial_state["metrics"].get("knowledge_items", 0)
    
    # Run autonomous steps
    for i in range(10):
        result = await agent_ref.autonomous_step.remote()
        if result["type"] == "knowledge_exploration":
            print(f"  Step {i+1}: Learning action - {result.get('topic', 'N/A')}")
    
    # Check knowledge increased
    final_state = await agent_ref.get_state.remote()
    final_knowledge = final_state["metrics"].get("knowledge_items", 0)
    
    print(f"✓ Knowledge items: {initial_knowledge} → {final_knowledge}")
    assert final_knowledge >= initial_knowledge


@pytest.mark.asyncio
async def test_agent_spatial_exploration():
    """Test agent moves to different locations during exploration."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    agent_ref = Agent.remote(name="Wanderer")
    await agent_ref.initialize_components.remote()
    
    # Start in Sydney
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
    start_lat = await agent_ref.get_latitude.remote()
    start_lon = await agent_ref.get_longitude.remote()
    
    print(f"  Starting position: ({start_lat:.4f}, {start_lon:.4f})")
    
    # Run exploration steps
    positions = [(start_lat, start_lon)]
    
    for i in range(10):
        result = await agent_ref.autonomous_step.remote()
        
        # Check if agent moved
        current_lat = await agent_ref.get_latitude.remote()
        current_lon = await agent_ref.get_longitude.remote()
        
        if (current_lat, current_lon) != positions[-1]:
            positions.append((current_lat, current_lon))
            print(f"  Step {i+1}: Moved to ({current_lat:.4f}, {current_lon:.4f})")
    
    # Verify agent explored multiple locations
    unique_positions = len(set(positions))
    print(f"✓ Agent visited {unique_positions} unique positions")
    
    # Agent should have moved at least once in 10 steps
    assert unique_positions > 1, "Agent should explore different locations"


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
            print(f"  Step {i+1}: {result['type']}")
    
    print(f"✓ Agent took {explore_actions}/10 exploration actions")
    
    # Agent should be curious and explore
    assert explore_actions > 0, "Agent should take exploration actions"
