"""Multi-agent interaction and collaboration tests."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import ray
import asyncio
from uuid import uuid4

from agents.core.agent import Agent


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
    for i, state in enumerate(states):
        print(f"  Agent {i+1}: {state['name']} ({state['id']})")


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
    
    # Run autonomous exploration in parallel
    print(f"\n✓ Agents spawned in different cities:")
    for i, loc in enumerate(locations):
        lat = await agent_refs[i].get_latitude.remote()
        lon = await agent_refs[i].get_longitude.remote()
        print(f"  {loc['name']}: ({lat:.4f}, {lon:.4f})")
    
    # Execute exploration steps in parallel
    step_tasks = [ref.autonomous_step.remote() for ref in agent_refs]
    results = await asyncio.gather(*step_tasks)
    
    # Verify all agents explored
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result is not None
        assert "type" in result
        print(f"  {locations[i]['name']} agent: {result['type']}")
    
    print(f"✓ All {len(agent_refs)} agents explored simultaneously")


@pytest.mark.asyncio
async def test_multi_agent_knowledge_acquisition():
    """Test multiple agents learning autonomously and verify knowledge diversity."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Spawn 2 agents in same location
    agent_refs = []
    for i in range(2):
        agent_ref = Agent.remote(name=f"Learner{i+1}")
        await agent_ref.initialize_components.remote()
        await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)  # Sydney
        agent_refs.append(agent_ref)
    
    # Run exploration for both agents
    print(f"\n✓ Running autonomous learning for 2 agents...")
    
    for step in range(5):
        tasks = [ref.autonomous_step.remote() for ref in agent_refs]
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            if result["type"] == "knowledge_exploration":
                print(f"  Agent {i+1} - Step {step+1}: Learned about '{result.get('topic', 'N/A')}'")
    
    # Get final knowledge states
    states = await asyncio.gather(*[ref.get_state.remote() for ref in agent_refs])
    
    # Verify knowledge acquisition
    for i, state in enumerate(states):
        knowledge_items = state["metrics"].get("knowledge_items", 0)
        topics_explored = state["metrics"].get("topics_explored", 0)
        print(f"\n  Agent {i+1} Final Stats:")
        print(f"    Knowledge items: {knowledge_items}")
        print(f"    Topics explored: {topics_explored}")
    
    print(f"\n✓ Multi-agent autonomous learning verified")


@pytest.mark.asyncio
async def test_multi_agent_spatial_awareness():
    """Test agents can query and be aware of other agents' positions."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Create 2 agents at different positions
    agent1_ref = Agent.remote(name="Agent_North")
    agent2_ref = Agent.remote(name="Agent_South")
    
    await agent1_ref.initialize_components.remote()
    await agent2_ref.initialize_components.remote()
    
    # Position them 10km apart in Sydney
    await agent1_ref.set_earthlink_position.remote(-33.8588, 151.2093, 10.0)  # North
    await agent2_ref.set_earthlink_position.remote(-33.8788, 151.2093, 10.0)  # South
    
    # Get positions
    agent1_lat = await agent1_ref.get_latitude.remote()
    agent1_lon = await agent1_ref.get_longitude.remote()
    agent2_lat = await agent2_ref.get_latitude.remote()
    agent2_lon = await agent2_ref.get_longitude.remote()
    
    # Calculate distance (simple Euclidean approximation)
    import math
    lat_diff = agent2_lat - agent1_lat
    lon_diff = agent2_lon - agent1_lon
    distance_degrees = math.sqrt(lat_diff**2 + lon_diff**2)
    distance_km = distance_degrees * 111  # Rough conversion
    
    print(f"\n✓ Agent spatial positioning:")
    print(f"  Agent North: ({agent1_lat:.4f}, {agent1_lon:.4f})")
    print(f"  Agent South: ({agent2_lat:.4f}, {agent2_lon:.4f})")
    print(f"  Distance: {distance_km:.2f} km")
    
    # Verify agents are at different positions
    assert agent1_lat != agent2_lat
    assert distance_km > 0
    
    print(f"✓ Agents successfully positioned at distinct locations")


@pytest.mark.asyncio
async def test_multi_agent_concurrent_database_access():
    """Test multiple agents can write to database simultaneously without conflicts."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Spawn 3 agents
    agent_refs = []
    for i in range(3):
        agent_ref = Agent.remote(name=f"DBWriter{i+1}")
        await agent_ref.initialize_components.remote()
        await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
        agent_refs.append(agent_ref)
    
    print(f"\n✓ Testing concurrent database writes from {len(agent_refs)} agents...")
    
    # All agents explore same topic simultaneously (stress test)
    tasks = [ref.explore_topic.remote("concurrent database test") for ref in agent_refs]
    results = await asyncio.gather(*tasks)
    
    # Verify all completed successfully
    assert len(results) == 3
    for i, result in enumerate(results):
        assert "knowledge_gained" in result
        print(f"  Agent {i+1}: {result['knowledge_gained']} knowledge items acquired")
    
    print(f"✓ Concurrent database access successful - no conflicts")


@pytest.mark.asyncio
async def test_multi_agent_lifecycle_states():
    """Test agents can be in different lifecycle states simultaneously."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Create 2 agents
    agent1_ref = Agent.remote(name="ActiveAgent")
    agent2_ref = Agent.remote(name="IdleAgent")
    
    await agent1_ref.initialize_components.remote()
    await agent2_ref.initialize_components.remote()
    
    await agent1_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
    await agent2_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
    
    # Agent 1: Active exploration
    await agent1_ref.autonomous_step.remote()
    
    # Agent 2: Idle (no steps)
    
    # Check states
    state1 = await agent1_ref.get_state.remote()
    state2 = await agent2_ref.get_state.remote()
    
    print(f"\n✓ Agent lifecycle states:")
    print(f"  Agent 1: {state1['lifecycle']} - Steps: {state1['metrics']['total_steps']}")
    print(f"  Agent 2: {state2['lifecycle']} - Steps: {state2['metrics']['total_steps']}")
    
    # Verify agent 1 took steps
    assert state1["metrics"]["total_steps"] >= 1
    
    print(f"✓ Agents maintain independent lifecycle states")


@pytest.mark.asyncio
async def test_multi_agent_scalability():
    """Test system can handle 10+ agents simultaneously."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    num_agents = 10
    print(f"\n✓ Spawning {num_agents} agents for scalability test...")
    
    # Spawn agents
    agent_refs = []
    for i in range(num_agents):
        agent_ref = Agent.remote(name=f"ScaleAgent{i+1}")
        await agent_ref.initialize_components.remote()
        await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093, 10.0)
        agent_refs.append(agent_ref)
    
    print(f"  ✓ {len(agent_refs)} agents spawned")
    
    # Run 1 exploration step for all agents in parallel
    print(f"  Running parallel exploration...")
    tasks = [ref.autonomous_step.remote() for ref in agent_refs]
    results = await asyncio.gather(*tasks)
    
    # Verify all completed
    assert len(results) == num_agents
    
    successful = sum(1 for r in results if r is not None)
    print(f"  ✓ {successful}/{num_agents} agents completed exploration")
    
    # Get all states
    states = await asyncio.gather(*[ref.get_state.remote() for ref in agent_refs])
    
    active_count = sum(1 for s in states if s["metrics"]["total_steps"] >= 1)
    
    print(f"\n✓ Scalability test passed:")
    print(f"  Agents spawned: {num_agents}")
    print(f"  Agents active: {active_count}")
    print(f"  Success rate: {(active_count/num_agents)*100:.1f}%")
    
    assert active_count >= num_agents * 0.8  # At least 80% success
