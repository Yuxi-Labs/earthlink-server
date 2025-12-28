"""Test agent spatial movement with OSM data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import ray
from agents.core import Agent


@pytest.mark.asyncio
async def test_agent_spawn_and_position():
    """Test spawning agent and setting position."""
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": "/app/src:/app"}}
        )
    
    # Spawn agent
    agent_ref = Agent.remote(name="TestExplorer", config={"state_dim": 256})
    
    # Set position in Sydney, Australia
    sydney_lat, sydney_lon = -33.8688, 151.2093
    await agent_ref.set_earthlink_position.remote(sydney_lat, sydney_lon, altitude=10.0)
    
    # Verify position
    lat = await agent_ref.get_latitude.remote()
    lon = await agent_ref.get_longitude.remote()
    alt = await agent_ref.get_altitude.remote()
    
    assert abs(lat - sydney_lat) < 0.001
    assert abs(lon - sydney_lon) < 0.001
    assert abs(alt - 10.0) < 0.001
    
    print(f"✓ Agent spawned at Sydney: ({lat:.4f}, {lon:.4f}, {alt}m)")


@pytest.mark.asyncio
async def test_agent_movement():
    """Test agent movement between locations."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="MovementTester")
    
    # Start in Sydney
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093)
    
    # Move to Melbourne
    melbourne_lat, melbourne_lon = -37.8136, 144.9631
    result = await agent_ref.move_to.remote(melbourne_lat, melbourne_lon)
    
    assert result["type"] == "movement"
    assert result["new_position"]["lat"] == melbourne_lat
    assert result["new_position"]["lon"] == melbourne_lon
    assert result["distance_km"] > 700  # Sydney to Melbourne is ~700km
    
    print(f"✓ Agent moved {result['distance_km']:.1f} km from Sydney to Melbourne")


@pytest.mark.asyncio
async def test_explore_random_location():
    """Test random exploration in Australia."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="RandomExplorer")
    await agent_ref.initialize_components.remote()
    
    # Explore random location in Australia
    result = await agent_ref.explore_random_location.remote()
    
    assert result["type"] == "random_exploration"
    assert "movement" in result
    assert "exploration" in result
    
    new_pos = result["movement"]["new_position"]
    print(f"✓ Agent explored random location: ({new_pos['lat']:.4f}, {new_pos['lon']:.4f})")
    print(f"  Nearby features: {result['movement']['nearby_features']}")
    print(f"  Regions: {result['movement']['containing_regions']}")


@pytest.mark.asyncio
async def test_find_nearest_features():
    """Test finding nearest buildings/POIs."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="FeatureFinder")
    
    # Position in Sydney CBD
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093)
    
    # Find nearest buildings
    buildings = await agent_ref.find_nearest.remote("buildings", max_distance_km=2.0, limit=10)
    
    assert len(buildings) > 0, "Should find buildings in Sydney CBD"
    assert all("distance_km" in b for b in buildings)
    
    print(f"✓ Found {len(buildings)} buildings near Sydney CBD")
    for i, building in enumerate(buildings[:3], 1):
        print(f"  {i}. {building['name']} - {building['distance_km']:.2f} km away")


@pytest.mark.asyncio
async def test_plan_exploration_route():
    """Test multi-waypoint route planning."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="RoutePlanner")
    
    # Start in Sydney
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093)
    
    # Plan 5-waypoint route
    route = await agent_ref.plan_exploration_route.remote(num_waypoints=5)
    
    assert len(route) == 5
    assert all("lat" in wp and "lon" in wp for wp in route)
    
    total_distance = sum(wp["distance_from_previous_km"] for wp in route)
    print(f"✓ Planned {len(route)}-waypoint route")
    print(f"  Total distance: {total_distance:.1f} km")
    for i, wp in enumerate(route, 1):
        print(f"  Waypoint {i}: ({wp['lat']:.4f}, {wp['lon']:.4f})")


if __name__ == "__main__":
    import asyncio
    
    async def run_all():
        print("\n=== Testing Agent Spatial Movement ===\n")
        
        await test_agent_spawn_and_position()
        print()
        
        await test_agent_movement()
        print()
        
        await test_explore_random_location()
        print()
        
        await test_find_nearest_features()
        print()
        
        await test_plan_exploration_route()
        
        print("\n=== All Tests Passed ===\n")
    
    asyncio.run(run_all())
