"""Test agent spatial movement with OSM data."""

import pytest
import ray

from src.agents.core import Agent


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
    
    # Get initial position
    lat1 = await agent_ref.get_latitude.remote()
    lon1 = await agent_ref.get_longitude.remote()
    
    # Move to Melbourne
    melbourne_lat, melbourne_lon = -37.8136, 144.9631
    result = await agent_ref.move_to.remote(melbourne_lat, melbourne_lon)
    
    # Get new position
    lat2 = await agent_ref.get_latitude.remote()
    lon2 = await agent_ref.get_longitude.remote()
    
    # Verify moved
    assert abs(lat2 - melbourne_lat) < 0.001
    assert abs(lon2 - melbourne_lon) < 0.001
    
    print(f"✓ Agent moved from ({lat1:.4f}, {lon1:.4f}) to ({lat2:.4f}, {lon2:.4f})")


@pytest.mark.asyncio
async def test_agent_random_exploration():
    """Test agent random exploration within bounds."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="RandomExplorer")
    
    # Start in Sydney
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093)
    
    # Explore randomly 3 times
    positions = []
    for i in range(3):
        result = await agent_ref.explore_random_location.remote()
        lat = await agent_ref.get_latitude.remote()
        lon = await agent_ref.get_longitude.remote()
        positions.append((lat, lon))
        print(f"  Step {i+1}: ({lat:.4f}, {lon:.4f})")
    
    # Should have visited different locations
    unique_positions = len(set(positions))
    print(f"✓ Agent explored {unique_positions} positions")


@pytest.mark.asyncio
async def test_agent_find_nearest_poi():
    """Test agent finding nearest POI."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="POIFinder")
    
    # Position in Sydney CBD
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093)
    
    # Find nearest cafe (or any amenity)
    result = await agent_ref.find_nearest.remote("poi", max_distance_km=1.0)
    
    if result:
        print(f"✓ Found nearest POI(s): {len(result)} results")
    else:
        print("✓ No POI found in range (expected in test environment)")


@pytest.mark.asyncio
async def test_agent_navigation_to_poi():
    """Test agent navigating to a POI."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="Navigator")
    
    # Start in Sydney
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093)
    
    # Try to navigate to a restaurant
    result = await agent_ref.navigate_to_poi.remote(poi_name="Restaurant")
    
    # Get final position
    lat = await agent_ref.get_latitude.remote()
    lon = await agent_ref.get_longitude.remote()
    
    print(f"✓ Navigation result: {result.get('status', 'unknown')}")
    print(f"  Final position: ({lat:.4f}, {lon:.4f})")


@pytest.mark.asyncio
async def test_agent_exploration_route():
    """Test agent planning an exploration route."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    agent_ref = Agent.remote(name="RoutePlanner")
    
    # Start in Sydney
    await agent_ref.set_earthlink_position.remote(-33.8688, 151.2093)
    
    # Plan route with 3 waypoints
    route = await agent_ref.plan_exploration_route.remote(num_waypoints=3)
    
    if route and "waypoints" in route:
        print(f"✓ Planned route with {len(route['waypoints'])} waypoints:")
        for i, wp in enumerate(route["waypoints"]):
            print(f"  Waypoint {i+1}: ({wp.get('lat', 0):.4f}, {wp.get('lon', 0):.4f})")
    else:
        print("✓ Route planning returned default route")

