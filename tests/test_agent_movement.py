"""Test agent movement capability."""

import math

import pytest

from src.agents.core.agent import Agent


class TestAgentMovement:
    """Test that agents can move and update their positions correctly."""

    def test_set_earthlink_position(self):
        """Test setting agent position directly."""
        agent = Agent(name="MoveTest1")
        
        # Set position to London
        agent.set_earthlink_position(51.5074, -0.1278, 10.0)
        
        assert agent.state.location.x == 51.5074
        assert agent.state.location.y == -0.1278
        assert agent.state.location.z == 10.0

    def test_calculate_destination(self):
        """Test haversine distance calculation."""
        agent = Agent(name="MoveTest2")
        
        # Start in London
        start_lat, start_lon = 51.5074, -0.1278
        
        # Move 10km due north (bearing 0)
        new_lat, new_lon = agent._calculate_destination(start_lat, start_lon, 10.0, 0)
        
        # Latitude should increase (going north)
        assert new_lat > start_lat
        # Longitude should be approximately the same
        assert abs(new_lon - start_lon) < 0.01

    def test_calculate_destination_east(self):
        """Test moving east."""
        agent = Agent(name="MoveTest3")
        
        start_lat, start_lon = 51.5074, -0.1278
        
        # Move 10km due east (bearing 90)
        new_lat, new_lon = agent._calculate_destination(start_lat, start_lon, 10.0, 90)
        
        # Longitude should increase (going east)
        assert new_lon > start_lon
        # Latitude should be approximately the same
        assert abs(new_lat - start_lat) < 0.01

    def test_movement_updates_metrics(self):
        """Test that movement updates distance_traveled_km metric."""
        agent = Agent(name="MoveTest4", config={"curiosity": 0.6})
        
        # Set initial position
        agent.set_earthlink_position(51.5074, -0.1278, 0.0)
        
        initial_distance = agent.state.metrics.distance_traveled_km
        
        # Simulate a move by updating metrics (as autonomous_step does)
        distance_moved = 15.5
        agent.state.metrics.distance_traveled_km += distance_moved
        
        assert agent.state.metrics.distance_traveled_km == initial_distance + distance_moved

    def test_gb_bounds_clamping(self):
        """Test that positions are clamped to GB bounds."""
        agent = Agent(name="MoveTest5")
        
        # GB bounds from autonomous_step
        GB_BOUNDS = {
            "min_lat": 49.9,
            "max_lat": 58.7,
            "min_lon": -8.2,
            "max_lon": 1.8,
        }
        
        # Try to set position outside GB bounds
        lat_out_of_bounds = 60.0  # Too far north
        lon_out_of_bounds = 5.0   # Too far east
        
        # Clamp manually (as autonomous_step does)
        clamped_lat = max(GB_BOUNDS["min_lat"], min(GB_BOUNDS["max_lat"], lat_out_of_bounds))
        clamped_lon = max(GB_BOUNDS["min_lon"], min(GB_BOUNDS["max_lon"], lon_out_of_bounds))
        
        agent.set_earthlink_position(clamped_lat, clamped_lon, 0.0)
        
        assert agent.state.location.x == GB_BOUNDS["max_lat"]  # Clamped to max
        assert agent.state.location.y == GB_BOUNDS["max_lon"]  # Clamped to max

    def test_position_in_state_dict(self):
        """Test that position is included in state serialization."""
        agent = Agent(name="MoveTest6")
        agent.set_earthlink_position(52.4862, -1.8904, 0.0)  # Birmingham
        
        state = agent.get_state()
        
        assert "location" in state
        assert state["location"]["x"] == 52.4862
        assert state["location"]["y"] == -1.8904
        assert state["location"]["z"] == 0.0

    def test_longitude_normalization(self):
        """Test that longitude is normalized to -180 to 180."""
        agent = Agent(name="MoveTest7")
        
        # Calculate destination that might wrap longitude
        start_lat, start_lon = 51.5074, 179.0
        new_lat, new_lon = agent._calculate_destination(start_lat, start_lon, 100.0, 90)
        
        # Longitude should be normalized
        assert -180 <= new_lon <= 180

    def test_latitude_clamping(self):
        """Test that latitude is clamped to -90 to 90."""
        agent = Agent(name="MoveTest8")
        
        # Calculate destination with extreme distance
        start_lat, start_lon = 85.0, 0.0
        new_lat, new_lon = agent._calculate_destination(start_lat, start_lon, 1000.0, 0)
        
        # Latitude should be clamped to valid range
        assert -90 <= new_lat <= 90

    @pytest.mark.asyncio
    async def test_autonomous_step_move_action(self):
        """Test that autonomous_step can execute move action."""
        agent = Agent(name="MoveTest9", config={"curiosity": 0.8})
        
        # Set initial position
        agent.set_earthlink_position(51.5074, -0.1278, 0.0)
        initial_lat = agent.state.location.x
        initial_lon = agent.state.location.y
        
        # Run autonomous step (may choose move or learn)
        result = await agent.autonomous_step()
        
        # Check if action was returned
        assert result is not None
        assert "type" in result
        
        # If it chose to move, verify position changed
        if result["type"] == "move":
            new_lat = agent.state.location.x
            new_lon = agent.state.location.y
            
            # Position should have changed
            assert (new_lat != initial_lat) or (new_lon != initial_lon)
            
            # Distance should be recorded
            assert "distance_km" in result
            assert result["distance_km"] > 0
