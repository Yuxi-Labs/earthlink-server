"""
Integration tests for agent archetype system

These tests actually spawn agents and verify:
1. Archetype detection works with real agent instances
2. Coverage assessment calculates correctly
3. API endpoints return correct data
4. Parameter randomization creates diversity
"""

import pytest
import ray
from httpx import AsyncClient, ASGITransport
from src.simulation.runner import SimulationRunner
from src.simulation.config import SimulationConfig
from src.main import app


@pytest.fixture(scope="function")
async def simulation(ray_session):
    """Initialize real simulation with database"""

    # Configure with no automatic spawning
    config = SimulationConfig(target_agents=0)
    runner = SimulationRunner(config=config)
    # Skip database restore for tests - we want clean state
    await runner.initialize(skip_db_restore=True)
    
    yield runner
    
    # Cleanup
    await runner.shutdown()


class TestArchetypeDetection:
    """Test archetype detection with real spawned agents"""
    
    @pytest.mark.asyncio
    async def test_systematic_mapper_agent(self, simulation):
        """Spawn agent with systematic mapper traits and verify archetype"""
        agent_id = await simulation.spawn_agent(
            name="Systematic Test Agent",
            config={
                "exploration_style": 0.2,  # Methodical (low)
                "depth_vs_breadth": 0.3,  # Breadth-first (low)
                "path_memory_strength": 0.9,  # Strong memory (high)
                "risk_tolerance": 0.3,  # Low risk
                "backtracking_tolerance": 0.2,  # Avoid backtracking (low)
                "detail_orientation": 0.8,  # High detail
                "collaborative_exploration": 0.2,  # Solo explorer (low)
                "adaptation_speed": 0.4,  # Not too adaptive
                "perception_window": 8,  # Moderate perception
            }
        )
        
        agent_ref = simulation._agents[agent_id]
        archetype = await agent_ref.get_exploration_archetype.remote()
        
        assert archetype["archetype"] == "Systematic Mapper"
        assert archetype["confidence"] > 0.5
        assert "Systematic Mapper" in archetype["scores"]
        assert archetype["scores"]["Systematic Mapper"] == archetype["confidence"]
    
    @pytest.mark.asyncio
    async def test_bold_pioneer_agent(self, simulation):
        """Spawn bold pioneer and verify archetype"""
        agent_id = await simulation.spawn_agent(
            name="Bold Test Agent",
            config={
                "exploration_style": 0.8,
                "risk_tolerance": 0.8,
                "curiosity": 0.9,
                "path_memory_strength": 0.2,
                "backtracking_tolerance": 0.2,
                "collaborative_exploration": 0.0,
                "depth_vs_breadth": 0.3,
                "perception_window": 5,
                "detail_orientation": 0.2,
                "obstacle_persistence": 0.2,
                "goal_flexibility": 0.9,
                "exploration_bonus": 0.2,
            }
        )
        
        agent_ref = simulation._agents[agent_id]
        archetype = await agent_ref.get_exploration_archetype.remote()
        
        assert archetype["archetype"] == "Bold Pioneer"
        assert archetype["confidence"] > 0.5
    
    @pytest.mark.asyncio
    async def test_archetype_has_all_scores(self, simulation):
        """Verify archetype detection returns all 7 archetype scores"""
        agent_id = await simulation.spawn_agent(name="Score Test Agent")
        agent_ref = simulation._agents[agent_id]
        archetype = await agent_ref.get_exploration_archetype.remote()
        
        expected_archetypes = [
            "Systematic Mapper",
            "Bold Pioneer",
            "Thorough Investigator",
            "Opportunistic Rover",
            "Team Scout",
            "Cautious Analyst",
            "Persistent Pathfinder"
        ]
        
        assert len(archetype["scores"]) == 7
        for arch in expected_archetypes:
            assert arch in archetype["scores"]
            assert 0 <= archetype["scores"][arch] <= 1
    
    @pytest.mark.asyncio
    async def test_archetype_returns_parameters(self, simulation):
        """Verify archetype includes the agent's parameters"""
        agent_id = await simulation.spawn_agent(
            name="Param Test Agent",
            config={"curiosity": 0.85, "risk_tolerance": 0.65}
        )
        
        agent_ref = simulation._agents[agent_id]
        archetype = await agent_ref.get_exploration_archetype.remote()
        
        assert "parameters" in archetype
        assert archetype["parameters"]["curiosity"] == 0.85
        assert archetype["parameters"]["risk_tolerance"] == 0.65


class TestCoverageAssessment:
    """Test coverage metrics with real agents"""
    
    @pytest.mark.asyncio
    async def test_new_agent_has_zero_coverage(self, simulation):
        """New agent should have zero coverage metrics"""
        agent_id = await simulation.spawn_agent(name="Coverage Test Agent")
        agent_ref = simulation._agents[agent_id]
        coverage = await agent_ref.assess_exploration_coverage.remote()
        
        assert coverage["unique_locations"] == 0
        assert coverage["area_km2"] == 0.0
        assert coverage["distance_traveled_km"] == 0.0
    
    @pytest.mark.asyncio
    async def test_coverage_has_all_metrics(self, simulation):
        """Coverage should include all efficiency metrics"""
        agent_id = await simulation.spawn_agent(name="Metrics Test Agent")
        agent_ref = simulation._agents[agent_id]
        coverage = await agent_ref.assess_exploration_coverage.remote()
        
        required = [
            "unique_locations",
            "area_km2",
            "distance_traveled_km",
            "locations_per_hour",
            "km_per_hour",
            "area_efficiency",
            "knowledge_per_location"
        ]
        
        for metric in required:
            assert metric in coverage
            assert isinstance(coverage[metric], (int, float))
            assert coverage[metric] >= 0


class TestParameterVariation:
    """Test that agents spawn with varied parameters"""
    
    @pytest.mark.asyncio
    async def test_random_agents_have_different_archetypes(self, simulation):
        """Multiple random agents should produce archetype diversity"""
        archetypes = []
        
        for i in range(10):
            agent_id = await simulation.spawn_agent(name=f"Random Agent {i}")
            agent_ref = simulation._agents[agent_id]
            archetype = await agent_ref.get_exploration_archetype.remote()
            archetypes.append(archetype["archetype"])
        
        # Should have at least 3 different archetypes with 10 random agents
        unique_archetypes = set(archetypes)
        assert len(unique_archetypes) >= 3, f"Only {len(unique_archetypes)} archetypes in {archetypes}"
    
    @pytest.mark.asyncio
    async def test_explicit_config_overrides_randomization(self, simulation):
        """Explicitly set parameters should not be randomized"""
        agent_id = await simulation.spawn_agent(
            name="Explicit Config Agent",
            config={"curiosity": 0.999, "risk_tolerance": 0.111}
        )
        
        agent_ref = simulation._agents[agent_id]
        archetype = await agent_ref.get_exploration_archetype.remote()
        
        assert abs(archetype["parameters"]["curiosity"] - 0.999) < 0.001
        assert abs(archetype["parameters"]["risk_tolerance"] - 0.111) < 0.001


class TestArchetypeAPI:
    """Test archetype API endpoints"""
    
    @pytest.mark.asyncio
    async def test_archetype_endpoint(self, simulation):
        """GET /agents/{id}/archetype returns archetype data"""
        agent_id = await simulation.spawn_agent(name="API Test Agent")
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/v1/agents/{agent_id}/archetype")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "archetype" in data
        assert "confidence" in data
        assert "scores" in data
        assert "parameters" in data
        assert len(data["scores"]) == 7
    
    @pytest.mark.asyncio
    async def test_coverage_endpoint(self, simulation):
        """GET /agents/{id}/coverage returns coverage metrics"""
        agent_id = await simulation.spawn_agent(name="Coverage API Test")
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/v1/agents/{agent_id}/coverage")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "unique_locations" in data
        assert "area_km2" in data
        assert "distance_traveled_km" in data
        assert "locations_per_hour" in data
    
    @pytest.mark.asyncio
    async def test_archetype_endpoint_404_for_invalid_id(self):
        """API should return 404 for non-existent agent"""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/v1/agents/{fake_id}/archetype")
        
        assert response.status_code == 404


class TestDestinationSelection:
    """Test strategic destination selection"""
    
    @pytest.mark.asyncio
    async def test_choose_next_destination_exists(self, simulation):
        """Agents should have choose_next_destination method"""
        agent_id = await simulation.spawn_agent(name="Destination Test")
        agent_ref = simulation._agents[agent_id]
        
        # Verify method exists by attempting to call it
        candidates = [
            {"name": "Safe nearby", "distance": 1.0, "risk": 0.2, "novelty": 0.3},
            {"name": "Risky far", "distance": 5.0, "risk": 0.8, "novelty": 0.9},
        ]
        
        # Method should execute without error
        result = await agent_ref.choose_next_destination.remote(candidates)
        assert result is not None
        assert "destination" in result or "choice" in result or result in candidates
