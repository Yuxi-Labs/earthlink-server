"""Integration tests for API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

from src.main import app


@pytest.mark.asyncio
async def test_create_agent():
    """Test creating an agent via API."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/agents",
            json={"name": "TestAgent"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TestAgent"
        assert "id" in data


@pytest.mark.asyncio
async def test_list_agents():
    """Test listing agents."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create an agent first
        await client.post("/api/v1/agents", json={"name": "Agent1"})
        
        # List agents
        response = await client.get("/api/v1/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0


@pytest.mark.asyncio
async def test_simulation_status():
    """Test getting simulation status."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/simulation/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "stats" in data


@pytest.mark.asyncio
async def test_simulation_start():
    """Test starting simulation."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/simulation/start")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "RUNNING"
