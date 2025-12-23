"""Basic API tests."""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_agents() -> None:
    """Test listing agents."""
    response = client.get("/api/v1/agents/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_worlds() -> None:
    """Test listing target worlds."""
    response = client.get("/api/v1/worlds/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
