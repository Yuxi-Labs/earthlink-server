"""Integration test configuration."""

import pytest


@pytest.fixture
async def test_app():
    """Create test application instance."""
    from src.main import app
    return app
