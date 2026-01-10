"""pytest configuration for server tests."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def ray_session():
    """Initialize a small Ray cluster for tests and ensure cleanup."""
    try:
        import ray
    except ImportError:  # pragma: no cover - skip when Ray missing
        pytest.skip("Ray not installed")

    # Prevent reinit collisions across tests
    if ray.is_initialized():
        ray.shutdown()

    object_store_mb = int(os.getenv("RAY_TEST_OBJECT_STORE_MB", "256"))
    num_cpus = int(os.getenv("RAY_TEST_CPUS", "2"))

    # Relax memory monitor threshold a bit for tests and keep resource footprint small
    os.environ.setdefault("RAY_memory_usage_threshold", "0.99")

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=num_cpus,
        object_store_memory=object_store_mb * 1024 * 1024,
        namespace="earthlink-tests",
    )

    yield ray

    # Always shut down Ray after the session
    ray.shutdown()


@pytest.fixture(autouse=True)
def stub_db(monkeypatch):
    """Stub database session maker to avoid real DB dependency during tests.

    Set USE_REAL_DB_TESTS=1 to disable this stub and use the configured database.
    """

    if os.getenv("USE_REAL_DB_TESTS", "0") == "1":
        return

    from src.simulation import runner as simulation_runner
    from src.db import database as db_module

    class _ResultStub:
        def scalar_one_or_none(self):
            return None

        def scalars(self):
            return self

        def all(self):
            return []

    class _SessionStub:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, *args, **kwargs):
            return _ResultStub()

        async def commit(self):
            return None

        def add(self, *args, **kwargs):
            return None

        async def delete(self, *args, **kwargs):
            return None

        async def close(self):
            return None

    def _session_factory():
        return _SessionStub()

    monkeypatch.setattr(simulation_runner, "async_session_maker", _session_factory, raising=False)
    monkeypatch.setattr(db_module, "async_session_maker", _session_factory, raising=False)


@pytest.fixture(autouse=True)
def stub_heavy_agent_init(monkeypatch):
    """Stub Agent.initialize_components to avoid heavy init during Ray tests."""

    from src.agents.core.agent import Agent

    async def _noop_initialize_components(self, memory_config=None, policy_config=None):
        return None

    monkeypatch.setattr(Agent, "initialize_components", _noop_initialize_components, raising=False)


@pytest.fixture
async def test_app():
    """Create test application instance."""
    from src.main import app
    return app
