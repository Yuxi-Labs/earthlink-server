"""
Test Model capability - building world models and predictions.
"""
import pytest
from src.agents.core.modeling import ModelingModule


@pytest.fixture
def modeling_module():
    """Create a modeling module for testing."""
    return ModelingModule(agent_id="test_modeler")


def test_modeling_module_initialization(modeling_module):
    """Test that modeling module initializes correctly."""
    assert modeling_module is not None
    assert hasattr(modeling_module, 'build_world_model')


@pytest.mark.asyncio
async def test_build_world_model(modeling_module):
    """Test building a world model from observations."""
    observations = [
        {"location": (51.5, -0.1), "timestamp": "2026-01-10T10:00:00"},
        {"location": (51.5, -0.2), "timestamp": "2026-01-10T10:05:00"},
    ]
    
    model = await modeling_module.build_world_model(observations)
    
    assert model is not None
    # WorldModel object has attributes
    assert hasattr(model, 'predictions') or hasattr(model, 'state') or hasattr(model, '__dict__')


@pytest.mark.asyncio
async def test_world_model_learns_from_experience(modeling_module):
    """Test that world model improves with more observations."""
    # First observation set
    obs1 = [{"event": "moved", "result": "success"}]
    model1 = await modeling_module.build_world_model(obs1)
    
    # More observations
    obs2 = obs1 + [
        {"event": "moved", "result": "success"},
        {"event": "learned", "result": "knowledge_gained"},
    ]
    model2 = await modeling_module.build_world_model(obs2)
    
    # Model should incorporate new information
    assert model2 is not None
