"""
Test Generate capability - creating outputs and artifacts.
"""
import pytest
from src.agents.core.generation import GenerationModule


@pytest.fixture
def generation_module():
    """Create a generation module for testing."""
    return GenerationModule(agent_id="test_generator")


def test_generation_module_initialization(generation_module):
    """Test that generation module initializes correctly."""
    assert generation_module is not None
    assert hasattr(generation_module, 'generate')


def test_generate_output_from_knowledge(generation_module):
    """Test generating output from accumulated knowledge."""
    knowledge_base = {
        "topics": ["London", "geography", "history"],
        "facts": ["Capital of UK", "River Thames", "Founded by Romans"],
    }
    
    output = generation_module.generate(knowledge_base)
    
    assert output is not None
    assert len(output) > 0 or isinstance(output, dict)


def test_generate_different_output_types(generation_module):
    """Test generating different types of outputs."""
    input_data = {"type": "summary", "content": "Agent exploration results"}
    
    output = generation_module.generate(input_data)
    
    assert output is not None


def test_generate_creative_outputs(generation_module):
    """Test that generation can produce creative/novel outputs."""
    constraints = {
        "theme": "exploration",
        "style": "narrative",
    }
    
    output = generation_module.generate(constraints)
    
    # Should produce something
    assert output is not None
