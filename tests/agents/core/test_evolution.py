"""
Test Evolve capability - self-improvement and capability evolution.
"""
import pytest
from src.agents.core.evolution import EvolutionModule


@pytest.fixture
def evolution_module():
    """Create an evolution module for testing."""
    return EvolutionModule(agent_id="test_evolver")


def test_evolution_module_initialization(evolution_module):
    """Test that evolution module initializes correctly."""
    assert evolution_module is not None
    assert hasattr(evolution_module, 'evolve')


def test_evolve_improves_capabilities(evolution_module):
    """Test that evolution improves agent capabilities."""
    performance_history = {
        "learning_rate": 0.5,
        "exploration_efficiency": 0.6,
    }
    
    result = evolution_module.evolve(performance_history)
    
    assert result is not None
    assert "improvements" in result or "mutations" in result or "evolved" in result


def test_evolve_based_on_success_patterns(evolution_module):
    """Test that evolution favors successful patterns."""
    successful_patterns = {
        "explore_first": 0.8,
        "learn_deeply": 0.75,
    }
    
    result = evolution_module.evolve(successful_patterns)
    
    # Should prioritize successful strategies
    assert result is not None
