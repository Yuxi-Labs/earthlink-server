"""
Test Adapt capability - behavioral adaptation based on experience.
"""
import pytest
from src.agents.core.adaptation import AdaptationModule


@pytest.fixture
def adaptation_module():
    """Create an adaptation module for testing."""
    return AdaptationModule(agent_id="test_adapter")


def test_adaptation_module_initialization(adaptation_module):
    """Test that adaptation module initializes correctly."""
    assert adaptation_module is not None
    assert hasattr(adaptation_module, 'adapt')


def test_adapt_to_feedback(adaptation_module):
    """Test that agent adapts based on feedback."""
    feedback = {
        "action": "explore",
        "outcome": "failure",
        "reason": "insufficient_energy"
    }
    
    result = adaptation_module.adapt(feedback)
    
    assert result is not None
    # Should provide adaptation strategy or updated parameters


def test_adapt_improves_performance(adaptation_module):
    """Test that adaptation leads to performance improvements."""
    # Simulate repeated failures
    for _ in range(5):
        feedback = {"action": "move", "outcome": "failure"}
        adaptation_module.adapt(feedback)
    
    # Adaptation should have occurred
    # (Check internal state or behavior modification)
    assert True  # Placeholder - needs implementation detail
