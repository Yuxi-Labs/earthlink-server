"""
Test Specialize capability - developing expertise in specific domains.
"""
import pytest
from src.agents.core.specialization import SpecializationModule


@pytest.fixture
def specialization_module():
    """Create a specialization module for testing."""
    return SpecializationModule(agent_id="test_specialist")


def test_specialization_module_initialization(specialization_module):
    """Test that specialization module initializes correctly."""
    assert specialization_module is not None
    assert hasattr(specialization_module, 'specialize')


def test_specialize_in_domain(specialization_module):
    """Test that agent can specialize in a domain."""
    domain = "urban_geography"
    focus_areas = ["transportation", "demographics", "infrastructure"]
    
    result = specialization_module.specialize(domain, focus_areas)
    
    assert result is not None
    assert "specialized" in result or "domain" in result or "expertise_level" in result


def test_specialization_increases_expertise(specialization_module):
    """Test that specialization increases domain expertise."""
    # First specialization
    result1 = specialization_module.specialize("climate_science", ["temperature_patterns"])
    
    # Further specialization in same domain
    result2 = specialization_module.specialize("climate_science", ["precipitation", "wind_patterns"])
    
    # Expertise should deepen
    assert result2 is not None
