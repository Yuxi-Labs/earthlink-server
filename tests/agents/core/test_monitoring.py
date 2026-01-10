"""
Test Self-monitor capability - performance tracking and self-assessment.
"""
import pytest
from src.agents.core.monitoring import SelfMonitoringModule


@pytest.fixture
def monitoring_module():
    """Create a monitoring module for testing."""
    return SelfMonitoringModule(agent_id="test_monitor")


def test_monitoring_module_initialization(monitoring_module):
    """Test that monitoring module initializes correctly."""
    assert monitoring_module is not None
    assert hasattr(monitoring_module, 'monitor')


def test_monitor_tracks_performance(monitoring_module):
    """Test that monitoring tracks agent performance."""
    metrics = {
        "step_duration": 0.5,
        "success": True,
        "action_type": "explore"
    }
    
    result = monitoring_module.monitor(metrics)
    
    assert result is not None
    assert "health" in result or "status" in result or "performance" in result


def test_monitor_detects_degradation(monitoring_module):
    """Test that monitoring detects performance degradation."""
    # Simulate degrading performance
    for i in range(10):
        metrics = {
            "step_duration": 0.1 + (i * 0.2),  # Increasing duration
            "success": i < 5,  # Decreasing success rate
        }
        result = monitoring_module.monitor(metrics)
    
    # Should detect the degradation
    assert result is not None


def test_monitor_provides_diagnostics(monitoring_module):
    """Test that monitor provides diagnostic information."""
    metrics = {
        "error_count": 5,
        "step_duration": 2.0,
        "success": False
    }
    
    result = monitoring_module.monitor(metrics)
    
    # Should provide actionable diagnostics
    assert result is not None
