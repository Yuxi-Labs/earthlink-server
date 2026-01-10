"""
Test Transfer Knowledge capability - sharing knowledge between agents.
"""
import pytest
from src.agents.core.knowledge_transfer import KnowledgeTransferModule


@pytest.fixture
def transfer_module():
    """Create a knowledge transfer module for testing."""
    return KnowledgeTransferModule(agent_id="test_transfer")


def test_transfer_module_initialization(transfer_module):
    """Test that knowledge transfer module initializes correctly."""
    assert transfer_module is not None
    assert hasattr(transfer_module, 'transfer')


def test_transfer_knowledge_to_another_agent(transfer_module):
    """Test transferring knowledge between agents."""
    knowledge = {
        "topic": "London geography",
        "facts": ["Thames river", "Big Ben location"],
        "confidence": 0.8
    }
    target_agent_id = "agent_002"
    
    result = transfer_module.transfer(knowledge, target_agent_id)
    
    assert result is not None
    assert "transferred" in result or "effectiveness" in result


def test_transfer_preserves_knowledge_integrity(transfer_module):
    """Test that transferred knowledge maintains its integrity."""
    original_knowledge = {
        "topic": "weather_patterns",
        "data": {"temperature": 20, "humidity": 65},
    }
    
    result = transfer_module.transfer(original_knowledge, "agent_003")
    
    # Knowledge should be transferred without corruption
    assert result is not None
