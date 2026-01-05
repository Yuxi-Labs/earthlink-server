"""Integration tests for the cognitive flow: PERCEIVE → REASON → DECIDE."""

import pytest
import asyncio
from uuid import UUID, uuid4

from src.agents.core.perception import PerceptionModule, AttentionFocus, PerceptionModality
from src.agents.core.reasoning import ReasoningEngine
from src.agents.core.decision import DecisionModule, DecisionStrategy, Action, Goal


@pytest.fixture
def agent_id():
    """Test agent ID."""
    return str(uuid4())


@pytest.fixture
def perception_module(agent_id):
    """Create PerceptionModule instance."""
    return PerceptionModule(agent_id)


@pytest.fixture
def reasoning_engine(agent_id):
    """Create ReasoningEngine instance."""
    return ReasoningEngine(agent_id)


@pytest.fixture
def decision_module(agent_id):
    """Create DecisionModule instance."""
    return DecisionModule(agent_id)


@pytest.fixture
def mock_environment():
    """Mock environment for testing."""
    return {
        "position": (10.5, 20.3),
        "knowledge_items": 15,
        "nearby_agents": [
            {"id": str(uuid4()), "distance": 5.0},
            {"id": str(uuid4()), "distance": 8.2},
        ],
        "nearby_places": [
            {"name": "Library", "type": "education", "distance": 2.0},
        ],
        "current_goal": "learn_knowledge",
        "timestamp": 1000,
        "agent_state": {
            "status": "active",
            "lifecycle": "exploring",
            "current_goal_id": str(uuid4()),
            "metrics": {"knowledge": 15},
        },
    }


class TestPerceptionToReasoning:
    """Test perception → reasoning integration."""
    
    @pytest.mark.asyncio
    async def test_basic_perception_flow(self, perception_module, mock_environment):
        """Test basic perception with attention focus."""
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL, PerceptionModality.KNOWLEDGE],
            keywords=["learn", "knowledge"],
            priority=0.8,
        )
        
        observations = await perception_module.perceive(mock_environment, focus)
        
        assert observations is not None
        assert len(observations) > 0
        # Should have spatial and knowledge modalities
        modalities = {obs.modality for obs in observations}
        assert PerceptionModality.SPATIAL in modalities
        assert PerceptionModality.KNOWLEDGE in modalities
    
    @pytest.mark.asyncio
    async def test_perception_feeds_reasoning(self, perception_module, reasoning_engine, mock_environment):
        """Test that perception output can be used for reasoning."""
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL, PerceptionModality.SOCIAL],
            keywords=["agent", "location"],
            priority=0.7,
        )
        
        observations = await perception_module.perceive(mock_environment, focus)
        
        # Convert to dict format expected by reasoning
        obs_dicts = [
            {
                "modality": str(obs.modality),
                "data": obs.data,
                "confidence": obs.confidence,
            }
            for obs in observations
        ]
        
        # Generate hypothesis from observations
        hypothesis = reasoning_engine.generate_hypothesis(obs_dicts)
        
        assert hypothesis is not None
        assert hypothesis.confidence > 0
        assert len(hypothesis.supporting_evidence) >= 0


class TestReasoningToDecision:
    """Test reasoning → decision integration."""
    
    def test_predictions_feed_decisions(self, reasoning_engine, decision_module):
        """Test that predictions influence decision-making."""
        state = {"knowledge": 50, "position": (10, 20)}
        
        # Generate predictions
        predictions = {}
        action_types = ["learn", "move", "explore"]
        for action_type in action_types:
            pred = reasoning_engine.predict(state, action_type)
            predictions[action_type] = {
                "confidence": pred.confidence,
                "predicted_outcome": pred.predicted_outcome,
            }
        
        # Make decision with predictions
        actions = [Action(type=atype, params={}) for atype in action_types]
        goals = [Goal(id="knowledge", description="Maximize knowledge", priority=0.9)]
        
        decision = decision_module.decide(
            available_actions=actions,
            goals=goals,
            world_state=state,
            predictions=predictions,
        )
        
        assert decision is not None
        assert decision.action in actions
        assert decision.confidence > 0


class TestFullCognitiveLoop:
    """Test complete PERCEIVE → REASON → DECIDE cycle."""
    
    @pytest.mark.asyncio
    async def test_complete_cycle(
        self, perception_module, reasoning_engine, decision_module, mock_environment
    ):
        """Test full cognitive cycle end-to-end."""
        # Step 1: PERCEIVE
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL, PerceptionModality.KNOWLEDGE],
            keywords=["learn", "knowledge"],
            priority=0.8,
        )
        
        perception_results = await perception_module.perceive(mock_environment, focus)
        assert len(perception_results) > 0
        
        # Step 2: REASON - Convert perception to observations and generate hypothesis
        obs_dicts = [
            {
                "modality": str(obs.modality),
                "data": obs.data,
                "confidence": obs.confidence,
            }
            for obs in perception_results
        ]
        
        hypothesis = reasoning_engine.generate_hypothesis(obs_dicts)
        assert hypothesis is not None
        
        # Step 3: REASON - Make predictions
        state = {"knowledge": mock_environment.get("knowledge_items", 0)}
        predictions = {}
        for action_type in ["learn", "move"]:
            pred = reasoning_engine.predict(state, action_type)
            predictions[action_type] = {
                "confidence": pred.confidence,
                "predicted_outcome": pred.predicted_outcome,
            }
        
        # Step 4: DECIDE
        actions = [
            Action(type="learn", params={}),
            Action(type="move", params={}),
        ]
        goals = [Goal(id="knowledge", description="Maximize knowledge", priority=0.9)]
        
        decision = decision_module.decide(
            available_actions=actions,
            goals=goals,
            world_state=state,
            predictions=predictions,
        )
        
        assert decision is not None
        assert decision.action.type in ["learn", "move"]
        assert decision.confidence > 0
        
        # Verify rationale was generated
        assert len(decision.rationale) > 0


class TestStatistics:
    """Test that statistics are tracked correctly."""
    
    def test_reasoning_statistics(self, reasoning_engine):
        """Test reasoning engine tracks statistics."""
        state = {"knowledge": 10}
        
        # Make multiple predictions
        for i in range(5):
            reasoning_engine.predict(state, f"action_{i}")
        
        # Check statistics
        assert reasoning_engine.reasoning_stats["predictions_made"] == 5
    
    def test_decision_statistics(self, decision_module):
        """Test decision module tracks statistics."""
        actions = [Action(type="test", params={})]
        goals = [Goal(id="test", description="Test", priority=1.0)]
        state = {"value": 1}
        
        # Make multiple decisions
        for i in range(3):
            decision_module.decide(
                available_actions=actions,
                goals=goals,
                world_state=state,
            )
        
        # Check statistics
        assert decision_module.stats["decisions_made"] == 3
