"""Integration tests for PERCEIVE→REASON→DECIDE cognitive flow."""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock
from src.agents.core.perception import PerceptionModule, AttentionFocus
from src.agents.core.reasoning import ReasoningEngine
from src.agents.core.decision import DecisionModule, Action, Goal, DecisionStrategy


class TestCognitiveFlowIntegration:
    """Test the full PERCEIVE→REASON→DECIDE pipeline."""

    @pytest.fixture
    def agent_id(self):
        """Create a test agent ID."""
        return uuid4()

    @pytest.fixture
    def perception_module(self, agent_id):
        """Create PerceptionModule instance."""
        return PerceptionModule(agent_id=agent_id)

    @pytest.fixture
    def reasoning_engine(self, agent_id):
        """Create ReasoningEngine instance."""
        return ReasoningEngine(agent_id=agent_id)

    @pytest.fixture
    def decision_module(self, agent_id):
        """Create DecisionModule instance."""
        return DecisionModule(agent_id=agent_id)

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment with relevant data."""
        return {
            "position": (10.5, 20.3),
            "nearby_agents": [
                {"id": str(uuid4()), "distance": 5.0},
                {"id": str(uuid4()), "distance": 8.0},
            ],
            "nearby_places": [
                {"name": "Library", "type": "education", "distance": 2.0},
            ],
            "current_goal": "learn_knowledge",
            "knowledge_items": 15,
            "time_step": 42,
        }

    def test_perceive_to_reason_flow(
        self, perception_module, reasoning_engine, mock_environment
    ):
        """Test that perception outputs feed correctly into reasoning."""
        from src.agents.core.perception import PerceptionModality
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL, PerceptionModality.SOCIAL, PerceptionModality.KNOWLEDGE],
            keywords=["learn", "knowledge"],
            priority=0.8,
        )
        
        observations = perception_module.perceive(mock_environment, focus)

        assert observations is not None
        assert len(observations) > 0

        hypothesis = reasoning_engine.generate_hypothesis(observations)

        assert hypothesis is not None
        assert hypothesis.confidence > 0.0
        assert len(hypothesis.description) > 0
        assert reasoning_engine.reasoning_stats["hypotheses_generated"] == 1

    def test_reason_to_decide_flow(
        self, reasoning_engine, decision_module, mock_environment
    ):
        """Test that reasoning outputs feed correctly into decision-making."""
        state = {"knowledge": 15, "position": (10.5, 20.3)}
        
        predictions = {}
        for action_type in ["move", "learn", "explore"]:
            pred = reasoning_engine.predict(state, action_type)
            predictions[action_type] = pred

        assert len(predictions) == 3
        assert all(p.confidence > 0 for p in predictions.values())

        actions = [
            Action(type="move", params={}),
            Action(type="learn", params={}),
            Action(type="explore", params={}),
        ]
        
        goals = [
            Goal(id="max_knowledge", description="Maximize knowledge", priority=0.8),
        ]

        world_state = {
            "knowledge": state["knowledge"],
            "position": state["position"],
        }

        hypotheses = list(reasoning_engine.hypotheses.values())
        
        decision = decision_module.decide(
            actions=actions,
            goals=goals,
            world_state=world_state,
            predictions=predictions,
            hypotheses=hypotheses,
            strategy=DecisionStrategy.MAXIMIZE_UTILITY,
        )

        assert decision is not None
        assert decision.chosen_action in ["move", "learn", "explore"]
        assert decision.confidence > 0.0
        assert len(decision.rationale) > 0

    def test_decision_strategy_consistency(self, decision_module, reasoning_engine):
        """Test that different strategies produce consistent results."""
        state = {"knowledge": 50}
        predictions = {
            "action1": reasoning_engine.predict(state, "action1"),
            "action2": reasoning_engine.predict(state, "action2"),
        }

        actions = [
            Action(type="action1", params={}),
            Action(type="action2", params={}),
        ]
        
        goals = [Goal(id="test", description="Test goal", priority=1.0)]

        strategies = [
            DecisionStrategy.MAXIMIZE_UTILITY,
            DecisionStrategy.MINIMIZE_RISK,
            DecisionStrategy.BALANCED,
            DecisionStrategy.SATISFICE,
        ]

        decisions = []
        for strategy in strategies:
            decision = decision_module.decide(
                actions=actions,
                goals=goals,
                world_state=state,
                predictions=predictions,
                hypotheses=[],
                strategy=strategy,
            )
            decisions.append(decision)
            assert decision is not None
            assert decision.chosen_action in ["action1", "action2"]

        assert len(decisions) == len(strategies)
