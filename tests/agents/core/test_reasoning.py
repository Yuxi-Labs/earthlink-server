"""Unit tests for ReasoningEngine capability."""

import pytest
from uuid import uuid4
from datetime import datetime
from src.agents.core.reasoning import (
    ReasoningEngine,
    Hypothesis,
    Prediction,
    CausalRelation,
)


class TestHypothesis:
    """Test Hypothesis dataclass and methods."""

    def test_hypothesis_creation(self):
        """Test creating a hypothesis."""
        hyp = Hypothesis(
            id=uuid4(),
            description="Moving increases knowledge gain",
            confidence=0.5,
            supporting_evidence=[],
            contradicting_evidence=[],
            created_at=datetime.now(),
        )
        assert hyp.confidence == 0.5
        assert hyp.confirmed is None
        assert len(hyp.supporting_evidence) == 0

    def test_update_confidence_with_supporting_evidence(self):
        """Test Bayesian confidence update with supporting evidence."""
        hyp = Hypothesis(
            id=uuid4(),
            description="Test hypothesis",
            confidence=0.5,
            supporting_evidence=[],
            contradicting_evidence=[],
            created_at=datetime.now(),
        )

        # Add supporting evidence
        hyp.supporting_evidence.append({"data": "test"})
        hyp.update_confidence()

        # Confidence should increase
        assert hyp.confidence > 0.5
        assert hyp.confidence < 1.0
        assert len(hyp.supporting_evidence) == 1

    def test_update_confidence_with_contradicting_evidence(self):
        """Test Bayesian confidence update with contradicting evidence."""
        hyp = Hypothesis(
            id=uuid4(),
            description="Test hypothesis",
            confidence=0.5,
            supporting_evidence=[],
            contradicting_evidence=[],
            created_at=datetime.now(),
        )

        # Add contradicting evidence
        hyp.contradicting_evidence.append({"data": "test"})
        hyp.update_confidence()

        # Confidence should decrease
        assert hyp.confidence < 0.5
        assert hyp.confidence > 0.0
        assert len(hyp.contradicting_evidence) == 1


class TestPrediction:
    """Test Prediction dataclass."""

    def test_prediction_creation(self):
        """Test creating a prediction."""
        pred = Prediction(
            id=uuid4(),
            action="move",
            predicted_outcome={"knowledge_gain": 0.5},
            confidence=0.7,
            created_at=datetime.now(),
        )
        assert pred.action == "move"
        assert pred.confidence == 0.7
        assert pred.actual_outcome is None
        assert pred.prediction_error is None


class TestCausalRelation:
    """Test CausalRelation dataclass."""

    def test_causal_relation_creation(self):
        """Test creating a causal relation."""
        relation = CausalRelation(
            cause="move",
            effect="knowledge_gain",
            strength=0.6,
            evidence_count=5,
        )
        assert relation.cause == "move"
        assert relation.effect == "knowledge_gain"
        assert relation.strength == 0.6
        assert relation.evidence_count == 5


class TestReasoningEngine:
    """Test ReasoningEngine core functionality."""

    @pytest.fixture
    def reasoning_engine(self):
        """Create a ReasoningEngine instance for testing."""
        return ReasoningEngine(agent_id=uuid4())

    def test_engine_initialization(self):
        """Test ReasoningEngine initializes correctly."""
        engine = ReasoningEngine(agent_id=uuid4())
        assert len(engine.hypotheses) == 0
        assert len(engine.predictions) == 0
        assert len(engine.causal_relations) == 0
        assert engine.reasoning_stats["hypotheses_generated"] == 0

    def test_generate_hypothesis_from_observations(self, reasoning_engine):
        """Test generating hypotheses from observations."""
        observations = [
            {"type": "spatial", "value": "location: (10, 20)", "confidence": 0.8},
            {"type": "social", "value": "nearby_agents: []", "confidence": 0.9},
            {"type": "temporal", "value": "time_step: 100", "confidence": 1.0},
        ]

        hypothesis = reasoning_engine.generate_hypothesis(observations)

        assert hypothesis is not None
        assert isinstance(hypothesis, Hypothesis)
        assert hypothesis.confidence > 0.0
        assert hypothesis.confidence < 1.0
        assert len(reasoning_engine.hypotheses) == 1
        assert reasoning_engine.reasoning_stats["hypotheses_generated"] == 1

    def test_predict_simple_persistence(self, reasoning_engine):
        """Test prediction using persistence assumption."""
        world_state = {
            "position": (10, 20),
            "knowledge": 0.5,
        }
        action = "move"

        prediction = reasoning_engine.predict(world_state, action)

        assert prediction is not None
        assert isinstance(prediction, Prediction)
        assert prediction.action == action
        assert prediction.confidence > 0.0
        assert "state_change" in prediction.predicted_outcome
        assert len(reasoning_engine.predictions) == 1
        assert reasoning_engine.reasoning_stats["predictions_made"] == 1

    def test_statistics_tracking(self, reasoning_engine):
        """Test that statistics are properly tracked."""
        # Generate hypotheses
        reasoning_engine.generate_hypothesis([{"type": "test", "value": "data", "confidence": 0.5}])
        reasoning_engine.generate_hypothesis([{"type": "test", "value": "data2", "confidence": 0.6}])

        # Make predictions
        reasoning_engine.predict({"knowledge": 0.5}, "move")
        reasoning_engine.predict({"knowledge": 0.3}, "learn")

        # Check statistics
        assert reasoning_engine.reasoning_stats["hypotheses_generated"] == 2
        assert reasoning_engine.reasoning_stats["predictions_made"] == 2
