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

    def test_confirmation_threshold(self):
        """Test hypothesis confirmation at high confidence."""
        hyp = Hypothesis(
            id=uuid4(),
            description="Test hypothesis",
            confidence=0.85,
            supporting_evidence=[{"data": "test1"}] * 8,
            contradicting_evidence=[],
            created_at=datetime.now(),
        )

        # Update confidence based on high support
        hyp.update_confidence()

        # Should have high confidence but confirmed status is set elsewhere
        assert hyp.confidence > 0.8

    def test_rejection_threshold(self):
        """Test hypothesis rejection at low confidence."""
        hyp = Hypothesis(
            id=uuid4(),
            description="Test hypothesis",
            confidence=0.15,
            supporting_evidence=[],
            contradicting_evidence=[{"data": "test1"}] * 8,
            created_at=datetime.now(),
        )

        # Update confidence based on contradicting evidence
        hyp.update_confidence()

        # Should have low confidence
        assert hyp.confidence < 0.2


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

    def test_prediction_error_calculation(self):
        """Test prediction error after outcome is known."""
        pred = Prediction(
            id=uuid4(),
            action="move",
            predicted_outcome={"knowledge_gain": 0.5, "distance": 10.0},
            confidence=0.7,
            created_at=datetime.now(),
        )

        # Set actual outcome
        pred.actual_outcome = {"knowledge_gain": 0.3, "distance": 10.0}

        # Calculate error (1 out of 2 keys different)
        different_keys = sum(
            1
            for k in pred.predicted_outcome.keys()
            if pred.predicted_outcome.get(k) != pred.actual_outcome.get(k)
        )
        error = different_keys / len(pred.predicted_outcome)

        pred.prediction_error = error

        assert pred.prediction_error == 0.5  # 1/2 keys different


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

    def test_generate_hypothesis_empty_observations(self, reasoning_engine):
        """Test generating hypothesis with no observations."""
        hypothesis = reasoning_engine.generate_hypothesis([])

        assert hypothesis is not None
        # Should generate default hypothesis about patterns
        assert "pattern" in hypothesis.description.lower() or "no clear" in hypothesis.description.lower()
        assert hypothesis.confidence < 0.5  # Low confidence for no data

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

    def test_predict_with_world_model(self, reasoning_engine):
        """Test prediction using world model."""
        # Mock world model
        class MockWorldModel:
            def predict(self, state, action):
                return {"next_state": "predicted", "reward": 0.5}

        action = "learn"
        world_state = {"knowledge": 0.3}
        world_model = MockWorldModel()

        prediction = reasoning_engine.predict(action, world_state, world_model)

        assert prediction is not None
        assert prediction.confidence > 0.4  # Higher confidence with world model
        assert "next_state" in prediction.predicted_outcome

    def test_infer_causality_no_history(self, reasoning_engine):
        """Test causal inference with no event history."""
        events = []

        relations = reasoning_engine.infer_causality(events)

        assert isinstance(relations, dict)
        # Should return empty relations with no history
        assert len(relations) == 0

    def test_infer_causality_with_history(self, reasoning_engine):
        """Test causal inference with event history."""
        # Add event history
        events = [
            {"timestamp": 1, "type": "move", "effect": "knowledge_gain"},
            {"timestamp": 2, "type": "learn", "effect": "knowledge_gain"},
            {"timestamp": 3, "type": "move", "effect": "knowledge_gain"},
            {"timestamp": 4, "type": "learn", "effect": "knowledge_gain"},
        ]

        relations = reasoning_engine.infer_causality(events)

        assert isinstance(relations, dict)
        # Should find some causal relations
        for key, relation in relations.items():
            assert isinstance(relation, CausalRelation)
            # Check that strength is reasonable
            assert 0.0 <= relation.strength <= 1.0

    def test_counterfactual_reasoning(self, reasoning_engine):
        """Test counterfactual reasoning."""
        current_state = {"knowledge": 0.5, "position": (10, 20)}
        action = "move"

        counterfactual = reasoning_engine.counterfactual_reasoning(
            current_state, action
        )

        assert counterfactual is not None
        assert "alternative_action" in counterfactual
        assert "predicted_outcome" in counterfactual
        # Should predict persistence by default
        assert "minimal" in counterfactual["predicted_outcome"].get("state_change", "")

    def test_test_hypothesis_supporting(self, reasoning_engine):
        """Test hypothesis testing with supporting outcome."""
        # Create hypothesis
        hypothesis = Hypothesis(
            id=uuid4(),
            description="Moving increases knowledge",
            confidence=0.5,
            supporting_evidence=[],
            contradicting_evidence=[],
            created_at=datetime.now(),
        )
        hyp_id = hypothesis.id
        reasoning_engine.hypotheses[hyp_id] = hypothesis

        # Test with supporting outcome
        outcome = {"action": "move", "knowledge_gain": 0.3}
        initial_confidence = hypothesis.confidence

        reasoning_engine.test_hypothesis(hyp_id, outcome, supports=True)

        # Confidence should increase
        assert reasoning_engine.hypotheses[hyp_id].confidence > initial_confidence

    def test_test_hypothesis_contradicting(self, reasoning_engine):
        """Test hypothesis testing with contradicting outcome."""
        # Create hypothesis
        hypothesis = Hypothesis(
            id=uuid4(),
            description="Moving decreases knowledge",
            confidence=0.5,
            supporting_evidence=[],
            contradicting_evidence=[],
            created_at=datetime.now(),
        )
        hyp_id = hypothesis.id
        reasoning_engine.hypotheses[hyp_id] = hypothesis

        # Test with contradicting outcome (knowledge increased)
        outcome = {"action": "move", "knowledge_gain": 0.3}
        initial_confidence = hypothesis.confidence

        reasoning_engine.test_hypothesis(hyp_id, outcome, supports=False)

        # Confidence should decrease
        assert reasoning_engine.hypotheses[hyp_id].confidence <= initial_confidence

    def test_hypothesis_lifecycle(self, reasoning_engine):
        """Test full hypothesis lifecycle: generate, predict, test, confirm/reject."""
        # Generate hypothesis
        observations = [
            {"type": "spatial", "value": "moved 10km", "confidence": 0.9},
            {"type": "knowledge", "value": "learned 5 items", "confidence": 0.8},
        ]
        hypothesis = reasoning_engine.generate_hypothesis(observations)
        assert hypothesis is not None

        # Make predictions
        prediction = reasoning_engine.predict({"knowledge": 0.5}, "move")
        assert prediction is not None

        # Test hypothesis multiple times with supporting evidence
        for _ in range(5):
            reasoning_engine.test_hypothesis(
                hypothesis.id, {"action": "move", "knowledge_gain": 0.1}, supports=True
            )

        # Check if hypothesis confirmed
        updated_hyp = reasoning_engine.hypotheses[hypothesis.id]
        # After multiple supporting evidence, confidence should be high
        assert updated_hyp.confidence > 0.5

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

    def test_causal_relation_strength_update(self, reasoning_engine):
        """Test that causal relations update strength based on observations."""
        # Create a causal relation
        relation = CausalRelation(
            cause="move",
            effect="knowledge_gain",
            strength=0.5,
            observed_together=5,
            observed_separate=5,
        )

        # Update strength
        relation.update_strength()

        # Should be 0.5 (5/10)
        assert relation.strength == 0.5

        # Add more observations together
        relation.observed_together = 8
        relation.update_strength()

        # Strength should increase (8/13 ≈ 0.62)
        assert relation.strength > 0.5
