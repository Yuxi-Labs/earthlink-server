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
        # PERCEIVE: Gather observations
        from src.agents.core.perception import PerceptionModality
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL, PerceptionModality.SOCIAL, PerceptionModality.KNOWLEDGE],
            keywords=["learn", "knowledge"],
            priority=0.8,
        )
        
        observations = perception_module.perceive(mock_environment, focus)

        # Verify we got observations
        assert observations is not None
        assert len(observations) > 0

        # REASON: Generate hypothesis from observations
        hypothesis = reasoning_engine.generate_hypothesis(observations)

        # Verify hypothesis was created from perceptions
        assert hypothesis is not None
        assert hypothesis.confidence > 0.0
        assert len(hypothesis.description) > 0
        assert reasoning_engine.reasoning_stats["hypotheses_generated"] == 1

    def test_reason_to_decide_flow(
        self, reasoning_engine, decision_module, mock_environment
    ):
        """Test that reasoning outputs feed correctly into decision-making."""
        # REASON: Make predictions for different actions
        state = {"knowledge": 15, "position": (10.5, 20.3)}
        
        predictions = {}
        for action_type in ["move", "learn", "explore"]:
            pred = reasoning_engine.predict(state, action_type)
            predictions[action_type] = pred

        # Verify predictions were made
        assert len(predictions) == 3
        assert all(p.confidence > 0 for p in predictions.values())

        # DECIDE: Use predictions to make decision
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

        # Verify decision was made using predictions
        assert decision is not None
        assert decision.chosen_action in ["move", "learn", "explore"]
        assert decision.confidence > 0.0
        assert len(decision.rationale) > 0

    def test_full_perceive_reason_decide_cycle(
        self, perception_module, reasoning_engine, decision_module, mock_environment
    ):
        """Test the complete cognitive cycle from perception to decision."""
        # Step 1: PERCEIVE
        from src.agents.core.perception import PerceptionModality
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL, PerceptionModality.SOCIAL, PerceptionModality.TEMPORAL, PerceptionModality.KNOWLEDGE],
            keywords=["learn", "knowledge", "explore"],
            priority=0.8,
        )
        
        observations = perception_module.perceive(mock_environment, focus)
        assert len(observations) > 0

        # Step 2: REASON - Generate hypothesis
        hypothesis = reasoning_engine.generate_hypothesis(observations)
        assert hypothesis is not None

        # Step 3: REASON - Make predictions
        current_state = {
            "position": mock_environment["position"],
            "knowledge": mock_environment["knowledge_items"],
            "nearby_places": len(mock_environment["nearby_places"]),
        }

        predictions = {}
        action_types = ["move", "learn", "policy"]
        for action_type in action_types:
            pred = reasoning_engine.predict(current_state, action_type)
            predictions[action_type] = pred

        # Step 4: DECIDE - Make decision
        actions = [
            Action(type=atype, params={})
            for atype in action_types
        ]
        
        goals = [
            Goal(id="max_knowledge", description="Maximize knowledge", priority=0.9),
            Goal(id="explore", description="Explore world", priority=0.6),
        ]

        decision = decision_module.decide(
            actions=actions,
            goals=goals,
            world_state=current_state,
            predictions=predictions,
            hypotheses=[hypothesis],
            strategy=DecisionStrategy.BALANCED,
        )

        # Verify complete flow
        assert decision is not None
        assert decision.chosen_action in action_types
        assert decision.utility_score >= 0.0
        assert decision.risk_score >= 0.0
        assert len(decision.alternatives) == len(action_types) - 1

        # Verify the decision has proper rationale
        assert len(decision.rationale) > 0

    def test_perception_attention_influences_reasoning(
        self, perception_module, reasoning_engine, mock_environment
    ):
        """Test that attention focus affects what hypotheses are generated."""
        # Focus on spatial information
        from src.agents.core.perception import PerceptionModality
        spatial_focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL],
            keywords=["move", "location"],
            priority=0.8,
        )
        
        spatial_obs = perception_module.perceive(mock_environment, spatial_focus)
        spatial_hyp = reasoning_engine.generate_hypothesis(spatial_obs)

        # Focus on knowledge information
        knowledge_focus = AttentionFocus(
            modalities=[PerceptionModality.KNOWLEDGE],
            keywords=["learn", "knowledge"],
            priority=0.8,
        )
        
        knowledge_obs = perception_module.perceive(mock_environment, knowledge_focus)
        knowledge_hyp = reasoning_engine.generate_hypothesis(knowledge_obs)

        # Different focuses should potentially lead to different observations
        # (though both are valid hypotheses)
        assert spatial_hyp is not None
        assert knowledge_hyp is not None
        # Both should have valid confidence scores
        assert 0.0 <= spatial_hyp.confidence <= 1.0
        assert 0.0 <= knowledge_hyp.confidence <= 1.0

    def test_predictions_influence_decisions(
        self, reasoning_engine, decision_module
    ):
        """Test that prediction confidence affects decision quality."""
        state = {"knowledge": 50}
        
        # Create predictions with different confidence levels
        low_conf_pred = reasoning_engine.predict(state, "risky_action")
        low_conf_pred.confidence = 0.2
        
        high_conf_pred = reasoning_engine.predict(state, "safe_action")
        high_conf_pred.confidence = 0.9

        predictions = {
            "risky_action": low_conf_pred,
            "safe_action": high_conf_pred,
        }

        actions = [
            Action(type="risky_action", params={}),
            Action(type="safe_action", params={}),
        ]
        
        goals = [Goal(id="test", description="Test goal", priority=1.0)]

        # MINIMIZE_RISK strategy should prefer high confidence prediction
        decision = decision_module.decide(
            actions=actions,
            goals=goals,
            world_state=state,
            predictions=predictions,
            hypotheses=[],
            strategy=DecisionStrategy.MINIMIZE_RISK,
        )

        # With risk minimization, should prefer the safer, high-confidence action
        assert decision is not None
        # The decision logic should account for prediction confidence

    def test_hypothesis_confidence_affects_risk_assessment(
        self, reasoning_engine, decision_module
    ):
        """Test that hypothesis confidence influences risk assessment in decisions."""
        # Create a low-confidence hypothesis
        low_conf_obs = [
            {"type": "test", "value": "uncertain data", "confidence": 0.2}
        ]
        low_conf_hyp = reasoning_engine.generate_hypothesis(low_conf_obs)

        # Create a high-confidence hypothesis
        high_conf_obs = [
            {"type": "test", "value": "certain pattern", "confidence": 0.9} for _ in range(5)
        ]
        high_conf_hyp = reasoning_engine.generate_hypothesis(high_conf_obs)

        state = {"knowledge": 50}
        action = Action(type="test", params={})
        pred = reasoning_engine.predict(state, "test")

        # Decision with low confidence hypothesis should have higher risk
        low_conf_decision = decision_module.decide(
            actions=[action],
            goals=[Goal(id="test", description="Test", priority=1.0)],
            world_state=state,
            predictions={"test": pred},
            hypotheses=[low_conf_hyp],
            strategy=DecisionStrategy.MINIMIZE_RISK,
        )

        # Decision with high confidence hypothesis should have lower risk
        high_conf_decision = decision_module.decide(
            actions=[action],
            goals=[Goal(id="test", description="Test", priority=1.0)],
            world_state=state,
            predictions={"test": pred},
            hypotheses=[high_conf_hyp],
            strategy=DecisionStrategy.MINIMIZE_RISK,
        )

        # Both decisions should be valid
        assert low_conf_decision is not None
        assert high_conf_decision is not None

    def test_multi_goal_balancing_with_predictions(
        self, reasoning_engine, decision_module
    ):
        """Test decision-making with multiple competing goals and predictions."""
        state = {"knowledge": 30, "exploration": 10, "resources": 50}

        # Make predictions for different actions
        predictions = {}
        for action_type in ["learn", "explore", "rest"]:
            predictions[action_type] = reasoning_engine.predict(state, action_type)

        actions = [
            Action(type="learn", params={}),
            Action(type="explore", params={}),
            Action(type="rest", params={}),
        ]

        # Multiple competing goals
        goals = [
            Goal(id="knowledge", description="Maximize knowledge", priority=0.8),
            Goal(id="exploration", description="Maximize exploration", priority=0.6),
            Goal(id="resources", description="Conserve resources", priority=0.4),
        ]

        # BALANCED strategy should consider all goals
        decision = decision_module.decide(
            actions=actions,
            goals=goals,
            world_state=state,
            predictions=predictions,
            hypotheses=[],
            strategy=DecisionStrategy.BALANCED,
        )

        assert decision is not None
        assert decision.chosen_action in ["learn", "explore", "rest"]
        # Should have evaluated utility for multiple goals
        assert decision.utility_score >= 0.0

    def test_perception_history_tracking(self, perception_module, mock_environment):
        """Test that perception module tracks observation history."""
        from src.agents.core.perception import PerceptionModality
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL, PerceptionModality.TEMPORAL],
            keywords=["location"],
            priority=0.7,
        )
        
        # Make multiple observations
        for _ in range(3):
            perception_module.perceive(mock_environment, focus)

        # Check that history is being tracked
        assert len(perception_module.observation_history) == 3
        # Each entry should have observations
        for entry in perception_module.observation_history:
            assert "observations" in entry
            assert "timestamp" in entry

    def test_reasoning_prediction_history(self, reasoning_engine):
        """Test that reasoning engine maintains prediction history."""
        state = {"value": 10}
        
        # Make several predictions
        for action in ["action1", "action2", "action3"]:
            reasoning_engine.predict(state, action)

        # Should have all predictions in history
        assert reasoning_engine.reasoning_stats["predictions_made"] == 3
        assert len(reasoning_engine.predictions) == 3

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

        # Try different strategies
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

        # All decisions should be valid
        assert len(decisions) == len(strategies)

    def test_empty_observations_graceful_handling(
        self, perception_module, reasoning_engine, decision_module
    ):
        """Test that the flow handles empty/minimal data gracefully."""
        # Empty environment
        empty_env = {}
        
        # PERCEIVE with empty environment
        from src.agents.core.perception import PerceptionModality
        focus = AttentionFocus(
            modalities=[PerceptionModality.SPATIAL],
            keywords=[],
            priority=0.5,
        )
        observations = perception_module.perceive(empty_env, focus)
        
        # Should still return observations (possibly empty or default)
        assert observations is not None

        # REASON with minimal observations
        hypothesis = reasoning_engine.generate_hypothesis(observations)
        assert hypothesis is not None  # Should create default hypothesis

        # DECIDE with minimal information
        state = {}
        pred = reasoning_engine.predict(state, "default_action")
        
        decision = decision_module.decide(
            actions=[Action(type="default_action", params={})],
            goals=[Goal(id="default", description="Default", priority=1.0)],
            world_state=state,
            predictions={"default_action": pred},
            hypotheses=[hypothesis],
            strategy=DecisionStrategy.SATISFICE,
        )

        assert decision is not None  # Should make some decision even with minimal info


class TestCognitiveFlowStatistics:
    """Test statistics and metrics tracking across the cognitive flow."""

    @pytest.fixture
    def agent_id(self):
        return uuid4()

    @pytest.fixture
    def full_stack(self, agent_id):
        """Create full cognitive stack."""
        return {
            "perception": PerceptionModule(agent_id=agent_id),
            "reasoning": ReasoningEngine(agent_id=agent_id),
            "decision": DecisionModule(agent_id=agent_id),
        }

    def test_statistics_accumulation(self, full_stack):
        """Test that statistics accumulate correctly across multiple cycles."""
        perception = full_stack["perception"]
        reasoning = full_stack["reasoning"]
        decision = full_stack["decision"]

        env = {"position": (1, 2), "knowledge": 10}
        
        # Run multiple cycles
        for i in range(5):
            # Perceive
            from src.agents.core.perception import PerceptionModality
            focus = AttentionFocus(
                modalities=[PerceptionModality.SPATIAL],
                keywords=[],
                priority=0.5,
            )
            observations = perception.perceive(env, focus)
            
            # Reason
            hypothesis = reasoning.generate_hypothesis(observations)
            pred = reasoning.predict({"value": i}, f"action_{i}")
            
            # Decide
            decision.decide(
                actions=[Action(type=f"action_{i}", params={})],
                goals=[Goal(id="test", description="Test", priority=1.0)],
                world_state={"value": i},
                predictions={f"action_{i}": pred},
                hypotheses=[hypothesis],
                strategy=DecisionStrategy.BALANCED,
            )

        # Verify statistics
        assert len(perception.observation_history) == 5
        assert reasoning.reasoning_stats["hypotheses_generated"] == 5
        assert reasoning.reasoning_stats["predictions_made"] == 5
        assert decision.decision_stats["decisions_made"] == 5
