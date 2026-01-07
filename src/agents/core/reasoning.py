"""
Reasoning engine for agents - hypothesis generation, causal inference, predictions.

Reason capability: Generate hypotheses from observations, predict outcomes,
infer causal relationships, perform counterfactual reasoning.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import torch


class ReasoningType(str, Enum):
    """Types of reasoning agents can perform."""
    
    HYPOTHESIS = "hypothesis"  # Generate hypothesis from observations
    PREDICTION = "prediction"  # Predict outcome of action
    CAUSAL = "causal"  # Infer cause-effect relationships
    COUNTERFACTUAL = "counterfactual"  # What-if reasoning
    INFERENCE = "inference"  # Logical inference chains


@dataclass
class Hypothesis:
    """A hypothesis generated from observations."""
    
    id: UUID = field(default_factory=uuid4)
    description: str = ""
    confidence: float = 0.5  # 0-1, how confident in this hypothesis
    supporting_evidence: list[dict] = field(default_factory=list)
    contradicting_evidence: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tested: bool = False
    confirmed: bool | None = None  # True=confirmed, False=rejected, None=unknown
    
    def update_confidence(self):
        """Update confidence based on evidence."""
        support = len(self.supporting_evidence)
        contradict = len(self.contradicting_evidence)
        total = support + contradict
        
        if total == 0:
            self.confidence = 0.5
        else:
            # Bayesian update: more support = higher confidence
            self.confidence = (support + 1) / (total + 2)  # +1/+2 for smoothing
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self.id),
            "description": self.description,
            "confidence": self.confidence,
            "supporting_evidence_count": len(self.supporting_evidence),
            "contradicting_evidence_count": len(self.contradicting_evidence),
            "tested": self.tested,
            "confirmed": self.confirmed,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Prediction:
    """A prediction about future state or outcome."""
    
    id: UUID = field(default_factory=uuid4)
    action: str = ""  # Action being predicted
    predicted_outcome: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    actual_outcome: dict[str, Any] | None = None
    prediction_error: float | None = None  # How wrong was prediction
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self.id),
            "action": self.action,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "prediction_error": self.prediction_error,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CausalRelation:
    """A cause-effect relationship inferred from observations."""
    
    cause: str
    effect: str
    strength: float = 0.5  # 0-1, how strong is the relationship
    evidence_count: int = 0
    observed_together: int = 0
    observed_separate: int = 0
    
    def update_strength(self):
        """Update strength based on co-occurrence."""
        total = self.observed_together + self.observed_separate
        if total == 0:
            self.strength = 0.5
        else:
            # P(effect|cause) vs P(effect|~cause)
            self.strength = self.observed_together / total


class ReasoningEngine:
    """
    Generate hypotheses, predictions, and causal inferences.
    
    Capabilities:
    - Hypothesis generation from observations
    - Outcome prediction before actions
    - Causal relationship inference
    - Counterfactual reasoning (what-if)
    - Logical inference chains
    """
    
    def __init__(
        self,
        agent_id: UUID,
        device: str = "cpu",
        reasoning_depth: int = 3,
        confidence_threshold: float = 0.6,
    ):
        self.agent_id = agent_id
        self.device = device
        self.reasoning_depth = reasoning_depth
        self.confidence_threshold = confidence_threshold
        
        # Active hypotheses
        self.hypotheses: dict[UUID, Hypothesis] = {}
        
        # Predictions
        self.predictions: dict[UUID, Prediction] = {}
        self.prediction_history: list[Prediction] = []
        
        # Causal graph
        self.causal_relations: dict[tuple[str, str], CausalRelation] = {}
        
        # Reasoning statistics
        self.reasoning_stats = {
            "hypotheses_generated": 0,
            "hypotheses_confirmed": 0,
            "hypotheses_rejected": 0,
            "predictions_made": 0,
            "average_prediction_error": 0.5,
            "causal_relations_discovered": 0,
        }
    
    def reason(
        self,
        observations: list[dict],
        context: dict[str, Any] | None = None,
    ) -> Hypothesis:
        """
        Main reasoning method - generate hypotheses from observations.
        
        This is the primary entry point for the Reason capability.
        Delegates to generate_hypothesis for actual logic.
        
        Args:
            observations: Recent observations/events
            context: Additional context (goals, state, etc.)
        
        Returns:
            Hypothesis explaining observations
        """
        return self.generate_hypothesis(observations, context)
    
    def generate_hypothesis(
        self,
        observations: list[dict],
        context: dict[str, Any] | None = None,
    ) -> Hypothesis:
        """
        Generate hypothesis from observations.
        
        Args:
            observations: Recent observations/events
            context: Additional context (goals, state, etc.)
        
        Returns:
            Hypothesis explaining observations
        """
        if not observations:
            # Default hypothesis when no observations
            return Hypothesis(
                description="No clear pattern detected",
                confidence=0.3,
            )
        
        # Simple pattern detection: find common features
        common_features = self._find_common_features(observations)
        
        # Generate hypothesis description
        if common_features:
            feature_str = ", ".join(f"{k}={v}" for k, v in list(common_features.items())[:3])
            description = f"Pattern detected: {feature_str}"
            confidence = min(1.0, len(common_features) * 0.2)
        else:
            description = "Events appear random or unrelated"
            confidence = 0.3
        
        hypothesis = Hypothesis(
            description=description,
            confidence=confidence,
            supporting_evidence=[{"observation": str(obs)} for obs in observations[:5]],
        )
        
        # Store hypothesis
        self.hypotheses[hypothesis.id] = hypothesis
        self.reasoning_stats["hypotheses_generated"] += 1
        
        return hypothesis
    
    def predict(
        self,
        state: dict[str, Any],
        action: str,
        world_model: Any = None,
    ) -> Prediction:
        """
        Predict outcome of action in current state.
        
        Args:
            state: Current state
            action: Action to predict outcome for
            world_model: Optional world model for prediction
        
        Returns:
            Prediction of outcome
        """
        # Use world model if available (assuming it's synchronous for now)
        if world_model and hasattr(world_model, 'predict'):
            try:
                predicted_outcome = world_model.predict(state, action)
                confidence = 0.7
            except Exception:
                predicted_outcome = self._naive_predict(state, action)
                confidence = 0.4
        else:
            predicted_outcome = self._naive_predict(state, action)
            confidence = 0.4
        
        prediction = Prediction(
            action=action,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
        )
        
        # Store prediction
        self.predictions[prediction.id] = prediction
        self.reasoning_stats["predictions_made"] += 1
        
        return prediction
    
    def infer_causality(
        self,
        events: list[dict[str, Any]],
    ) -> dict[tuple[str, str], CausalRelation]:
        """
        Infer causal relationships from event sequences.
        
        Args:
            events: List of events with type and timestamp
        
        Returns:
            Dictionary of (cause, effect) → CausalRelation
        """
        if len(events) < 2:
            return {}
        
        # Sort events by timestamp
        sorted_events = sorted(
            events,
            key=lambda e: e.get("timestamp", datetime.now(UTC))
        )
        
        # Look for temporal patterns (A before B suggests A causes B)
        for i in range(len(sorted_events) - 1):
            cause_event = sorted_events[i]
            effect_event = sorted_events[i + 1]
            
            cause_type = cause_event.get("type", "unknown")
            effect_type = effect_event.get("type", "unknown")
            
            key = (cause_type, effect_type)
            
            # Update or create causal relation
            if key in self.causal_relations:
                relation = self.causal_relations[key]
                relation.observed_together += 1
                relation.evidence_count += 1
                relation.update_strength()
            else:
                relation = CausalRelation(
                    cause=cause_type,
                    effect=effect_type,
                    observed_together=1,
                    evidence_count=1,
                )
                relation.update_strength()
                self.causal_relations[key] = relation
                self.reasoning_stats["causal_relations_discovered"] += 1
        
        return self.causal_relations
    
    def counterfactual_reasoning(
        self,
        actual_state: dict[str, Any],
        alternative_action: str,
    ) -> dict[str, Any]:
        """
        What-if reasoning: what would happen if different action taken?
        
        Args:
            actual_state: The state that actually occurred
            alternative_action: Alternative action to consider
        
        Returns:
            Hypothetical outcome if alternative action was taken
        """
        # Use prediction to simulate alternative
        alternative_prediction = self.predict(
            actual_state,
            alternative_action,
        )
        
        return {
            "alternative_action": alternative_action,
            "predicted_outcome": alternative_prediction.predicted_outcome,
            "confidence": alternative_prediction.confidence,
            "reasoning": f"If {alternative_action} instead, likely outcome: {alternative_prediction.predicted_outcome}",
        }
    
    def test_hypothesis(
        self,
        hypothesis_id: UUID,
        observation: dict[str, Any],
        supports: bool,
    ):
        """
        Test hypothesis against new observation.
        
        Args:
            hypothesis_id: Hypothesis to test
            observation: New observation
            supports: Whether observation supports hypothesis
        """
        if hypothesis_id not in self.hypotheses:
            return
        
        hypothesis = self.hypotheses[hypothesis_id]
        hypothesis.tested = True
        
        if supports:
            hypothesis.supporting_evidence.append(observation)
        else:
            hypothesis.contradicting_evidence.append(observation)
        
        # Update confidence
        hypothesis.update_confidence()
        
        # Confirm or reject if strong evidence
        if hypothesis.confidence > 0.8:
            hypothesis.confirmed = True
            self.reasoning_stats["hypotheses_confirmed"] += 1
        elif hypothesis.confidence < 0.2:
            hypothesis.confirmed = False
            self.reasoning_stats["hypotheses_rejected"] += 1
    
    def update_prediction_error(
        self,
        prediction_id: UUID,
        actual_outcome: dict[str, Any],
    ):
        """
        Update prediction with actual outcome and calculate error.
        
        Args:
            prediction_id: Prediction to update
            actual_outcome: What actually happened
        """
        if prediction_id not in self.predictions:
            return
        
        prediction = self.predictions[prediction_id]
        prediction.actual_outcome = actual_outcome
        
        # Calculate simple error: difference in keys/values
        predicted = prediction.predicted_outcome
        error = self._calculate_prediction_error(predicted, actual_outcome)
        
        prediction.prediction_error = error
        
        # Update average prediction error
        self.prediction_history.append(prediction)
        if self.prediction_history:
            avg_error = np.mean([
                p.prediction_error for p in self.prediction_history[-50:]
                if p.prediction_error is not None
            ])
            self.reasoning_stats["average_prediction_error"] = float(avg_error)
    
    def _find_common_features(
        self,
        observations: list[dict],
    ) -> dict[str, Any]:
        """Find features common across observations."""
        if not observations:
            return {}
        
        # Collect all keys - handle both dicts and PerceptionResult objects
        all_keys = set()
        for obs in observations:
            if hasattr(obs, 'keys'):
                # It's a dict
                all_keys.update(obs.keys())
            elif hasattr(obs, '__dict__'):
                # It's an object (like PerceptionResult), use its attributes
                all_keys.update(vars(obs).keys())
        
        # Find common values
        common = {}
        for key in all_keys:
            values = []
            for obs in observations:
                if isinstance(obs, dict) and key in obs:
                    values.append(obs.get(key))
                elif hasattr(obs, key):
                    values.append(getattr(obs, key))
            
            if len(values) >= len(observations) * 0.5:  # At least 50% have this key
                # Check if same value
                unique_values = set(str(v) for v in values)
                if len(unique_values) == 1:
                    common[key] = values[0]
        
        return common
    
    def _naive_predict(
        self,
        state: dict[str, Any],
        action: str,
    ) -> dict[str, Any]:
        """Simple prediction without world model."""
        # Use causal relations if available
        for (cause, effect), relation in self.causal_relations.items():
            if cause in action:
                return {
                    "likely_effect": effect,
                    "strength": relation.strength,
                    "method": "causal_inference",
                }
        
        # Default: state persists
        return {
            "state_change": "minimal",
            "method": "persistence_assumption",
        }
    
    def _calculate_prediction_error(
        self,
        predicted: dict[str, Any],
        actual: dict[str, Any],
    ) -> float:
        """Calculate error between predicted and actual outcomes."""
        # Simple metric: proportion of differing keys
        all_keys = set(predicted.keys()) | set(actual.keys())
        if not all_keys:
            return 0.0
        
        differences = sum(
            1 for key in all_keys
            if predicted.get(key) != actual.get(key)
        )
        
        return differences / len(all_keys)
    
    def get_reasoning_summary(self) -> dict[str, Any]:
        """Get summary of reasoning activity."""
        active_hypotheses = [
            h for h in self.hypotheses.values()
            if h.confirmed is None
        ]
        
        recent_predictions = self.prediction_history[-10:]
        
        return {
            "statistics": self.reasoning_stats,
            "active_hypotheses": len(active_hypotheses),
            "confirmed_hypotheses": sum(
                1 for h in self.hypotheses.values() if h.confirmed is True
            ),
            "rejected_hypotheses": sum(
                1 for h in self.hypotheses.values() if h.confirmed is False
            ),
            "recent_predictions": [p.to_dict() for p in recent_predictions],
            "causal_relations_count": len(self.causal_relations),
            "top_causal_relations": [
                {"cause": cause, "effect": effect, "strength": rel.strength}
                for (cause, effect), rel in sorted(
                    self.causal_relations.items(),
                    key=lambda x: x[1].strength,
                    reverse=True,
                )[:5]
            ],
        }
