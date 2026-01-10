"""
Decision Module - Goal-driven action selection with utility evaluation.

Provides intelligent decision-making capabilities:
- Multi-objective utility calculation
- Risk assessment from predictions
- Goal-based trade-off analysis
- Expected value computation
- Constraint satisfaction
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
import numpy as np


class DecisionStrategy(Enum):
    """Decision-making strategies."""
    MAXIMIZE_UTILITY = "maximize_utility"  # Choose highest expected utility
    MINIMIZE_RISK = "minimize_risk"  # Choose lowest risk
    BALANCED = "balanced"  # Balance utility and risk
    SATISFICE = "satisfice"  # First action above threshold
    EXPLORE = "explore"  # Maximize exploration/learning


@dataclass
class Action:
    """Potential action with metadata."""
    type: str  # "move", "learn", "communicate", etc.
    params: dict[str, Any]
    estimated_cost: float = 0.0  # Resource cost (time, energy, etc.)
    constraints: list[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.type, str(sorted(self.params.items()))))


@dataclass
class Goal:
    """Agent goal with priority and success criteria."""
    id: str
    description: str
    priority: float  # 0-1, higher = more important
    deadline: Optional[datetime] = None
    success_criteria: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0  # 0-1
    
    def is_satisfied(self, state: dict) -> bool:
        """Check if goal is satisfied by current state."""
        for key, target_value in self.success_criteria.items():
            if key not in state:
                return False
            if isinstance(target_value, (int, float)):
                if state[key] < target_value:
                    return False
            elif state[key] != target_value:
                return False
        return True


@dataclass
class UtilityComponents:
    """Breakdown of utility calculation."""
    goal_alignment: float  # How well action aligns with goals
    expected_reward: float  # Expected immediate reward
    information_gain: float  # Learning/exploration value
    social_value: float  # Collaboration/communication value
    long_term_value: float  # Future state value
    cost: float  # Resource expenditure (negative)
    total: float  # Sum of all components
    
    def __post_init__(self):
        self.total = (
            self.goal_alignment +
            self.expected_reward +
            self.information_gain +
            self.social_value +
            self.long_term_value -
            self.cost
        )


@dataclass
class RiskAssessment:
    """Risk evaluation for an action."""
    overall_risk: float  # 0-1, higher = more risky
    uncertainty: float  # Prediction uncertainty
    potential_loss: float  # Worst-case loss
    probability_failure: float  # Likelihood of failure
    mitigation_strategies: list[str] = field(default_factory=list)
    risk_factors: dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    """Selected action with rationale."""
    action: Action
    utility: UtilityComponents
    risk: RiskAssessment
    confidence: float  # 0-1, confidence in decision quality
    rationale: str  # Human-readable explanation
    alternatives_considered: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @property
    def chosen_action(self):
        """Backward compatibility: return action type."""
        if isinstance(self.action, str):
            return self.action
        return self.action.type if hasattr(self.action, 'type') else str(self.action)


class DecisionModule:
    """
    Intelligent decision-making for autonomous agents.
    
    Evaluates actions based on:
    - Multiple competing goals
    - Predicted outcomes from reasoning
    - Risk tolerance
    - Resource constraints
    - Exploration/exploitation trade-offs
    """
    
    def __init__(
        self,
        agent_id: str,
        strategy: DecisionStrategy = DecisionStrategy.BALANCED,
        risk_tolerance: float = 0.5,
        exploration_bonus: float = 0.1,
    ):
        self.agent_id = agent_id
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance  # 0 = risk-averse, 1 = risk-seeking
        self.exploration_bonus = exploration_bonus
        
        # Decision history for learning
        self.decision_history: list[Decision] = []
        
        # Statistics
        self.stats = {
            "decisions_made": 0,
            "avg_confidence": 0.0,
            "avg_utility": 0.0,
            "avg_risk": 0.0,
            "goals_achieved": 0,
            "decisions_by_type": {},
        }
    
    def decide(
        self,
        available_actions: list[Action] = None,
        goals: list[Goal] = None,
        world_state: dict = None,
        predictions: dict[str, Any] = None,
        reasoning_context: dict[str, Any] = None,
        # Legacy parameter names for backward compatibility
        actions: list[Action] = None,
        hypotheses: list = None,
        strategy: DecisionStrategy = None,
    ) -> Decision:
        """
        Select optimal action from available options.
        
        Args:
            available_actions: List of possible actions
            goals: Active goals to optimize for
            world_state: Current state of the world
            predictions: Predictions from ReasoningEngine (optional)
            reasoning_context: Hypotheses, causal relations (optional)
        
        Returns:
            Decision with selected action and rationale
        """
        # Handle legacy parameter names
        if actions is not None:
            available_actions = actions
        if hypotheses is not None and reasoning_context is None:
            reasoning_context = {"hypotheses": hypotheses}
        if strategy is not None:
            self.strategy = strategy
        
        if not available_actions:
            raise ValueError("No actions available for decision-making")
        
        if not goals:
            # No explicit goals - default to exploration
            goals = [Goal(
                id="default_explore",
                description="Explore and learn",
                priority=0.5,
            )]
        
        # Evaluate each action
        evaluations = []
        for action in available_actions:
            utility = self.calculate_utility(action, goals, world_state, predictions)
            risk = self.assess_risk(action, predictions, reasoning_context)
            
            # Calculate overall score based on strategy
            score = self._compute_decision_score(utility, risk)
            
            evaluations.append({
                "action": action,
                "utility": utility,
                "risk": risk,
                "score": score,
            })
        
        # Sort by score (descending)
        evaluations.sort(key=lambda x: x["score"], reverse=True)
        
        # Select action based on strategy
        selected = evaluations[0]  # Best action
        
        # Build decision rationale
        rationale = self._build_rationale(selected, goals, evaluations)
        
        # Calculate confidence
        if len(evaluations) > 1:
            # Confidence based on gap between best and second-best
            gap = selected["score"] - evaluations[1]["score"]
            confidence = min(0.5 + gap / 2, 1.0)
        else:
            confidence = 0.7  # Single option
        
        decision = Decision(
            action=selected["action"],
            utility=selected["utility"],
            risk=selected["risk"],
            confidence=confidence,
            rationale=rationale,
            alternatives_considered=len(evaluations),
        )
        
        # Update statistics
        self._update_stats(decision)
        
        # Store in history
        self.decision_history.append(decision)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        return decision
    
    def calculate_utility(
        self,
        action: Action,
        goals: list[Goal],
        world_state: dict,
        predictions: dict[str, Any] = None,
    ) -> UtilityComponents:
        """
        Calculate expected utility of action for current goals.
        
        Utility = goal_alignment + expected_reward + info_gain + social + long_term - cost
        """
        # 1. Goal alignment - how well action advances goals
        goal_alignment = self._calculate_goal_alignment(action, goals, world_state)
        
        # 2. Expected reward from predictions
        expected_reward = 0.0
        if predictions and action.type in predictions:
            pred = predictions[action.type]
            # Handle both dict and Prediction objects
            if isinstance(pred, dict):
                pred_outcome = pred.get("predicted_outcome", {})
                expected_reward = pred_outcome.get("reward", 0.0)
            elif hasattr(pred, 'predicted_outcome'):
                pred_outcome = pred.predicted_outcome
                expected_reward = getattr(pred_outcome, 'reward', 0.0) if pred_outcome else 0.0
        
        # 3. Information gain (exploration bonus)
        information_gain = self._calculate_information_gain(action, world_state)
        
        # 4. Social value (collaboration potential)
        social_value = self._calculate_social_value(action, world_state)
        
        # 5. Long-term value (future state improvement)
        long_term_value = self._estimate_long_term_value(action, goals, predictions)
        
        # 6. Cost (resources required)
        cost = action.estimated_cost
        
        return UtilityComponents(
            goal_alignment=goal_alignment,
            expected_reward=expected_reward,
            information_gain=information_gain,
            social_value=social_value,
            long_term_value=long_term_value,
            cost=cost,
            total=0.0,  # Computed in __post_init__
        )
    
    def assess_risk(
        self,
        action: Action,
        predictions: dict[str, Any] = None,
        reasoning_context: dict[str, Any] = None,
    ) -> RiskAssessment:
        """
        Evaluate risks associated with action.
        
        Risk factors:
        - Prediction uncertainty
        - Potential losses
        - Constraint violations
        - Hypothesis contradictions
        """
        risk_factors = {}
        
        # 1. Prediction uncertainty
        uncertainty = 0.5  # Default moderate uncertainty
        if predictions and action.type in predictions:
            pred = predictions[action.type]
            # Handle both dict and Prediction objects
            if isinstance(pred, dict):
                uncertainty = 1.0 - pred.get("confidence", 0.5)
            elif hasattr(pred, 'confidence'):
                uncertainty = 1.0 - pred.confidence
        risk_factors["uncertainty"] = uncertainty
        
        # 2. Potential loss (worst-case scenario)
        potential_loss = action.estimated_cost * 2  # Assume cost could double
        if predictions and action.type in predictions:
            pred = predictions[action.type]
            # Handle both dict and Prediction objects
            if isinstance(pred, dict):
                pred_outcome = pred.get("predicted_outcome", {})
                potential_loss = max(potential_loss, -pred_outcome.get("min_reward", 0))
            elif hasattr(pred, 'predicted_outcome'):
                pred_outcome = pred.predicted_outcome
                min_reward = getattr(pred_outcome, 'min_reward', 0) if pred_outcome else 0
                potential_loss = max(potential_loss, -min_reward)
        risk_factors["potential_loss"] = min(potential_loss / 10, 1.0)  # Normalize
        
        # 3. Constraint violations
        constraint_risk = len(action.constraints) * 0.1  # More constraints = more risk
        risk_factors["constraints"] = min(constraint_risk, 1.0)
        
        # 4. Hypothesis contradictions (from reasoning)
        hypothesis_risk = 0.0
        if reasoning_context and "hypotheses" in reasoning_context:
            # Check if action contradicts confirmed hypotheses
            hypotheses = reasoning_context["hypotheses"]
            # Handle both dict and list of hypotheses
            hyp_list = hypotheses.values() if isinstance(hypotheses, dict) else hypotheses
            for hyp in hyp_list:
                # hyp is a Hypothesis object
                if (hyp.confirmed and 
                    action.type in hyp.description.lower()):
                    # Contradicting confirmed hypothesis is risky
                    if any(word in hyp.description.lower() 
                           for word in ["avoid", "dangerous", "fail"]):
                        hypothesis_risk += 0.2
        risk_factors["hypothesis_contradiction"] = min(hypothesis_risk, 1.0)
        
        # Overall risk (weighted average)
        overall_risk = (
            risk_factors["uncertainty"] * 0.3 +
            risk_factors["potential_loss"] * 0.4 +
            risk_factors["constraints"] * 0.1 +
            risk_factors["hypothesis_contradiction"] * 0.2
        )
        
        # Probability of failure
        probability_failure = overall_risk * 0.5  # Conservative estimate
        
        # Mitigation strategies
        mitigation = []
        if uncertainty > 0.7:
            mitigation.append("Gather more information before acting")
        if potential_loss > 0.5:
            mitigation.append("Prepare fallback plan")
        if constraint_risk > 0.3:
            mitigation.append("Verify all constraints are satisfied")
        
        return RiskAssessment(
            overall_risk=overall_risk,
            uncertainty=uncertainty,
            potential_loss=potential_loss,
            probability_failure=probability_failure,
            mitigation_strategies=mitigation,
            risk_factors=risk_factors,
        )
    
    def _calculate_goal_alignment(
        self,
        action: Action,
        goals: list[Goal],
        world_state: dict,
    ) -> float:
        """Calculate how well action advances goals."""
        if not goals:
            return 0.0
        
        total_alignment = 0.0
        total_weight = 0.0
        
        for goal in goals:
            weight = goal.priority
            
            # Simple keyword matching for alignment
            alignment = 0.0
            goal_keywords = goal.description.lower().split()
            action_text = f"{action.type} {str(action.params)}".lower()
            
            # Check keyword overlap
            matches = sum(1 for kw in goal_keywords if kw in action_text)
            if goal_keywords:
                alignment = matches / len(goal_keywords)
            
            # Boost if action type directly matches goal
            if action.type in goal.description.lower():
                alignment = max(alignment, 0.7)

            # Heuristic boosts for common goal/action pairs
            if action.type == "move" and "explor" in goal.description.lower():
                alignment = max(alignment, 0.7)
            if action.type == "learn" and ("learn" in goal.description.lower() or "knowledge" in goal.description.lower()):
                alignment = max(alignment, 0.7)
            
            # Reduce if goal is nearly complete (diminishing returns)
            alignment *= (1.0 - goal.progress * 0.5)
            
            total_alignment += alignment * weight
            total_weight += weight
        
        return total_alignment / max(total_weight, 1.0)
    
    def _calculate_information_gain(
        self,
        action: Action,
        world_state: dict,
    ) -> float:
        """Estimate learning value of action."""
        # Both move and learn provide equal information gain
        # Moving to new locations provides spatial knowledge
        # Learning provides conceptual knowledge
        if action.type == "move":
            # Movement provides discovery of new locations, features, patterns
            return self.exploration_bonus * 2
        
        if action.type in ["explore", "learn", "experiment"]:
            return self.exploration_bonus * 2
        
        # Communicating can provide info
        if action.type in ["communicate", "query", "observe"]:
            return self.exploration_bonus * 1.5
        
        return 0.0
    
    def _calculate_social_value(
        self,
        action: Action,
        world_state: dict,
    ) -> float:
        """Estimate social/collaborative value."""
        if action.type in ["communicate", "share", "collaborate", "help"]:
            return 0.2
        return 0.0
    
    def _estimate_long_term_value(
        self,
        action: Action,
        goals: list[Goal],
        predictions: dict[str, Any] = None,
    ) -> float:
        """Estimate future value from action."""
        # Both learning and movement have equal long-term value
        # Movement enables access to new regions and resources
        # Learning builds knowledge for better decisions
        if action.type == "learn":
            return 0.2
        
        if action.type == "move":
            return 0.2  # Movement is equally valuable - enables exploration
        
        # Actions advancing high-priority goals have long-term value
        if goals:
            max_priority = max(g.priority for g in goals)
            if action.type in str([g.description for g in goals]).lower():
                return max_priority * 0.2
        
        return 0.0
    
    def _compute_decision_score(
        self,
        utility: UtilityComponents,
        risk: RiskAssessment,
    ) -> float:
        """Compute overall decision score from utility and risk."""
        if self.strategy == DecisionStrategy.MAXIMIZE_UTILITY:
            return utility.total
        
        elif self.strategy == DecisionStrategy.MINIMIZE_RISK:
            return -risk.overall_risk
        
        elif self.strategy == DecisionStrategy.BALANCED:
            # Balance utility and risk based on risk tolerance
            risk_penalty = risk.overall_risk * (1.0 - self.risk_tolerance)
            return utility.total - risk_penalty
        
        elif self.strategy == DecisionStrategy.SATISFICE:
            # First action above threshold
            threshold = 0.5
            return utility.total if utility.total >= threshold else -1.0
        
        elif self.strategy == DecisionStrategy.EXPLORE:
            # Maximize exploration
            return utility.information_gain * 2 + utility.total * 0.5
        
        return utility.total
    
    def _build_rationale(
        self,
        selected: dict,
        goals: list[Goal],
        all_evaluations: list[dict],
    ) -> str:
        """Build human-readable explanation of decision."""
        action = selected["action"]
        utility = selected["utility"]
        risk = selected["risk"]
        
        rationale_parts = []
        
        # Primary reason
        if utility.goal_alignment > 0.5:
            top_goal = max(goals, key=lambda g: g.priority)
            rationale_parts.append(f"Advances '{top_goal.description}' (priority {top_goal.priority:.2f})")
        elif utility.information_gain > 0.3:
            rationale_parts.append("High exploration/learning value")
        elif utility.expected_reward > 0.3:
            rationale_parts.append(f"Expected reward: {utility.expected_reward:.2f}")
        else:
            rationale_parts.append("Best available option")
        
        # Risk consideration
        if risk.overall_risk > 0.7:
            rationale_parts.append(f"HIGH RISK ({risk.overall_risk:.2f})")
        elif risk.overall_risk < 0.3:
            rationale_parts.append("Low risk")
        
        # Comparison to alternatives
        if len(all_evaluations) > 1:
            gap = selected["score"] - all_evaluations[1]["score"]
            if gap > 0.3:
                rationale_parts.append(f"Clearly superior to alternatives (+{gap:.2f})")
            elif gap < 0.1:
                rationale_parts.append("Marginal advantage over alternatives")
        
        return "; ".join(rationale_parts)
    
    def _update_stats(self, decision: Decision):
        """Update decision-making statistics."""
        self.stats["decisions_made"] += 1
        n = self.stats["decisions_made"]
        
        # Running averages
        self.stats["avg_confidence"] = (
            (self.stats["avg_confidence"] * (n - 1) + decision.confidence) / n
        )
        self.stats["avg_utility"] = (
            (self.stats["avg_utility"] * (n - 1) + decision.utility.total) / n
        )
        self.stats["avg_risk"] = (
            (self.stats["avg_risk"] * (n - 1) + decision.risk.overall_risk) / n
        )
        
        # Count by action type
        action_type = decision.action.type
        if action_type not in self.stats["decisions_by_type"]:
            self.stats["decisions_by_type"][action_type] = 0
        self.stats["decisions_by_type"][action_type] += 1
    
    def evaluate_decision_outcome(
        self,
        decision: Decision,
        actual_outcome: dict,
    ) -> float:
        """
        Evaluate how well decision performed.
        
        Returns quality score (0-1).
        """
        # Compare predicted vs actual utility
        predicted_utility = decision.utility.total
        
        actual_utility = 0.0
        if "reward" in actual_outcome:
            actual_utility = actual_outcome["reward"]
        
        # Quality is inverse of error
        error = abs(predicted_utility - actual_utility)
        quality = max(0.0, 1.0 - error)
        
        return quality
    
    def get_decision_summary(self) -> dict:
        """Get summary of decision-making performance."""
        recent_decisions = self.decision_history[-100:] if self.decision_history else []
        
        return {
            **self.stats,
            "recent_decisions_count": len(recent_decisions),
            "recent_avg_confidence": (
                np.mean([d.confidence for d in recent_decisions])
                if recent_decisions else 0.0
            ),
            "strategy": self.strategy.value,
            "risk_tolerance": self.risk_tolerance,
        }
