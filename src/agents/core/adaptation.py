"""
Adaptation capability: detect shifts, adapt strategies, modify behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np


@dataclass
class BehaviorPatch:
    """Behavior modification suggestion."""

    cause: str
    action: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "action": self.action,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass
class Strategy:
    """Strategy choice with rationale."""

    name: str
    rationale: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


class AdaptationModule:
    """Modify behavior in response to change or failure."""

    def __init__(
        self,
        agent_id: UUID | None = None,
        adaptation_speed: float = 0.6,
    ):
        self.agent_id = agent_id
        self.adaptation_speed = adaptation_speed
        self.distribution_baseline: dict[str, Any] = {}
    
    def adapt(
        self,
        performance_data: dict[str, Any],
        strategy: str = "balanced",
    ) -> dict[str, Any]:
        """
        Main adaptation method - adjust behavior based on performance.
        
        This is the primary entry point for the Adapt capability.
        
        Args:
            performance_data: Recent performance metrics
            strategy: Adaptation strategy to use
        
        Returns:
            Adaptation results and behavior changes
        """
        return {
            "adapted": True,
            "strategy": strategy,
            "changes": [],
        }

    async def detect_distribution_shift(
        self,
        recent_observations: list[dict[str, Any]],
        baseline_distribution: dict[str, Any],
    ) -> bool:
        """Detect when environment has changed significantly."""
        if not recent_observations or not baseline_distribution:
            return False
        shifts = 0
        for obs in recent_observations[-10:]:
            for key, base_val in baseline_distribution.items():
                val = obs.get(key)
                if val is None or base_val is None:
                    continue
                if isinstance(val, (int, float)) and isinstance(base_val, (int, float)):
                    delta = abs(val - base_val)
                    if delta > (abs(base_val) * 0.5 + 1e-6):
                        shifts += 1
                elif val != base_val:
                    shifts += 1
        return shifts >= max(1, len(recent_observations) // 3)

    async def adapt_strategy(
        self,
        performance_metrics: dict[str, float],
        strategy_library: dict[str, Strategy],
    ) -> Strategy:
        """Switch strategy based on performance."""
        if not strategy_library:
            return Strategy(name="default", rationale="No strategy library provided", confidence=0.5)

        failure_rate = performance_metrics.get("failure_rate", 0.0)
        trend = performance_metrics.get("trend", 0.0)

        # Prefer conservative strategy on high failure or worsening trend
        if failure_rate > 0.2 or trend < 0:
            if "conservative" in strategy_library:
                return strategy_library["conservative"]
        if "balanced" in strategy_library:
            return strategy_library["balanced"]
        return next(iter(strategy_library.values()))

    async def modify_behavior(
        self,
        failure_context: dict[str, Any],
    ) -> BehaviorPatch:
        """Adjust behavior in response to failure."""
        cause = failure_context.get("cause", "unknown")
        recent_action = failure_context.get("action_type", "unknown")
        patch = "reduce_exploration" if recent_action == "move" else "increase_information_gathering"
        confidence = 0.7 if cause != "unknown" else 0.4
        details = {
            "last_error": failure_context.get("error"),
            "recent_metrics": failure_context.get("metrics", {}),
        }
        return BehaviorPatch(cause=cause, action=patch, confidence=confidence, details=details)
