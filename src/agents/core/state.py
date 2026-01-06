"""Agent state representation."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import torch


class AgentLifecycle(str, Enum):
    """Agent developmental lifecycle stages - internal tracking only (NOT UI-facing)."""

    SPAWNED = "spawned"
    ORIENTED = "oriented"
    EXPLORES = "explores"
    LEARNS = "learns"
    ADAPTS = "adapts"
    DIFFERENTIATES = "differentiates"
    ACTS = "acts"
    TRANSFORMS = "transforms"
    EXPIRES = "expires"
    ARCHIVED = "archived"


class AgentStatus(str, Enum):
    """Agent operational status (UI-facing)."""

    IDLE = "idle"
    EXPLORING = "exploring"
    LEARNING = "learning"
    INTERACTING = "interacting"
    EXECUTING = "executing"
    ADAPTING = "adapting"
    OVERLOADED = "overloaded"
    CORRUPTED = "corrupted"
    RETIRED = "retired"


@dataclass
class Coordinates:
    """Position in abstract VW space."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.z], dtype=torch.float32)

    def distance_to(self, other: "Coordinates") -> float:
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


@dataclass
class AgentMetrics:
    """
    Metrics for agents in Earthlink - grounded in what the system actually does.
    
    Organized by stakeholder:
    - Regular Users: Is my agent exploring and learning?
    - Researchers: What intelligence patterns are emerging?
    - Developers: Is the system functioning correctly?
    """
    
    # ===== REGULAR USERS: Agent Activity & Progress =====
    # What is my agent doing right now and what have they accomplished?
    
    distance_traveled_km: float = 0.0  # Total distance moved across Earth's surface
    unique_locations_visited: int = 0  # Distinct lat/lon grid cells visited (e.g., 10km x 10km)
    exploration_area_km2: float = 0.0  # Geographic area covered (convex hull)
    time_alive_hours: float = 0.0  # Hours since spawning
    
    knowledge_items_learned: int = 0  # Distinct topics/facts acquired
    goals_achieved: int = 0  # Completed goals
    goals_active: int = 0  # Currently pursuing goals
    
    agents_encountered: int = 0  # Other agents met (within interaction distance)
    messages_sent: int = 0  # Messages sent to other agents
    messages_received: int = 0  # Messages received from other agents
    
    current_activity: str = ""  # Human-readable: "Moving north", "Learning about Melbourne"
    
    # ===== RESEARCHERS: Cognitive & Behavioral Patterns =====
    # How is intelligence developing? What behaviors emerge?
    
    exploration_entropy: float = 0.0  # Movement randomness (0=fixed path, 1=totally random)
    knowledge_diversity: float = 0.0  # Breadth of topics learned (0-1, Shannon entropy)
    goal_persistence: float = 0.0  # Average % completion before switching goals
    
    learning_rate: float = 0.0  # Knowledge items per hour
    social_frequency: float = 0.0  # Interactions per hour
    
    decision_distribution: dict[str, int] = field(default_factory=dict)  # {"move": 45, "learn": 38, "interact": 17}
    
    curiosity_score: float = 0.5  # Intrinsic motivation (from neural network)
    prediction_accuracy: float = 0.0  # World model prediction quality (0-1)
    decision_accuracy: float = 0.0  # Chosen action accuracy (0-1)
    perception_noise_reduction: float = 0.0  # Filtering effectiveness (0-1)
    avg_decision_confidence: float = 0.0  # Rolling decision confidence
    uncertainty_score: float = 0.0  # Self-estimated uncertainty (0-1)
    rolling_step_duration_ms: float = 0.0  # Recent average step duration
    failure_rate: float = 0.0  # Rolling failure rate over monitoring window
    performance_trend: str = ""  # improving/stable/worsening
    last_diagnosis: str = ""  # Latest self-monitoring diagnosis summary
    
    # ===== DEVELOPERS: System Health & Debugging =====
    # Is the agent working? Any errors? Performance issues?
    
    total_steps_executed: int = 0  # Total autonomous_step() calls
    successful_steps: int = 0  # Steps without errors
    failed_steps: int = 0  # Steps that raised exceptions
    
    avg_step_duration_ms: float = 0.0  # Processing time per step
    last_error: str = ""  # Most recent error message
    error_count: int = 0  # Total errors encountered
    
    neural_network_calls: int = 0  # Inference forward passes
    training_updates: int = 0  # Gradient descent steps performed
    memory_entries: int = 0  # Items in episodic memory
    
    last_activity_timestamp: str = ""  # ISO timestamp of last action


@dataclass
class AgentState:
    """Complete agent state representation."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    lifecycle: AgentLifecycle = AgentLifecycle.SPAWNED
    status: AgentStatus = AgentStatus.IDLE
    location: Coordinates = field(default_factory=Coordinates)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Current goal (if any)
    current_goal_id: UUID | None = None

    # Target world (if specializing/deployed)
    target_world: str | None = None

    # Internal state tensor (for neural networks)
    _latent_state: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "lifecycle": self.lifecycle.value,
            "status": self.status.value,
            "location": {"x": self.location.x, "y": self.location.y, "z": self.location.z},
            "metrics": {
                # Regular Users: Activity & Progress
                "distance_traveled_km": self.metrics.distance_traveled_km,
                "unique_locations_visited": self.metrics.unique_locations_visited,
                "exploration_area_km2": self.metrics.exploration_area_km2,
                "time_alive_hours": self.metrics.time_alive_hours,
                "knowledge_items_learned": self.metrics.knowledge_items_learned,
                "goals_achieved": self.metrics.goals_achieved,
                "goals_active": self.metrics.goals_active,
                "agents_encountered": self.metrics.agents_encountered,
                "messages_sent": self.metrics.messages_sent,
                "messages_received": self.metrics.messages_received,
                "current_activity": self.metrics.current_activity,
                
                # Researchers: Cognitive Patterns
                "exploration_entropy": self.metrics.exploration_entropy,
                "knowledge_diversity": self.metrics.knowledge_diversity,
                "goal_persistence": self.metrics.goal_persistence,
                "learning_rate": self.metrics.learning_rate,
                "social_frequency": self.metrics.social_frequency,
                "decision_distribution": self.metrics.decision_distribution,
                "curiosity_score": self.metrics.curiosity_score,
                "prediction_accuracy": self.metrics.prediction_accuracy,
                "decision_accuracy": self.metrics.decision_accuracy,
                "perception_noise_reduction": self.metrics.perception_noise_reduction,
                "avg_decision_confidence": self.metrics.avg_decision_confidence,
                "uncertainty_score": self.metrics.uncertainty_score,
                "rolling_step_duration_ms": self.metrics.rolling_step_duration_ms,
                "failure_rate": self.metrics.failure_rate,
                "performance_trend": self.metrics.performance_trend,
                "last_diagnosis": self.metrics.last_diagnosis,
                
                # Developers: System Health
                "total_steps_executed": self.metrics.total_steps_executed,
                "successful_steps": self.metrics.successful_steps,
                "failed_steps": self.metrics.failed_steps,
                "avg_step_duration_ms": self.metrics.avg_step_duration_ms,
                "last_error": self.metrics.last_error,
                "error_count": self.metrics.error_count,
                "neural_network_calls": self.metrics.neural_network_calls,
                "training_updates": self.metrics.training_updates,
                "memory_entries": self.metrics.memory_entries,
                "last_activity_timestamp": self.metrics.last_activity_timestamp,
            },
            "current_goal_id": str(self.current_goal_id) if self.current_goal_id else None,
            "target_world": self.target_world,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @property
    def latent_state(self) -> torch.Tensor:
        """Get or initialize latent state tensor."""
        if self._latent_state is None:
            self._latent_state = torch.zeros(256, dtype=torch.float32)
        return self._latent_state

    @latent_state.setter
    def latent_state(self, value: torch.Tensor) -> None:
        self._latent_state = value
