"""Agent state representation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import torch


class AgentLifecycle(str, Enum):
    """Agent lifecycle states."""

    SPAWNED = "spawned"
    TRAINING = "training"
    EXPLORING = "exploring"
    TESTING = "testing"
    DEPLOYED = "deployed"
    SUSPENDED = "suspended"


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
    """Agent performance metrics."""

    knowledge_acquired: float = 0.0
    worlds_explored: int = 0
    collaboration_score: float = 0.0
    innovation_index: float = 0.0
    total_steps: int = 0
    total_rewards: float = 0.0
    curiosity_score: float = 0.0
    topics_explored: int = 0
    knowledge_sources_queried: int = 0
    prediction_errors: float = 0.0
    novelty_encountered: float = 0.0
    training_steps: int = 0
    last_training_loss: float = 0.0


@dataclass
class AgentState:
    """Complete agent state representation."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    lifecycle: AgentLifecycle = AgentLifecycle.SPAWNED
    location: Coordinates = field(default_factory=Coordinates)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

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
            "location": {"x": self.location.x, "y": self.location.y, "z": self.location.z},
            "metrics": {
                "knowledge_acquired": self.metrics.knowledge_acquired,
                "worlds_explored": self.metrics.worlds_explored,
                "collaboration_score": self.metrics.collaboration_score,
                "innovation_index": self.metrics.innovation_index,
                "total_steps": self.metrics.total_steps,
                "total_rewards": self.metrics.total_rewards,
                "curiosity_score": self.metrics.curiosity_score,
                "topics_explored": self.metrics.topics_explored,
                "knowledge_sources_queried": self.metrics.knowledge_sources_queried,
                "prediction_errors": self.metrics.prediction_errors,
                "novelty_encountered": self.metrics.novelty_encountered,
                "training_steps": self.metrics.training_steps,
                "last_training_loss": self.metrics.last_training_loss,
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
