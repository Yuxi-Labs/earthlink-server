"""Goal data structures."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4

import torch


class GoalStatus(Enum):
    """Goal lifecycle status."""

    PROPOSED = "proposed"     # Newly generated, not yet committed
    ACTIVE = "active"         # Currently being pursued
    IN_PROGRESS = "in_progress"  # Alias for active progress
    SUSPENDED = "suspended"   # Temporarily paused
    ACHIEVED = "achieved"     # Successfully completed
    ABANDONED = "abandoned"   # Given up (too difficult or irrelevant)
    FAILED = "failed"         # Attempted but could not achieve


class GoalType(Enum):
    """Types of goals based on origin and nature."""

    EXPLORATION = auto()  # Driven by curiosity
    COMPETENCE = auto()   # Skill improvement
    TASK = auto()         # Externally assigned
    SOCIAL = auto()       # Learned from other agents
    MAINTENANCE = auto()  # Self-preservation (energy, health)
    DISCOVERY = auto()    # Finding new information


@dataclass
class Goal:
    """
    Represents an agent goal.
    
    Goals are formed autonomously based on:
    - Curiosity signals (explore novel areas)
    - Competence progress (improve skills)
    - External tasks (assigned objectives)
    - Social learning (goals from other agents)
    """

    id: UUID = field(default_factory=uuid4)
    
    # Goal description
    description: str = ""
    goal_type: GoalType = GoalType.EXPLORATION
    status: GoalStatus = GoalStatus.PROPOSED
    
    # Goal embedding (for similarity comparisons)
    embedding: torch.Tensor | None = None
    
    # Target state (what we're trying to achieve)
    target_state: torch.Tensor | None = None
    
    # Priority and urgency
    priority: float = 0.5          # 0-1, higher = more important
    urgency: float = 0.5           # 0-1, higher = more time-sensitive
    intrinsic_value: float = 0.0   # Computed value from curiosity/competence
    
    # Hierarchy
    parent_id: UUID | None = None
    subgoal_ids: list[UUID] = field(default_factory=list)
    
    # Progress tracking
    progress: float = 0.0          # 0-1, completion percentage
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    
    # Time tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    deadline: datetime | None = None
    
    # Metadata
    source: str = "self"           # Origin: self, external, social
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark goal as active."""
        self.status = GoalStatus.ACTIVE
        self.started_at = datetime.now(UTC)
        self.attempts += 1

    def achieve(self) -> None:
        """Mark goal as achieved."""
        self.status = GoalStatus.ACHIEVED
        self.completed_at = datetime.now(UTC)
        self.progress = 1.0
        self.successes += 1

    def fail(self) -> None:
        """Mark goal as failed."""
        self.status = GoalStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.failures += 1

    def abandon(self) -> None:
        """Abandon the goal."""
        self.status = GoalStatus.ABANDONED
        self.completed_at = datetime.now(UTC)

    def suspend(self) -> None:
        """Temporarily suspend the goal."""
        self.status = GoalStatus.SUSPENDED

    def resume(self) -> None:
        """Resume a suspended goal."""
        self.status = GoalStatus.ACTIVE

    def update_progress(self, new_progress: float) -> None:
        """Update goal progress."""
        self.progress = max(0.0, min(1.0, new_progress))
        
        if self.progress >= 1.0:
            self.achieve()

    @property
    def is_terminal(self) -> bool:
        """Check if goal is in terminal state."""
        return self.status in (
            GoalStatus.ACHIEVED,
            GoalStatus.ABANDONED,
            GoalStatus.FAILED,
        )

    @property
    def is_active(self) -> bool:
        """Check if goal is currently being pursued."""
        return self.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS)

    @property
    def effective_priority(self) -> float:
        """
        Compute effective priority considering urgency.
        
        Uses urgency to boost priority as deadline approaches.
        """
        if self.deadline:
            time_remaining = (self.deadline - datetime.now(UTC)).total_seconds()
            if time_remaining <= 0:
                urgency_boost = 1.0
            else:
                # Exponential urgency increase
                time_to_deadline = (self.deadline - self.created_at).total_seconds()
                urgency_boost = 1.0 - (time_remaining / max(time_to_deadline, 1))
        else:
            urgency_boost = self.urgency

        return self.priority * (1 + 0.5 * urgency_boost)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "description": self.description,
            "goal_type": self.goal_type.name,
            "status": self.status.name,
            "priority": self.priority,
            "urgency": self.urgency,
            "intrinsic_value": self.intrinsic_value,
            "progress": self.progress,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "subgoal_ids": [str(sid) for sid in self.subgoal_ids],
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Goal":
        """Create from dictionary."""
        goal = cls()
        goal.id = UUID(data["id"])
        goal.description = data["description"]
        goal.goal_type = GoalType[data["goal_type"]]
        goal.status = GoalStatus[data["status"]]
        goal.priority = data["priority"]
        goal.urgency = data["urgency"]
        goal.intrinsic_value = data.get("intrinsic_value", 0.0)
        goal.progress = data["progress"]
        goal.attempts = data["attempts"]
        goal.successes = data["successes"]
        goal.failures = data["failures"]
        goal.parent_id = UUID(data["parent_id"]) if data["parent_id"] else None
        goal.subgoal_ids = [UUID(sid) for sid in data["subgoal_ids"]]
        goal.source = data["source"]
        goal.created_at = datetime.fromisoformat(data["created_at"])
        goal.started_at = datetime.fromisoformat(data["started_at"]) if data["started_at"] else None
        goal.completed_at = datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None
        goal.metadata = data.get("metadata", {})
        return goal
