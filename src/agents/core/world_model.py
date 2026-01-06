"""
Data structures for agent world and agent models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class WorldModel:
    """Spatial/semantic world representation."""

    id: UUID = field(default_factory=uuid4)
    regions: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "regions": self.regions,
            "entities": self.entities,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class AgentModel:
    """Theory of mind for another agent."""

    agent_id: UUID
    traits: dict[str, Any] = field(default_factory=dict)
    intents: list[str] = field(default_factory=list)
    reliability: float = 0.5
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "traits": self.traits,
            "intents": self.intents,
            "reliability": self.reliability,
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass
class Concept:
    """Abstract concept distilled from observations."""

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    exemplars: list[dict[str, Any]] = field(default_factory=list)
    cohesion: float = 0.0  # 0-1 clustering quality
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "label": self.label,
            "cohesion": self.cohesion,
            "exemplars": self.exemplars,
            "created_at": self.created_at.isoformat(),
        }
