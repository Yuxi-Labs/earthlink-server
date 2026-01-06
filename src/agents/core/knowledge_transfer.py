"""
Knowledge Transfer capability: extract and share knowledge between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class Knowledge:
    """Transferable knowledge package."""

    id: UUID = field(default_factory=uuid4)
    domain: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "domain": self.domain,
            "content": self.content,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TransferResult:
    """Result of a knowledge transfer attempt."""

    success: bool
    recipient: str
    knowledge_id: str
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "recipient": self.recipient,
            "knowledge_id": self.knowledge_id,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
        }


class KnowledgeTransferModule:
    """Extract, package, and transfer knowledge to others or collectives."""

    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.knowledge_history: list[Knowledge] = []
        self.transfer_history: list[TransferResult] = []

    async def extract_knowledge(
        self,
        domain: str,
        observations: list[dict[str, Any]] | None = None,
        hypotheses: list[Any] | None = None,
    ) -> Knowledge:
        """
        Extract knowledge from observations and hypotheses for a domain.
        """
        observations = observations or []
        hypotheses = hypotheses or []

        distilled = {
            "facts": observations[:5],
            "hypotheses": [getattr(h, "description", str(h)) for h in hypotheses][:5],
        }
        confidence = min(1.0, 0.5 + 0.1 * len(observations) + 0.1 * len(hypotheses))

        knowledge = Knowledge(
            domain=domain,
            content=distilled,
            confidence=confidence,
            provenance={"agent_id": str(self.agent_id), "source_count": len(observations)},
        )
        self.knowledge_history.append(knowledge)
        return knowledge

    async def transfer_to_agent(
        self,
        recipient: UUID,
        knowledge: Knowledge,
    ) -> TransferResult:
        """
        Directly transfer knowledge to another agent (simulated).
        """
        rationale = "Direct transfer prepared."
        result = TransferResult(
            success=True,
            recipient=str(recipient),
            knowledge_id=str(knowledge.id),
            rationale=rationale,
        )
        self.transfer_history.append(result)
        return result

    async def contribute_to_collective(
        self,
        knowledge: Knowledge,
        collective_id: str,
    ) -> bool:
        """
        Contribute knowledge to a shared collective (simulated).
        """
        # In a full system, this would write to a shared store or bus.
        self.transfer_history.append(
            TransferResult(
                success=True,
                recipient=collective_id,
                knowledge_id=str(knowledge.id),
                rationale="Contributed to collective knowledge base.",
            )
        )
        return True
