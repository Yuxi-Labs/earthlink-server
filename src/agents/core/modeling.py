"""
Modeling capability: build world, agent (theory of mind), and concept models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import UUID

import numpy as np

from .world_model import AgentModel, Concept, WorldModel


@dataclass
class ModelingResult:
    """Combined modeling outputs for convenience."""

    world_model: WorldModel | None = None
    agent_models: list[AgentModel] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_model": self.world_model.to_dict() if self.world_model else None,
            "agent_models": [a.to_dict() for a in self.agent_models],
            "concepts": [c.to_dict() for c in self.concepts],
        }


class ModelingModule:
    """Build representations of environment, other agents, and concepts."""

    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self._world_model: WorldModel | None = None
        self._agent_models: dict[UUID, AgentModel] = {}
        self._concepts: list[Concept] = []

    async def build_world_model(
        self,
        observations: list[dict[str, Any]],
    ) -> WorldModel:
        """
        Construct a simple world model from spatial observations.
        """
        regions = []
        entities = []

        for obs in observations:
            data = obs.get("data", {})
            if "containing_regions" in data:
                regions.extend(data.get("containing_regions", []))
            if "nearby_features" in data:
                entities.extend(data.get("nearby_features", []))

        confidence = min(1.0, 0.5 + 0.05 * len(observations))
        self._world_model = WorldModel(
            regions=regions[:50],
            entities=entities[:100],
            confidence=confidence,
        )
        return self._world_model

    async def model_agent(
        self,
        agent_id: UUID,
        interaction_history: list[dict[str, Any]],
    ) -> AgentModel:
        """
        Build a theory-of-mind model for another agent based on interactions.
        """
        intents = []
        traits = {}

        # Extract simple intent/trait heuristics
        for event in interaction_history[-20:]:
            if intent := event.get("intent"):
                intents.append(intent)
            if event.get("trust_score") is not None:
                traits["trust_score"] = event["trust_score"]
            if event.get("domain"):
                traits["domain"] = event["domain"]

        reliability = min(1.0, 0.5 + 0.02 * len(interaction_history))
        model = AgentModel(
            agent_id=agent_id,
            intents=intents[:10],
            traits=traits,
            reliability=reliability,
        )
        self._agent_models[agent_id] = model
        return model

    async def extract_concept(
        self,
        data: list[dict[str, Any]],
        clustering_threshold: float = 0.5,
    ) -> Concept:
        """
        Extract a concept by clustering numeric fields and labeling by mode keys.
        """
        if not data:
            return Concept(label="unknown", cohesion=0.0)

        numeric_vectors = []
        labels = []
        for item in data:
            vec = [v for v in item.values() if isinstance(v, (int, float))]
            if vec:
                numeric_vectors.append(vec)
            labels.extend([k for k, v in item.items() if isinstance(v, str)])

        cohesion = 0.0
        if numeric_vectors:
            lengths = [len(v) for v in numeric_vectors]
            min_len = min(lengths)
            if min_len > 0:
                trimmed = np.array([v[:min_len] for v in numeric_vectors])
                variances = trimmed.var(axis=0)
                cohesion = float(max(0.0, 1.0 - variances.mean()))
        label = self._most_common_label(labels) or "concept"

        concept = Concept(label=label, exemplars=data[:5], cohesion=cohesion)
        self._concepts.append(concept)
        return concept

    def get_state(self) -> ModelingResult:
        """Return current modeling outputs."""
        return ModelingResult(
            world_model=self._world_model,
            agent_models=list(self._agent_models.values()),
            concepts=self._concepts,
        )

    def _most_common_label(self, labels: Iterable[str]) -> str | None:
        """Return the most common label if available."""
        from collections import Counter

        counts = Counter(labels)
        if not counts:
            return None
        return counts.most_common(1)[0][0]
