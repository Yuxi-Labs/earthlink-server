"""
Specialization capability: detect emerging expertise and niches.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class SpecializationProfile:
    """Detected specialization with supporting evidence."""

    domain: str | None
    dominance: float  # fraction of activity in this domain (0-1)
    expertise_level: float  # normalized expertise score (0-1)
    confidence: float  # detection confidence (0-1)
    evidence_count: int
    recency_bias: float
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "dominance": self.dominance,
            "expertise_level": self.expertise_level,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "recency_bias": self.recency_bias,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ExpertiseEstimate:
    """Expertise estimate for a domain."""

    domain: str
    proficiency: float
    success_rate: float
    volume: int
    rationale: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "proficiency": self.proficiency,
            "success_rate": self.success_rate,
            "volume": self.volume,
            "rationale": self.rationale,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Niche:
    """Candidate niche for differentiation."""

    id: UUID = field(default_factory=uuid4)
    domain: str = ""
    saturation_score: float = 0.0  # lower = less crowded
    opportunity: float = 0.0  # higher = better niche
    rationale: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "domain": self.domain,
            "saturation_score": self.saturation_score,
            "opportunity": self.opportunity,
            "rationale": self.rationale,
            "created_at": self.created_at.isoformat(),
        }


class SpecializationModule:
    """
    Detect specialization patterns, estimate expertise, and find niches.
    """

    def __init__(self, agent_id: UUID, history_window: int = 200) -> None:
        self.agent_id = agent_id
        self.history_window = history_window
        self.activity_history: list[dict[str, Any]] = []

    async def detect_specialization(
        self,
        activity_history: list[dict[str, Any]],
    ) -> SpecializationProfile:
        """
        Detect dominant domain from activity history.

        Each activity item can include:
        - domain: str
        - success: bool
        - reward: float
        - timestamp: datetime or unix seconds
        """
        if activity_history:
            self.activity_history.extend(activity_history)
            self.activity_history = self.activity_history[-self.history_window :]

        if not self.activity_history:
            return SpecializationProfile(
                domain=None,
                dominance=0.0,
                expertise_level=0.0,
                confidence=0.0,
                evidence_count=0,
                recency_bias=0.0,
            )

        domains = [item.get("domain", "unknown") for item in self.activity_history]
        counts = Counter(domains)
        top_domain, top_count = counts.most_common(1)[0]
        total = max(1, sum(counts.values()))
        dominance = top_count / total

        # Success weighting and recency bias
        successes = [
            1.0 if item.get("success", True) else 0.0
            for item in self.activity_history
            if item.get("domain", "unknown") == top_domain
        ]
        success_rate = sum(successes) / max(1, len(successes))

        recency_bias = self._compute_recency_bias(top_domain)
        expertise_level = min(1.0, 0.6 * dominance + 0.3 * success_rate + 0.1 * recency_bias)
        confidence = min(1.0, 0.5 * dominance + 0.3 * recency_bias + 0.2 * success_rate)

        return SpecializationProfile(
            domain=top_domain,
            dominance=dominance,
            expertise_level=expertise_level,
            confidence=confidence,
            evidence_count=total,
            recency_bias=recency_bias,
        )

    async def evaluate_expertise(self, domain: str) -> ExpertiseEstimate:
        """
        Evaluate expertise level for a given domain based on history.
        """
        if not domain:
            return ExpertiseEstimate(
                domain="",
                proficiency=0.0,
                success_rate=0.0,
                volume=0,
                rationale="No domain specified.",
            )

        domain_activities = [
            item for item in self.activity_history if item.get("domain") == domain
        ]
        volume = len(domain_activities)
        if volume == 0:
            return ExpertiseEstimate(
                domain=domain,
                proficiency=0.0,
                success_rate=0.0,
                volume=0,
                rationale="No activity recorded for this domain.",
            )

        success_rate = sum(1.0 if a.get("success", True) else 0.0 for a in domain_activities) / max(1, volume)
        avg_reward = sum(a.get("reward", 0.0) for a in domain_activities) / max(1, volume)
        recency_bonus = self._compute_recency_bias(domain)

        proficiency = min(1.0, 0.5 * success_rate + 0.3 * recency_bonus + 0.2 * min(1.0, avg_reward))
        rationale = (
            f"Volume={volume}, success_rate={success_rate:.2f}, "
            f"recency={recency_bonus:.2f}, reward={avg_reward:.2f}"
        )

        return ExpertiseEstimate(
            domain=domain,
            proficiency=proficiency,
            success_rate=success_rate,
            volume=volume,
            rationale=rationale,
        )

    async def discover_niche(
        self,
        agent_population: list[UUID],
        population_profiles: list[dict[str, Any]] | None = None,
    ) -> Niche:
        """
        Identify an under-served niche given population specializations.

        population_profiles should contain dicts with {"agent_id", "domain"}.
        """
        population_profiles = population_profiles or []

        # Count how crowded each domain is
        domain_counts = Counter([p.get("domain", "unknown") for p in population_profiles if p])
        if not domain_counts:
            return Niche(
                domain="exploration",
                saturation_score=0.0,
                opportunity=1.0,
                rationale="No population data provided; defaulting to exploration niche.",
            )

        # Find domain with lowest saturation
        least_common = domain_counts.most_common()[:-2:-1]
        domain, count = least_common[0]
        saturation_score = count / max(1, len(agent_population)) if agent_population else 0.0
        opportunity = max(0.0, 1.0 - saturation_score)
        rationale = f"Domain '{domain}' chosen with saturation {saturation_score:.2f} among population."

        return Niche(
            domain=domain,
            saturation_score=saturation_score,
            opportunity=opportunity,
            rationale=rationale,
        )

    def _compute_recency_bias(self, domain: str) -> float:
        """Weight recent activity higher to favor current specialization trends."""
        recent = [
            item for item in self.activity_history[-20:] if item.get("domain", "unknown") == domain
        ]
        total_recent = len(recent)
        total = len(self.activity_history)
        if total == 0:
            return 0.0
        return min(1.0, (total_recent / total) * 2.0)
