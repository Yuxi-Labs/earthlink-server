"""
Multi-modal perception and attention mechanisms for agents.

Perceive capability: Read and interpret environment, filter based on goals,
build contextual awareness of current state.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np
import torch


class PerceptionModality(str, Enum):
    """Types of perception agents can use."""
    
    SPATIAL = "spatial"  # Geographic location, nearby features
    SOCIAL = "social"  # Other agents in vicinity
    TEMPORAL = "temporal"  # Time-based patterns
    KNOWLEDGE = "knowledge"  # Information/data available
    SELF = "self"  # Internal state monitoring
    VISUAL = "visual"  # Images or visual descriptors when available


@dataclass
class PerceptionResult:
    """Result of perception operation."""
    
    modality: PerceptionModality
    data: dict[str, Any]
    confidence: float  # 0-1, how confident in this perception
    attention_score: float  # 0-1, how relevant to current goals
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "modality": self.modality.value,
            "data": self.data,
            "confidence": self.confidence,
            "attention_score": self.attention_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AttentionFocus:
    """What agent should pay attention to."""
    
    modalities: list[PerceptionModality]  # Which senses to use
    keywords: list[str]  # What to look for
    priority: float  # 0-1, importance
    goal_id: UUID | None = None  # Associated goal


class PerceptionModule:
    """
    Multi-modal environment perception with attention mechanisms.
    
    Capabilities:
    - Spatial perception: Location, nearby features, geography
    - Social perception: Other agents in vicinity
    - Temporal perception: Time patterns, schedules
    - Knowledge perception: Available information/data
    - Self perception: Internal state monitoring
    
    Uses attention mechanisms to filter perception based on current goals.
    """
    
    def __init__(
        self,
        agent_id: UUID,
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.device = device
        
        # Attention weights (learned over time)
        self.attention_weights: dict[PerceptionModality, float] = {
            modality: 0.2 for modality in PerceptionModality
        }
        
        # Perception history (for temporal patterns)
        self.perception_history: list[PerceptionResult] = []
        self.max_history = 100
    
    async def perceive(
        self,
        environment: dict[str, Any],
        attention_focus: AttentionFocus | None = None,
    ) -> list[PerceptionResult]:
        """
        Perceive environment using multiple modalities.
        
        Args:
            environment: Current environment state (location, agents, data, etc.)
            attention_focus: What to focus on (if None, perceive everything)
        
        Returns:
            List of perception results, filtered by attention
        """
        results = []
        
        # Determine which modalities to use
        if attention_focus:
            modalities = attention_focus.modalities
        else:
            modalities = list(PerceptionModality)
        
        # Perceive each modality
        for modality in modalities:
            if modality == PerceptionModality.SPATIAL:
                result = await self._perceive_spatial(environment)
            elif modality == PerceptionModality.SOCIAL:
                result = await self._perceive_social(environment)
            elif modality == PerceptionModality.TEMPORAL:
                result = await self._perceive_temporal(environment)
            elif modality == PerceptionModality.KNOWLEDGE:
                result = await self._perceive_knowledge(environment)
            elif modality == PerceptionModality.SELF:
                result = await self._perceive_self(environment)
            elif modality == PerceptionModality.VISUAL:
                result = await self._perceive_visual(environment)
            else:
                continue
            
            # Apply attention filter
            if attention_focus:
                result.attention_score = self._calculate_attention_score(
                    result, attention_focus
                )
            else:
                result.attention_score = self.attention_weights[modality]
            
            results.append(result)
        
        # Sort by attention score (most relevant first)
        results.sort(key=lambda r: r.attention_score, reverse=True)
        
        # Update history
        self.perception_history.extend(results)
        if len(self.perception_history) > self.max_history:
            self.perception_history = self.perception_history[-self.max_history:]
        
        return results

    def filter_perception(
        self,
        raw_results: list[PerceptionResult],
        attention_focus: AttentionFocus | None = None,
        attention_threshold: float = 0.25,
        confidence_threshold: float = 0.3,
    ) -> tuple[list[PerceptionResult], float]:
        """
        Filter perceptions based on attention focus and confidence.
        
        Returns filtered perceptions and estimated noise reduction ratio.
        """
        if not raw_results:
            return [], 0.0

        filtered = []
        for r in raw_results:
            if attention_focus and attention_focus.goal_id:
                # Boost attention score for goal-linked perceptions
                r.attention_score = max(r.attention_score, attention_focus.priority * 0.5)
            if r.attention_score >= attention_threshold and r.confidence >= confidence_threshold:
                filtered.append(r)

        noise_reduction = 1.0 - (len(filtered) / max(len(raw_results), 1))
        return filtered, max(0.0, min(1.0, noise_reduction))
    
    async def _perceive_spatial(
        self,
        environment: dict[str, Any],
    ) -> PerceptionResult:
        """Perceive spatial information: location, nearby features."""
        location = environment.get("location", {})
        nearby_features = environment.get("nearby_features", [])
        containing_regions = environment.get("containing_regions", [])
        
        data = {
            "latitude": location.get("x", 0.0),
            "longitude": location.get("y", 0.0),
            "altitude": location.get("z", 0.0),
            "nearby_features": nearby_features[:10],  # Top 10 nearest
            "containing_regions": containing_regions[:5],  # Top 5 regions
            "feature_count": len(nearby_features),
        }
        
        # Confidence based on data availability
        confidence = 0.5
        if nearby_features:
            confidence += 0.3
        if containing_regions:
            confidence += 0.2
        
        return PerceptionResult(
            modality=PerceptionModality.SPATIAL,
            data=data,
            confidence=min(1.0, confidence),
            attention_score=0.0,  # Will be set by attention mechanism
        )
    
    async def _perceive_social(
        self,
        environment: dict[str, Any],
    ) -> PerceptionResult:
        """Perceive social information: other agents nearby."""
        nearby_agents = environment.get("nearby_agents", [])
        
        data = {
            "agent_count": len(nearby_agents),
            "agents": nearby_agents[:20],  # Top 20 nearest
            "has_agents": len(nearby_agents) > 0,
        }
        
        confidence = 1.0 if "nearby_agents" in environment else 0.3
        
        return PerceptionResult(
            modality=PerceptionModality.SOCIAL,
            data=data,
            confidence=confidence,
            attention_score=0.0,
        )
    
    async def _perceive_temporal(
        self,
        environment: dict[str, Any],
    ) -> PerceptionResult:
        """Perceive temporal patterns: time-based information."""
        current_time = datetime.now(UTC)
        
        # Detect patterns in perception history
        patterns = self._detect_temporal_patterns()
        
        data = {
            "current_time": current_time.isoformat(),
            "hour_of_day": current_time.hour,
            "day_of_week": current_time.weekday(),
            "patterns_detected": len(patterns),
            "patterns": patterns[:5],  # Top 5 patterns
        }
        
        confidence = 0.6 if patterns else 0.4
        
        return PerceptionResult(
            modality=PerceptionModality.TEMPORAL,
            data=data,
            confidence=confidence,
            attention_score=0.0,
        )
    
    async def _perceive_knowledge(
        self,
        environment: dict[str, Any],
    ) -> PerceptionResult:
        """Perceive available knowledge/information."""
        knowledge_sources = environment.get("knowledge_sources", [])
        available_topics = environment.get("available_topics", [])
        
        data = {
            "sources": knowledge_sources,
            "topics": available_topics[:20],  # Top 20 topics
            "source_count": len(knowledge_sources),
            "topic_count": len(available_topics),
        }
        
        confidence = 0.7 if knowledge_sources else 0.3
        
        return PerceptionResult(
            modality=PerceptionModality.KNOWLEDGE,
            data=data,
            confidence=confidence,
            attention_score=0.0,
        )
    
    async def _perceive_self(
        self,
        environment: dict[str, Any],
    ) -> PerceptionResult:
        """Perceive internal state."""
        agent_state = environment.get("agent_state", {})
        
        data = {
            "status": agent_state.get("status", "unknown"),
            "lifecycle": agent_state.get("lifecycle", "unknown"),
            "current_goal": agent_state.get("current_goal_id"),
            "metrics": agent_state.get("metrics", {}),
        }
        
        confidence = 1.0  # Always confident about own state
        
        return PerceptionResult(
            modality=PerceptionModality.SELF,
            data=data,
            confidence=confidence,
            attention_score=0.0,
        )

    async def _perceive_visual(
        self,
        environment: dict[str, Any],
    ) -> PerceptionResult:
        """Perceive visual information when provided in environment."""
        visual = environment.get("visual_data")
        summary = {}
        if isinstance(visual, dict):
            summary = visual
        elif isinstance(visual, str):
            summary = {"description": visual[:256]}

        confidence = 0.4 if summary else 0.1
        return PerceptionResult(
            modality=PerceptionModality.VISUAL,
            data=summary,
            confidence=confidence,
            attention_score=0.0,
        )
    
    def _calculate_attention_score(
        self,
        perception: PerceptionResult,
        focus: AttentionFocus,
    ) -> float:
        """
        Calculate how relevant this perception is to current attention focus.
        
        Uses keyword matching and modality weighting.
        """
        base_score = self.attention_weights[perception.modality]
        
        # Keyword matching bonus
        keyword_bonus = 0.0
        if focus.keywords:
            perception_text = str(perception.data).lower()
            matches = sum(
                1 for keyword in focus.keywords 
                if keyword.lower() in perception_text
            )
            keyword_bonus = min(0.4, matches * 0.1)
        
        # Priority boost
        priority_boost = focus.priority * 0.2
        
        # Confidence penalty (don't focus on uncertain perceptions)
        confidence_factor = perception.confidence
        
        final_score = (base_score + keyword_bonus + priority_boost) * confidence_factor
        
        return min(1.0, final_score)
    
    def _detect_temporal_patterns(self) -> list[dict[str, Any]]:
        """Detect repeating patterns in perception history."""
        if len(self.perception_history) < 10:
            return []
        
        patterns = []
        
        # Simple pattern detection: repeating modality sequences
        recent = self.perception_history[-20:]
        modality_sequence = [p.modality for p in recent]
        
        # Find most common modality
        modality_counts = {}
        for modality in modality_sequence:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        if modality_counts:
            most_common = max(modality_counts.items(), key=lambda x: x[1])
            patterns.append({
                "type": "modality_frequency",
                "modality": most_common[0].value,
                "count": most_common[1],
                "frequency": most_common[1] / len(recent),
            })
        
        return patterns
    
    def update_attention_weights(
        self,
        modality: PerceptionModality,
        reward: float,
    ):
        """
        Update attention weights based on feedback.
        
        If perceiving a modality led to reward, increase its weight.
        """
        learning_rate = 0.1
        current = self.attention_weights[modality]
        
        # Move weight toward reward signal
        self.attention_weights[modality] = current + learning_rate * (reward - current)
        
        # Normalize to ensure sum = 1.0
        total = sum(self.attention_weights.values())
        if total > 0:
            for m in PerceptionModality:
                self.attention_weights[m] /= total
    
    def get_contextual_awareness(self) -> dict[str, Any]:
        """
        Build contextual awareness summary from recent perceptions.
        
        Returns high-level understanding of current situation.
        """
        if not self.perception_history:
            return {"awareness_level": "none", "context": {}}
        
        recent = self.perception_history[-10:]
        
        # Aggregate by modality
        by_modality = {}
        for perception in recent:
            modality = perception.modality.value
            if modality not in by_modality:
                by_modality[modality] = []
            by_modality[modality].append(perception)
        
        # Build context summary
        context = {
            "awareness_level": "high" if len(recent) >= 5 else "moderate",
            "modalities_active": list(by_modality.keys()),
            "recent_perceptions": len(recent),
            "average_confidence": np.mean([p.confidence for p in recent]),
            "summary_by_modality": {},
        }
        
        # Summarize each modality
        for modality, perceptions in by_modality.items():
            latest = perceptions[-1]
            context["summary_by_modality"][modality] = {
                "latest_data": latest.data,
                "confidence": latest.confidence,
                "perception_count": len(perceptions),
            }
        
        return context
