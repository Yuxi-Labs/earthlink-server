"""Episode management system for tracking agent experiences."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import torch


@dataclass
class Episode:
    """A single episode of agent experience."""
    
    id: UUID = field(default_factory=uuid4)
    agent_id: UUID | None = None
    world_id: str | None = None
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    
    # Episode data
    steps: int = 0
    total_reward: float = 0.0
    
    # Trajectory
    states: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    failure_reason: str | None = None
    
    def add_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        observation: dict[str, Any] | None = None,
    ) -> None:
        """Add a step to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if observation:
            self.observations.append(observation)
        
        self.steps += 1
        self.total_reward += reward
    
    def end(self, success: bool = False, reason: str | None = None) -> None:
        """Mark episode as ended."""
        self.ended_at = datetime.now()
        self.success = success
        self.failure_reason = reason
    
    @property
    def duration(self) -> float:
        """Get episode duration in seconds."""
        if self.ended_at is None:
            return (datetime.now() - self.started_at).total_seconds()
        return (self.ended_at - self.started_at).total_seconds()
    
    @property
    def average_reward(self) -> float:
        """Get average reward per step."""
        if self.steps == 0:
            return 0.0
        return self.total_reward / self.steps
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "world_id": self.world_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "duration": self.duration,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "metadata": self.metadata,
        }


class EpisodeManager:
    """Manages episodes for agents."""
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        
        # Current episodes (agent_id -> Episode)
        self._current_episodes: dict[UUID, Episode] = {}
        
        # Completed episodes history
        self._history: list[Episode] = []
        
        # Statistics
        self._stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "total_steps": 0,
            "total_reward": 0.0,
        }
    
    def start_episode(
        self,
        agent_id: UUID,
        world_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Start a new episode for an agent."""
        # End current episode if exists
        if agent_id in self._current_episodes:
            self.end_episode(agent_id, success=False, reason="new_episode_started")
        
        episode = Episode(
            agent_id=agent_id,
            world_id=world_id,
            metadata=metadata or {},
        )
        
        self._current_episodes[agent_id] = episode
        return episode
    
    def get_current_episode(self, agent_id: UUID) -> Episode | None:
        """Get current episode for an agent."""
        return self._current_episodes.get(agent_id)
    
    def add_step(
        self,
        agent_id: UUID,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        observation: dict[str, Any] | None = None,
    ) -> None:
        """Add a step to agent's current episode."""
        episode = self._current_episodes.get(agent_id)
        if episode is None:
            # Auto-start episode if not exists
            episode = self.start_episode(agent_id)
        
        episode.add_step(state, action, reward, observation)
    
    def end_episode(
        self,
        agent_id: UUID,
        success: bool = False,
        reason: str | None = None,
    ) -> Episode | None:
        """End current episode for an agent."""
        episode = self._current_episodes.pop(agent_id, None)
        if episode is None:
            return None
        
        episode.end(success=success, reason=reason)
        
        # Add to history
        self._history.append(episode)
        if len(self._history) > self.max_episodes:
            self._history = self._history[-self.max_episodes:]
        
        # Update stats
        self._stats["total_episodes"] += 1
        if success:
            self._stats["successful_episodes"] += 1
        else:
            self._stats["failed_episodes"] += 1
        self._stats["total_steps"] += episode.steps
        self._stats["total_reward"] += episode.total_reward
        
        return episode
    
    def get_agent_episodes(
        self,
        agent_id: UUID,
        limit: int = 10,
    ) -> list[Episode]:
        """Get recent episodes for an agent."""
        episodes = [e for e in self._history if e.agent_id == agent_id]
        return episodes[-limit:]
    
    def get_world_episodes(
        self,
        world_id: str,
        limit: int = 10,
    ) -> list[Episode]:
        """Get recent episodes for a world."""
        episodes = [e for e in self._history if e.world_id == world_id]
        return episodes[-limit:]
    
    def get_recent_episodes(self, limit: int = 10) -> list[Episode]:
        """Get most recent episodes."""
        return self._history[-limit:]
    
    def get_best_episodes(
        self,
        limit: int = 10,
        by: str = "reward",
    ) -> list[Episode]:
        """Get best performing episodes."""
        if by == "reward":
            sorted_episodes = sorted(
                self._history,
                key=lambda e: e.total_reward,
                reverse=True,
            )
        elif by == "steps":
            sorted_episodes = sorted(
                self._history,
                key=lambda e: e.steps,
                reverse=True,
            )
        else:
            sorted_episodes = self._history
        
        return sorted_episodes[:limit]
    
    def get_stats(self) -> dict[str, Any]:
        """Get episode statistics."""
        avg_reward = 0.0
        avg_steps = 0.0
        
        if self._stats["total_episodes"] > 0:
            avg_reward = self._stats["total_reward"] / self._stats["total_episodes"]
            avg_steps = self._stats["total_steps"] / self._stats["total_episodes"]
        
        return {
            **self._stats,
            "average_reward_per_episode": avg_reward,
            "average_steps_per_episode": avg_steps,
            "current_episodes": len(self._current_episodes),
            "history_size": len(self._history),
        }
    
    def reset_agent(self, agent_id: UUID) -> None:
        """Reset/clear episodes for an agent."""
        self._current_episodes.pop(agent_id, None)
        self._history = [e for e in self._history if e.agent_id != agent_id]
    
    def clear_all(self) -> None:
        """Clear all episodes."""
        self._current_episodes.clear()
        self._history.clear()
        self._stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "total_steps": 0,
            "total_reward": 0.0,
        }
