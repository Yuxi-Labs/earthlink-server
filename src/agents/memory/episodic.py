"""Episodic memory - experience replay buffer for learning from past actions."""

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Transition:
    """Single transition (s, a, r, s', done)."""

    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool = False
    goal: torch.Tensor | None = None
    info: dict[str, Any] = field(default_factory=dict)

    # Priority for prioritized experience replay
    priority: float = 1.0


class EpisodicMemory:
    """
    Episodic memory for experience replay.
    
    Supports both uniform and prioritized experience replay.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        prioritized: bool = True,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta

        self.buffer: list[Transition] = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, transition: Transition) -> None:
        """Add transition to replay buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        # Set max priority for new transitions
        self.priorities[self.position] = self.max_priority
        transition.priority = self.max_priority

        self.position = (self.position + 1) % self.capacity

    def push_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        goal: torch.Tensor | None = None,
        **info: Any,
    ) -> None:
        """Convenience method to push individual components."""
        transition = Transition(
            state=state.detach().cpu(),
            action=action.detach().cpu() if isinstance(action, torch.Tensor) else torch.tensor(action),
            reward=reward,
            next_state=next_state.detach().cpu(),
            done=done,
            goal=goal.detach().cpu() if goal is not None else None,
            info=info,
        )
        self.push(transition)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch of transitions.
        
        Returns (transitions, indices, importance_weights).
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if self.prioritized:
            return self._prioritized_sample(batch_size)
        else:
            return self._uniform_sample(batch_size)

    def _uniform_sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Uniform random sampling."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return transitions, indices, weights

    def _prioritized_sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Prioritized experience replay sampling."""
        priorities = self.priorities[: len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        transitions = [self.buffer[i] for i in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

            if idx < len(self.buffer):
                self.buffer[idx].priority = priority

    def to_batch_tensors(
        self,
        transitions: list[Transition],
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, torch.Tensor]:
        """Convert transitions to batched tensors."""
        return {
            "states": torch.stack([t.state for t in transitions]).to(device),
            "actions": torch.stack([t.action for t in transitions]).to(device),
            "rewards": torch.tensor([t.reward for t in transitions], dtype=torch.float32).to(device),
            "next_states": torch.stack([t.next_state for t in transitions]).to(device),
            "dones": torch.tensor([t.done for t in transitions], dtype=torch.float32).to(device),
            "goals": torch.stack([t.goal for t in transitions]).to(device) if transitions[0].goal is not None else None,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear replay buffer."""
        self.buffer.clear()
        self.position = 0
        self.priorities.fill(0)
        self.max_priority = 1.0
