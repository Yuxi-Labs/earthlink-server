"""Short-term memory - recent observations, working memory buffer."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch


@dataclass
class Observation:
    """Single observation with metadata."""

    tensor: torch.Tensor
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "environment"
    metadata: dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """
    Short-term memory buffer.
    
    Stores recent observations in a fixed-size deque.
    Used for working memory and immediate context.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: deque[Observation] = deque(maxlen=capacity)

    def add(self, observation: torch.Tensor, source: str = "environment", **metadata: Any) -> None:
        """Add observation to short-term memory."""
        obs = Observation(
            tensor=observation.detach().cpu(),
            source=source,
            metadata=metadata,
        )
        self.buffer.append(obs)

    def get_recent(self, n: int = 10) -> list[Observation]:
        """Get n most recent observations."""
        return list(self.buffer)[-n:]

    def get_context_tensor(self, n: int = 10) -> torch.Tensor:
        """Get recent observations as stacked tensor."""
        recent = self.get_recent(n)
        if not recent:
            return torch.zeros(n, 64)  # Default size
        
        tensors = [obs.tensor for obs in recent]
        
        # Pad if needed
        while len(tensors) < n:
            tensors.insert(0, torch.zeros_like(tensors[0]))
        
        return torch.stack(tensors)

    def clear(self) -> None:
        """Clear all short-term memory."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
