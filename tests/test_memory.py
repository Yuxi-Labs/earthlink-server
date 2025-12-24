"""Tests for memory systems."""

import pytest
from uuid import uuid4
import numpy as np

from src.agents.memory import ShortTermMemory, LongTermMemory, EpisodicMemory


def test_short_term_memory():
    """Test short-term memory (working memory)."""
    memory = ShortTermMemory(capacity=5)
    
    # Add observations
    for i in range(7):
        memory.add({"step": i, "data": f"obs_{i}"})
    
    # Should only keep last 5
    assert len(memory.memory) == 5
    assert memory.memory[0]["step"] == 2
    assert memory.memory[-1]["step"] == 6


def test_short_term_memory_recent():
    """Test getting recent observations."""
    memory = ShortTermMemory(capacity=10)
    
    for i in range(10):
        memory.add({"step": i})
    
    recent = memory.get_recent(3)
    assert len(recent) == 3
    assert recent[0]["step"] == 7
    assert recent[-1]["step"] == 9


@pytest.mark.asyncio
async def test_episodic_memory():
    """Test episodic memory (replay buffer)."""
    memory = EpisodicMemory(capacity=100)
    
    # Add transitions
    for i in range(50):
        memory.add(
            state={"pos": i},
            action={"move": 1},
            reward=1.0,
            next_state={"pos": i + 1},
            done=False,
        )
    
    assert len(memory.buffer) == 50
    
    # Sample batch
    batch = memory.sample(10)
    assert len(batch) == 10
    assert "state" in batch[0]
    assert "action" in batch[0]
    assert "reward" in batch[0]


def test_episodic_memory_capacity():
    """Test episodic memory capacity limit."""
    memory = EpisodicMemory(capacity=10)
    
    # Add more than capacity
    for i in range(20):
        memory.add(
            state={"step": i},
            action={},
            reward=0.0,
            next_state={"step": i + 1},
            done=False,
        )
    
    # Should only keep 10
    assert len(memory.buffer) == 10
