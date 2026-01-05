"""Tests for the integrated memory system."""

import torch

from src.agents.memory.memory_system import MemorySystem
from src.agents.memory.memory_system import Transition


def test_short_term_and_consolidation_moves_to_long_term_and_semantic():
    mem = MemorySystem()

    # Add short-term observation
    obs = torch.ones(4)
    mem.add_short_term(obs, source="test")

    # Add episodic transition to feed semantic extraction
    transition = Transition(state={"s": 1}, action={"a": 1}, reward=1.0, next_state={"s": 2})
    mem.store_episode({"state": transition.state, "action": transition.action, "reward": transition.reward, "next_state": transition.next_state})

    # Consolidate and check counts
    result = mem.consolidate()
    assert result["short_to_long"] >= 1
    assert result["episodic_to_semantic"] >= 1

    # Long-term count should reflect added embedding (in-memory fallback OK)
    assert mem.long_term.count() >= 1

    # Semantic graph should have at least one relation
    assert mem.semantic.count_edges() >= 1

