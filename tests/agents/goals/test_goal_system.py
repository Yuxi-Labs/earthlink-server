"""Tests for the goal system."""

import torch

from src.agents.goals.goal_system import GoalSystem
from src.agents.goals.goal import GoalType


def test_goal_system_propose_and_value():
    gs = GoalSystem(state_dim=16, goal_dim=8, hidden_dim=16, device="cpu")
    state = torch.randn(1, 16)

    proposals = gs.goal_proposer(state)
    assert proposals.shape == (1, gs.goal_proposer.num_proposals, gs.goal_proposer.goal_dim)

    values = gs.goal_value(state, proposals.squeeze(0))
    assert values.shape == (gs.goal_proposer.num_proposals,)


def test_goal_system_add_and_get_goal():
    gs = GoalSystem(state_dim=8, goal_dim=4, hidden_dim=8, device="cpu")
    goal = gs.add_goal(
        description="Explore new area",
        goal_type=GoalType.EXPLORATION,
        priority=0.7,
    )
    fetched = gs.get_goal(goal.id)
    assert fetched is not None
    assert fetched.description == "Explore new area"


def test_goal_state_updates():
    gs = GoalSystem(state_dim=8, goal_dim=4, hidden_dim=8, device="cpu")
    goal = gs.add_goal(description="Test goal", goal_type=GoalType.EXPLORATION, priority=0.5)
    gs.update_goal_status(goal.id, status="in_progress")
    assert gs.get_goal(goal.id).status.value == "in_progress"
    gs.update_goal_progress(goal.id, progress=0.4)
    assert gs.get_goal(goal.id).progress == 0.4

