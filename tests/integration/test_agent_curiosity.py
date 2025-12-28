"""Integration-style test for curiosity-driven action selection."""

import importlib
import sys

import pytest


@pytest.mark.asyncio
async def test_agent_curiosity_gates_actions(monkeypatch):
    """Agent should pick knowledge exploration when signal >= threshold, else policy action."""

    # Patch ray.remote to return the class unchanged so we can instantiate locally
    monkeypatch.setattr("ray.remote", lambda cls: cls)

    # Reload agent module with patched ray.remote
    sys.modules.pop("src.agents.core.agent", None)
    agent_mod = importlib.import_module("src.agents.core.agent")
    Agent = agent_mod.Agent

    agent = Agent(name="Curious", config={"exploration_threshold": 0.5, "action_dim": 4})

    # Case 1: high exploration signal -> explore knowledge
    explore_called = {"val": False}
    step_called = {"val": False}

    async def fake_explore_knowledge():
        explore_called["val"] = True
        return {"knowledge_gained": 0, "sources_used": [], "goal_achieved": False, "topic": "test"}

    def fake_step(observation):
        step_called["val"] = True
        return {"action": "policy"}

    agent._explore_knowledge = fake_explore_knowledge
    agent.step = fake_step
    agent._compute_exploration_signal = lambda: 0.6

    await agent.autonomous_step()

    assert explore_called["val"] is True
    assert step_called["val"] is False

    # Case 2: low exploration signal -> policy action
    explore_called["val"] = False
    step_called["val"] = False
    agent._compute_exploration_signal = lambda: 0.2

    await agent.autonomous_step()

    assert explore_called["val"] is False
    assert step_called["val"] is True
