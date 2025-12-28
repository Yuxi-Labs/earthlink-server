"""Integration-style test for agent autonomous learning pipeline."""

import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_agent_autonomous_learning(monkeypatch):
    """
    Verify that an agent performing knowledge exploration:
    - logs knowledge acquisition
    - records combined rewards
    - increments training metrics
    """

    # Patch ray.remote to return the class unchanged so we can instantiate locally
    monkeypatch.setattr("ray.remote", lambda cls: cls)

    # Reload agent module with patched ray.remote
    sys.modules.pop("src.agents.core.agent", None)
    agent_mod = importlib.import_module("src.agents.core.agent")
    Agent = agent_mod.Agent

    # Create agent with low exploration threshold
    agent = Agent(name="TestAgent", config={"exploration_threshold": 0.0, "action_dim": 4})

    # Stub memory with minimal async methods
    episodes: list = []

    async def fake_store_semantic(content: str, metadata: dict | None = None):
        episodes.append({"semantic": content, "metadata": metadata})
        return "semantic-id"

    def fake_store_episode(transition):
        episodes.append({"episode": transition})

    def fake_sample_for_learning(batch_size: int = 32, device=None):
        return None  # skip actual network updates

    agent._memory = SimpleNamespace(
        store_semantic=fake_store_semantic,
        store_episode=fake_store_episode,
        sample_for_learning=fake_sample_for_learning,
    )

    # Track knowledge log calls
    log_calls = {}

    async def fake_log_knowledge_acquisition(topic, sources, knowledge_count, content_summary):
        log_calls["topic"] = topic
        log_calls["sources"] = sources
        log_calls["knowledge_count"] = knowledge_count
        log_calls["content_summary"] = content_summary

    agent._log_knowledge_acquisition = fake_log_knowledge_acquisition

    # Make exploration deterministic
    async def fake_explore_topic(topic: str):
        sources = {"wikipedia": {"results": [1, 2, 3]}}
        await agent._log_knowledge_acquisition(
            topic=topic,
            sources=list(sources.keys()),
            knowledge_count=3,
            content_summary="stubbed",
        )
        return {"sources": sources, "knowledge_gained": 3}

    agent.explore_topic = fake_explore_topic

    async def fake_sample_exploration_topic():
        return "test topic"

    agent._sample_exploration_topic = fake_sample_exploration_topic

    # Stub learn to increment training metrics
    def fake_learn(transition):
        agent.state.metrics.training_steps += 1
        agent.state.metrics.last_training_loss = 0.123
        return {"loss": 0.123}

    agent.learn = fake_learn

    # Run knowledge exploration
    result = await agent._explore_knowledge()

    # Assertions
    assert log_calls["knowledge_count"] == 3
    assert agent.state.metrics.knowledge_acquired >= 3
    assert agent.state.metrics.training_steps > 0
    assert result["knowledge_gained"] == 3
    assert result["topic"] == "test topic"
