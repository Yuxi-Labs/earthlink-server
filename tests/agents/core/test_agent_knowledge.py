"""Tests for agent knowledge query functionality."""

import asyncio
from typing import Any

import pytest

from src.agents.core.agent import Agent


@pytest.mark.asyncio
async def test_agent_explore_topic_returns_sources(monkeypatch):
    agent = Agent.options(name="agent_knowledge_test", config={})._remote()  # type: ignore

    # Mock knowledge calls to avoid real network
    async def mock_wikipedia(query: str, **kwargs) -> dict[str, Any]:
        return {"title": query, "summary": "summary", "url": "http://example.com"}

    async def mock_duckduckgo(query: str, **kwargs) -> dict[str, Any]:
        return {"results": [{"title": query, "url": "http://example.com"}]}

    async def mock_ollama(prompt: str, **kwargs) -> dict[str, Any]:
        return {"response": f"answer for {prompt}"}

    async def mock_geo(lat: float, lon: float, radius_meters: float = 1000) -> dict[str, Any]:
        return {"nearby_features": [], "containing_regions": [{"name": "region"}]}

    agent.query_wikipedia = mock_wikipedia  # type: ignore
    agent.query_search = mock_duckduckgo  # type: ignore
    agent.query_ollama = mock_ollama  # type: ignore
    agent.query_geo = mock_geo  # type: ignore

    result = await agent.explore_topic("test topic")
    assert result["knowledge_gained"] >= 1
    assert "wikipedia" in result["sources"]
    assert "search" in result["sources"]
    assert "ollama" in result["sources"]


@pytest.mark.asyncio
async def test_agent_query_methods_handles_errors(monkeypatch):
    agent = Agent.options(name="agent_knowledge_test_error", config={})._remote()  # type: ignore

    # Force error in wikipedia query
    async def failing_wikipedia(query: str, **kwargs) -> dict[str, Any]:
        raise RuntimeError("fail")

    agent.query_wikipedia = failing_wikipedia  # type: ignore
    agent.query_search = failing_wikipedia  # type: ignore

    result = await agent.explore_topic("fallback topic")
    # Should still return structure even when sources fail
    assert "sources" in result
    assert isinstance(result["knowledge_gained"], int)

