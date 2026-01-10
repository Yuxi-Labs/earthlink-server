"""
Unit tests for archetype detection logic.

These tests exercise the archetype scoring implemented in `Agent.get_exploration_archetype`
without spinning up Ray actors or the full simulation runner (which is too heavy for CI).
"""

import pytest

from src.agents.core.agent import Agent


class TestArchetypeDetection:
    """Test archetype detection based on agent parameters (pure unit tests)."""

    def _archetype(self, config):
        agent = Agent(name="Test", config=config)
        result = agent.get_exploration_archetype()
        return result, agent.config

    def test_systematic_mapper_agent(self):
        config = {
            "exploration_style": 0.2,
            "depth_vs_breadth": 0.3,
            "path_memory_strength": 0.9,
            "risk_tolerance": 0.3,
            "curiosity": 0.6,
            "backtracking_tolerance": 0.5,
            "social_priority": 0.4,
            "goal_persistence": 0.7,
            "novelty_seeking": 0.5,
        }
        result, _ = self._archetype(config)
        assert result["archetype"] == "Systematic Mapper"

    def test_bold_pioneer_agent(self):
        config = {
            "exploration_style": 0.9,
            "risk_tolerance": 0.9,
            "curiosity": 0.95,
            "path_memory_strength": 0.2,
            "backtracking_tolerance": 0.1,
            "exploration_bonus": 0.2,
            "goal_flexibility": 0.9,
            "collaborative_exploration": 0.1,
            "adaptation_speed": 0.3,
            "strategy_stickiness": 0.2,
        }
        result, _ = self._archetype(config)
        assert result["archetype"] == "Bold Pioneer"

    def test_archetype_has_all_scores(self):
        result, _ = self._archetype({"exploration_style": 0.5})
        scores = result["scores"]
        assert isinstance(scores, dict)
        assert len(scores) > 0
        assert all(0 <= v <= 1 for v in scores.values())
        assert isinstance(result["archetype"], str) and result["archetype"]

    def test_archetype_returns_parameters(self):
        result, params = self._archetype({"exploration_style": 0.7, "risk_tolerance": 0.6})
        assert params["exploration_style"] == 0.7
        assert params["risk_tolerance"] == 0.6
        assert result["parameters"]["exploration_style"] == 0.7


class TestCoverageAssessment:
    def test_coverage_calculation(self):
        agent = Agent(name="Coverage")
        coverage = agent.assess_exploration_coverage()
        assert isinstance(coverage, dict)
        assert "unique_locations" in coverage

    def test_visited_locations_tracking(self):
        agent = Agent(name="Visited")
        coverage = agent.assess_exploration_coverage()
        assert coverage["unique_locations"] == 0


class TestParameterVariation:
    def test_specified_parameters_preserved(self):
        agent = Agent(name="Params", config={"exploration_style": 0.123, "curiosity": 0.456})
        result = agent.get_exploration_archetype()
        assert result["parameters"]["exploration_style"] == 0.123
        assert result["parameters"]["curiosity"] == 0.456


class TestArchetypeAPI:
    def test_list_archetypes_contains_labels(self):
        agent = Agent(name="API")
        result = agent.get_exploration_archetype()
        assert "scores" in result and len(result["scores"]) > 0


class TestDestinationSelection:
    def test_archetype_influences_destination_placeholder(self):
        # Placeholder to keep parity with original suite; real destination logic depends on world
        agent = Agent(name="Dest", config={"exploration_style": 0.2})
        result = agent.get_exploration_archetype()
        assert result["archetype"]
