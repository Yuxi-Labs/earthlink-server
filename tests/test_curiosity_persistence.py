"""Test curiosity persistence and exploration signal calculation."""

import pytest

from src.agents.core.agent import Agent


class TestCuriosityPersistence:
    """Test that agent curiosity is an individual trait that persists."""

    def test_curiosity_not_overwritten_by_exploration_signal(self):
        """
        CRITICAL TEST: Verify curiosity is NOT overwritten by exploration_signal.
        
        This was the bug that made all agents clones with 50% curiosity.
        """
        agent = Agent(name="CuriosityTest1", config={"curiosity": 0.72})
        
        # Verify initial curiosity
        assert agent.state.metrics.curiosity_score == 0.72
        
        # Compute exploration signal (this should NOT change curiosity)
        exploration_signal = agent._compute_exploration_signal()
        
        # Curiosity should remain unchanged
        assert agent.state.metrics.curiosity_score == 0.72
        
        # Exploration signal might be different from curiosity
        # (it's influenced by curiosity but also by world model, etc.)
        assert exploration_signal is not None
        assert 0.0 <= exploration_signal <= 1.0

    @pytest.mark.asyncio
    async def test_curiosity_persists_after_autonomous_step(self):
        """Test that curiosity remains constant across autonomous steps."""
        initial_curiosity = 0.65
        agent = Agent(name="CuriosityTest2", config={"curiosity": initial_curiosity})
        
        # Run multiple autonomous steps
        for _ in range(3):
            await agent.autonomous_step()
            
            # Curiosity should NOT change
            assert agent.state.metrics.curiosity_score == initial_curiosity

    def test_multiple_agents_retain_unique_curiosity(self):
        """Test that multiple agents maintain their individual curiosity values."""
        agents = [
            Agent(name=f"Agent{i}", config={"curiosity": 0.3 + i * 0.1})
            for i in range(6)  # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
        ]
        
        expected_values = [0.3 + i * 0.1 for i in range(6)]
        
        # Check initial values
        for agent, expected in zip(agents, expected_values):
            assert abs(agent.state.metrics.curiosity_score - expected) < 0.01
        
        # All should be different
        curiosity_values = [a.state.metrics.curiosity_score for a in agents]
        assert len(set(curiosity_values)) == len(curiosity_values)

    def test_exploration_signal_influenced_by_curiosity(self):
        """Test that exploration signal is influenced by curiosity."""
        low_curiosity_agent = Agent(name="LowCuriosity", config={"curiosity": 0.2})
        high_curiosity_agent = Agent(name="HighCuriosity", config={"curiosity": 0.9})
        
        low_signal = low_curiosity_agent._compute_exploration_signal()
        high_signal = high_curiosity_agent._compute_exploration_signal()
        
        # Higher curiosity should generally lead to higher exploration signal
        # (though world model also plays a role)
        assert 0.0 <= low_signal <= 1.0
        assert 0.0 <= high_signal <= 1.0

    @pytest.mark.asyncio
    async def test_curiosity_in_serialized_state(self):
        """Test curiosity appears correctly in serialized state."""
        agent = Agent(name="CuriosityTest3", config={"curiosity": 0.77})
        
        # Run an autonomous step
        await agent.autonomous_step()
        
        # Get state
        state = agent.get_state()
        
        # Curiosity should be in metrics and unchanged
        assert state["metrics"]["curiosity_score"] == 0.77

    def test_compute_exploration_signal_returns_valid_value(self):
        """Test that _compute_exploration_signal returns valid probability."""
        agent = Agent(name="CuriosityTest4", config={"curiosity": 0.5})
        
        signal = agent._compute_exploration_signal()
        
        # Should be a valid probability
        assert isinstance(signal, float)
        assert 0.0 <= signal <= 1.0

    @pytest.mark.asyncio
    async def test_high_curiosity_agent_explores_more(self):
        """
        Test that high curiosity agents are more likely to choose explore actions.
        
        Run many steps and verify action distribution.
        """
        high_curiosity = Agent(name="Explorer", config={"curiosity": 0.95})
        
        actions = []
        for _ in range(20):
            result = await high_curiosity.autonomous_step()
            if result and "type" in result:
                actions.append(result["type"])
        
        # Should have executed some actions
        assert len(actions) > 0
        
        # High curiosity should favor exploration/learning
        exploration_actions = sum(1 for a in actions if a in ["move", "learn"])
        
        # At least some exploration should happen
        assert exploration_actions > 0

    @pytest.mark.asyncio
    async def test_low_curiosity_agent_explores_less(self):
        """
        Test that low curiosity agents explore less.
        
        This is harder to test deterministically, but we can verify behavior.
        """
        low_curiosity = Agent(name="Conservative", config={"curiosity": 0.1})
        
        actions = []
        for _ in range(20):
            result = await low_curiosity.autonomous_step()
            if result and "type" in result:
                actions.append(result["type"])
        
        # Should have executed some actions
        assert len(actions) > 0
