"""Tests for agent autonomous_step functionality."""

import pytest

from src.agents.core.agent import Agent
from src.agents.core.decision import Action, UtilityComponents, RiskAssessment, Decision


class StubDecisionModule:
    """Return a fixed decision to avoid side effects in autonomous_step tests."""

    def decide(self, available_actions, goals, world_state, predictions=None, reasoning_context=None):
        action = Action(type="policy", params={}, estimated_cost=0.0)
        utility = UtilityComponents(
            goal_alignment=0.0,
            expected_reward=0.0,
            information_gain=0.0,
            social_value=0.0,
            long_term_value=0.0,
            cost=0.0,
            total=0.0,
        )
        risk = RiskAssessment(
            overall_risk=0.1,
            uncertainty=0.1,
            potential_loss=0.0,
            probability_failure=0.0,
            mitigation_strategies=[],
            risk_factors={},
        )
        return Decision(
            action=action,
            utility=utility,
            risk=risk,
            confidence=1.0,
            rationale="stubbed decision",
            alternatives_considered=len(available_actions),
        )


@pytest.mark.asyncio
async def test_autonomous_step_completes_with_stub_decision(monkeypatch):
    agent = Agent(name="autonomous-test")

    # Use stub decision module and bypass initialization
    agent._decision = StubDecisionModule()
    monkeypatch.setattr(agent, "_ensure_decision", lambda: None)

    action = await agent.autonomous_step()

    assert isinstance(action, dict)
    assert action.get("type") in {"policy", "move", "learn"}  # stub returns policy
    # Ensure metrics incremented
    assert agent.state.metrics.total_steps_executed >= 1

