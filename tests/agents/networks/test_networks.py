"""Tests for neural network components."""

import torch

from src.agents.networks.policy import PolicyNetwork, ActorCritic
from src.agents.networks.world_model import WorldModel
from src.agents.networks.curiosity import ICM


def test_policy_network_discrete_forward_and_action():
    policy = PolicyNetwork(state_dim=8, action_dim=4, hidden_dim=16, continuous=False)
    state = torch.randn(2, 8)
    dist = policy(state)
    action, log_prob = policy.get_action(state)
    assert dist.probs.shape == (2, 4)
    assert action.shape == (2,)
    assert log_prob.shape == (2,)


def test_actor_critic_forward():
    ac = ActorCritic(state_dim=10, action_dim=5, hidden_dim=16, continuous=False)
    state = torch.randn(3, 10)
    action, log_prob, entropy, value = ac.evaluate_actions(state, torch.tensor([1, 2, 3]))
    assert action.shape == (3,)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3,)
    assert value.shape == (3,)


def test_world_model_predict_and_loss():
    wm = WorldModel(state_dim=6, action_dim=3, hidden_dim=12)
    states = torch.randn(4, 6)
    actions = torch.randn(4, 3)
    preds = wm(states, actions)
    assert preds.shape == (4, 6)
    next_states = torch.randn(4, 6)
    rewards = torch.randn(4, 1)
    dones = torch.zeros(4, 1)
    loss = wm.compute_loss(states, actions, next_states, rewards, dones)
    assert "total_loss" in loss


def test_icm_forward_and_loss():
    icm = ICM(state_dim=5, action_dim=2, hidden_dim=8)
    states = torch.randn(4, 5)
    actions = torch.randn(4, 2)
    next_states = torch.randn(4, 5)
    intrinsic_reward = icm.compute_intrinsic_reward(states, actions, next_states)
    assert intrinsic_reward.shape == (4, 1)
    loss = icm.update(states, actions, next_states)
    assert "total_loss" in loss

