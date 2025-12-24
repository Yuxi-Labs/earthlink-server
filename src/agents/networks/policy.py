"""Policy networks - Actor-Critic for decision making."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class PolicyNetwork(nn.Module):
    """
    Policy network for action selection.
    
    Maps state to action distribution (discrete or continuous).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
        num_layers: int = 2,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Shared backbone
        layers = []
        in_dim = state_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        if continuous:
            # Continuous action: output mean and log_std
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # Discrete action: output logits
            self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """
        Forward pass returning action distribution.
        
        Args:
            state: State tensor (batch, state_dim)
            
        Returns:
            Action distribution
        """
        features = self.backbone(state)

        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_log_std.exp().expand_as(mean)
            return Normal(mean, std)
        else:
            logits = self.action_head(features)
            return Categorical(logits=logits)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            (action, log_prob)
        """
        dist = self.forward(state)

        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Sum log_prob for continuous actions
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)

        return action, log_prob

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        
        Returns:
            (log_probs, entropy)
        """
        dist = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        if self.continuous:
            log_probs = log_probs.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        return log_probs, entropy


class ValueNetwork(nn.Module):
    """
    Value network (critic) for state value estimation.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = state_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value."""
        return self.network(state).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network.
    
    Shared backbone with separate heads for policy and value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
        num_layers: int = 2,
        shared_backbone: bool = True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.shared_backbone = shared_backbone

        if shared_backbone:
            # Shared feature extraction
            layers = []
            in_dim = state_dim

            for _ in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ])
                in_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)

            # Actor head
            if continuous:
                self.actor_mean = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            else:
                self.actor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )

            # Critic head
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Separate networks
            self.policy = PolicyNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                continuous=continuous,
                num_layers=num_layers,
            )
            self.value = ValueNetwork(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (action_distribution, state_value)
        """
        if self.shared_backbone:
            features = self.backbone(state)

            if self.continuous:
                mean = self.actor_mean(features)
                std = self.actor_log_std.exp().expand_as(mean)
                action_dist = Normal(mean, std)
            else:
                logits = self.actor(features)
                action_dist = Categorical(logits=logits)

            value = self.critic(features).squeeze(-1)
        else:
            action_dist = self.policy(state)
            value = self.value(state)

        return action_dist, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value.
        
        Returns:
            (action, log_prob, entropy, value)
        """
        action_dist, value = self.forward(state)

        if deterministic:
            if self.continuous:
                action = action_dist.mean
            else:
                action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            (log_probs, entropy, values)
        """
        action_dist, values = self.forward(states)

        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        if self.continuous:
            log_probs = log_probs.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        return log_probs, entropy, values

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get state value only."""
        if self.shared_backbone:
            features = self.backbone(state)
            return self.critic(features).squeeze(-1)
        else:
            return self.value(state)
