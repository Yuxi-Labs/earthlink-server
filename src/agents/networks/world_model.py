"""World model - predictive model for state transitions."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """
    World model for predicting state transitions.
    
    Learns dynamics: s' = f(s, a)
    Also predicts reward: r = g(s, a, s')
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        probabilistic: bool = True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.probabilistic = probabilistic

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Action encoder (handles both discrete and continuous)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim // 2),
        )

        # Transition model
        transition_input_dim = latent_dim + latent_dim // 2

        layers = []
        in_dim = transition_input_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        self.transition_backbone = nn.Sequential(*layers)

        if probabilistic:
            # Output mean and variance for probabilistic prediction
            self.next_state_mean = nn.Linear(hidden_dim, state_dim)
            self.next_state_log_var = nn.Linear(hidden_dim, state_dim)
        else:
            # Deterministic prediction
            self.next_state_predictor = nn.Linear(hidden_dim, state_dim)

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Done predictor (episode termination)
        self.done_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent space."""
        return self.state_encoder(state)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Encode action to latent space."""
        # Convert discrete actions to one-hot if needed
        if action.dim() == 1 or action.shape[-1] == 1:
            # Assume discrete, convert to one-hot
            if action.dtype in (torch.long, torch.int):
                action_one_hot = F.one_hot(
                    action.long(),
                    num_classes=self.action_dim,
                ).float()
            else:
                # Continuous 1D action
                action_one_hot = action.float()
                if action_one_hot.dim() == 1:
                    action_one_hot = action_one_hot.unsqueeze(-1)
        else:
            action_one_hot = action.float()

        return self.action_encoder(action_one_hot)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        return_full: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Predict next state, reward, and done.
        
        Returns dict with:
            - next_state_mean: Predicted next state mean
            - next_state_log_var: Log variance (if probabilistic)
            - reward: Predicted reward
            - done_prob: Probability of episode termination
        """
        # Encode state and action
        state_latent = self.encode_state(state)
        action_latent = self.encode_action(action)

        # Combine for transition prediction
        combined = torch.cat([state_latent, action_latent], dim=-1)
        features = self.transition_backbone(combined)

        # Predict next state
        if self.probabilistic:
            next_state_mean = self.next_state_mean(features)
            next_state_log_var = self.next_state_log_var(features)
        else:
            next_state_mean = self.next_state_predictor(features)
            next_state_log_var = None

        # Predict reward and done
        reward = self.reward_predictor(features)
        done_prob = self.done_predictor(features)

        result = {
            "next_state_mean": next_state_mean,
            "reward": reward,
            "done_prob": done_prob,
        }

        if self.probabilistic:
            result["next_state_log_var"] = next_state_log_var

        return result if return_full else next_state_mean

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """Predict next state only."""
        outputs = self.forward(state, action, return_full=True)

        if self.probabilistic and sample:
            # Sample from predicted distribution
            mean = outputs["next_state_mean"]
            std = (outputs["next_state_log_var"] * 0.5).exp()
            return mean + std * torch.randn_like(std)
        else:
            return outputs["next_state_mean"]

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute world model loss.
        
        Returns dict with:
            - total_loss: Combined loss
            - state_loss: Next state prediction loss
            - reward_loss: Reward prediction loss
            - done_loss: Done prediction loss
        """
        outputs = self.forward(states, actions, return_full=True)

        # State prediction loss
        if self.probabilistic:
            # Gaussian NLL
            mean = outputs["next_state_mean"]
            log_var = outputs["next_state_log_var"]
            var = log_var.exp()

            state_loss = 0.5 * (
                log_var + (next_states - mean) ** 2 / var
            ).mean()
        else:
            state_loss = F.mse_loss(outputs["next_state_mean"], next_states)

        # Reward prediction loss
        reward_loss = F.mse_loss(outputs["reward"], rewards)

        # Done prediction loss (binary cross-entropy)
        done_loss = F.binary_cross_entropy(
            outputs["done_prob"],
            dones.float(),
        )

        # Combined loss
        total_loss = state_loss + 0.5 * reward_loss + 0.1 * done_loss

        return {
            "total_loss": total_loss,
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "done_loss": done_loss,
        }

    def imagine(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Imagine future trajectory given action sequence.
        
        Args:
            initial_state: Starting state (batch, state_dim)
            action_sequence: Sequence of actions (batch, horizon, action_dim)
            
        Returns:
            Dictionary with imagined states, rewards, dones
        """
        batch_size, horizon, _ = action_sequence.shape

        states = [initial_state]
        rewards = []
        dones = []

        current_state = initial_state

        for t in range(horizon):
            action = action_sequence[:, t]
            outputs = self.forward(current_state, action)

            # Get next state (sample if probabilistic)
            if self.probabilistic:
                mean = outputs["next_state_mean"]
                std = (outputs["next_state_log_var"] * 0.5).exp()
                next_state = mean + std * torch.randn_like(std)
            else:
                next_state = outputs["next_state_mean"]

            states.append(next_state)
            rewards.append(outputs["reward"])
            dones.append(outputs["done_prob"])

            current_state = next_state

        return {
            "states": torch.stack(states[1:], dim=1),  # (batch, horizon, state_dim)
            "rewards": torch.stack(rewards, dim=1),     # (batch, horizon)
            "dones": torch.stack(dones, dim=1),         # (batch, horizon)
        }
