"""Curiosity modules - intrinsic motivation for exploration."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CuriosityModule(nn.Module, ABC):
    """Base class for curiosity-driven exploration modules."""

    @abstractmethod
    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intrinsic reward based on curiosity."""
        pass

    @abstractmethod
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update curiosity module and return losses."""
        pass


class ICM(CuriosityModule):
    """
    Intrinsic Curiosity Module.
    
    Uses prediction error as intrinsic reward:
    - Forward model predicts next state encoding
    - Inverse model predicts action from state pair
    - Intrinsic reward = prediction error of forward model
    
    Reference: "Curiosity-driven Exploration by Self-Supervised Prediction"
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 64,
        beta: float = 0.2,  # Weight of forward vs inverse loss
        scaling: float = 0.01,  # Intrinsic reward scaling
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.beta = beta
        self.scaling = scaling

        # Feature encoder (shared)
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Forward model: f(phi(s), a) -> phi(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Inverse model: g(phi(s), phi(s')) -> a
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to feature space."""
        return self.feature_encoder(state)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass computing predictions.
        
        Returns:
            - phi_state: Encoded current state
            - phi_next_state: Encoded next state
            - phi_next_pred: Predicted next state encoding
            - action_pred: Predicted action from inverse model
        """
        # Encode states
        phi_state = self.encode_state(state)
        phi_next_state = self.encode_state(next_state)

        # Handle discrete actions
        if action.dtype in (torch.long, torch.int):
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            action_one_hot = action

        if action_one_hot.dim() == 1:
            action_one_hot = action_one_hot.unsqueeze(-1)

        # Forward model prediction
        forward_input = torch.cat([phi_state, action_one_hot], dim=-1)
        phi_next_pred = self.forward_model(forward_input)

        # Inverse model prediction
        inverse_input = torch.cat([phi_state, phi_next_state], dim=-1)
        action_pred = self.inverse_model(inverse_input)

        return {
            "phi_state": phi_state,
            "phi_next_state": phi_next_state,
            "phi_next_pred": phi_next_pred,
            "action_pred": action_pred,
        }

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward as forward model prediction error.
        """
        with torch.no_grad():
            outputs = self.forward(state, action, next_state)

            # Intrinsic reward = L2 prediction error
            pred_error = F.mse_loss(
                outputs["phi_next_pred"],
                outputs["phi_next_state"],
                reduction="none",
            ).mean(dim=-1)

            return self.scaling * pred_error.unsqueeze(-1)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update ICM and return losses."""
        outputs = self.forward(states, actions, next_states)

        # Forward model loss
        forward_loss = F.mse_loss(
            outputs["phi_next_pred"],
            outputs["phi_next_state"].detach(),
        )

        # Inverse model loss
        if actions.dtype in (torch.long, torch.int):
            inverse_loss = F.cross_entropy(outputs["action_pred"], actions)
        else:
            inverse_loss = F.mse_loss(outputs["action_pred"], actions)

        # Combined loss
        total_loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss

        return {
            "total_loss": total_loss,
            "forward_loss": forward_loss,
            "inverse_loss": inverse_loss,
        }


class RND(CuriosityModule):
    """
    Random Network Distillation.
    
    Uses fixed random target network and trains predictor to match it.
    Intrinsic reward = prediction error (high in novel states).
    
    Reference: "Exploration by Random Network Distillation"
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,  # Not used, but kept for interface consistency
        hidden_dim: int = 256,
        feature_dim: int = 64,
        scaling: float = 0.01,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.scaling = scaling

        # Random (fixed) target network
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Predictor network (trained)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # Running statistics for normalization
        self.obs_mean = nn.Parameter(torch.zeros(state_dim), requires_grad=False)
        self.obs_var = nn.Parameter(torch.ones(state_dim), requires_grad=False)
        self.reward_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.reward_var = nn.Parameter(torch.ones(1), requires_grad=False)
        self.count = nn.Parameter(torch.zeros(1), requires_grad=False)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics."""
        return (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)

    def update_obs_stats(self, obs: torch.Tensor) -> None:
        """Update running observation statistics."""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]

        total_count = self.count + batch_count

        # Welford's online algorithm
        delta = batch_mean - self.obs_mean
        self.obs_mean.data = (
            self.obs_mean + delta * batch_count / total_count
        )
        self.obs_var.data = (
            (self.obs_var * self.count + batch_var * batch_count
             + delta ** 2 * self.count * batch_count / total_count)
            / total_count
        )
        self.count.data = total_count

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            - target_features: Output of fixed random network
            - predictor_features: Output of trained predictor
        """
        # Normalize observations
        state_norm = self.normalize_obs(state)

        target_features = self.target(state_norm)
        predictor_features = self.predictor(state_norm)

        return {
            "target_features": target_features,
            "predictor_features": predictor_features,
        }

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,  # Not used
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward as prediction error on next state.
        """
        with torch.no_grad():
            outputs = self.forward(next_state)

            # Intrinsic reward = L2 prediction error
            pred_error = F.mse_loss(
                outputs["predictor_features"],
                outputs["target_features"],
                reduction="none",
            ).mean(dim=-1)

            # Normalize reward
            normalized_reward = pred_error / (self.reward_var.sqrt() + 1e-8)

            return self.scaling * normalized_reward

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,  # Not used
        next_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update RND predictor and return loss."""
        # Update observation statistics
        self.update_obs_stats(next_states)

        outputs = self.forward(next_states)

        # Predictor loss
        loss = F.mse_loss(
            outputs["predictor_features"],
            outputs["target_features"].detach(),
        )

        # Update reward statistics
        with torch.no_grad():
            pred_error = F.mse_loss(
                outputs["predictor_features"],
                outputs["target_features"],
                reduction="none",
            ).mean(dim=-1)
            self.reward_var.data = (
                0.99 * self.reward_var + 0.01 * pred_error.var()
            )

        return {
            "total_loss": loss,
            "predictor_loss": loss,
        }


class CompetenceProgress(nn.Module):
    """
    Competence-based intrinsic motivation.
    
    Tracks learning progress as intrinsic reward:
    - Reward = improvement in prediction accuracy
    - Encourages seeking experiences where learning is happening
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        window_size: int = 100,
        scaling: float = 0.01,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.scaling = scaling

        # Forward predictor
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Error history for computing progress
        self.error_history: list[torch.Tensor] = []

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state."""
        if action.dtype in (torch.long, torch.int):
            action = F.one_hot(action, num_classes=self.action_dim).float()

        combined = torch.cat([state, action], dim=-1)
        return self.predictor(combined)

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward based on learning progress."""
        with torch.no_grad():
            pred = self.forward(state, action)
            current_error = F.mse_loss(pred, next_state, reduction="none").mean(dim=-1)

            # Compute progress (decrease in error)
            if len(self.error_history) >= 2:
                old_error = torch.stack(self.error_history[-self.window_size:]).mean(dim=0)
                progress = old_error - current_error
                intrinsic = torch.clamp(progress, min=0) * self.scaling
            else:
                intrinsic = torch.zeros_like(current_error)

            return intrinsic

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update predictor and return loss."""
        pred = self.forward(states, actions)
        loss = F.mse_loss(pred, next_states)

        # Track error history
        with torch.no_grad():
            error = F.mse_loss(pred, next_states, reduction="none").mean(dim=-1)
            self.error_history.append(error)

            # Keep only recent history
            if len(self.error_history) > self.window_size * 2:
                self.error_history = self.error_history[-self.window_size:]

        return {
            "total_loss": loss,
            "predictor_loss": loss,
        }
