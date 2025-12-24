"""Encoder networks - observation and state encoding."""

from typing import Any

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Base encoder network.
    
    Converts raw observations into latent representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            ])
            in_dim = out_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space."""
        return self.network(x)


class ObservationEncoder(nn.Module):
    """
    Multi-modal observation encoder.
    
    Handles different observation modalities and fuses them.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.latent_dim = latent_dim

        # Separate encoder for each modality
        self.encoders = nn.ModuleDict()
        for name, dim in modality_dims.items():
            self.encoders[name] = Encoder(
                input_dim=dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
            )

        # Fusion layer
        total_latent = latent_dim * len(modality_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_latent, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multi-modal observations."""
        encoded = []

        for name, encoder in self.encoders.items():
            if name in observations:
                encoded.append(encoder(observations[name]))
            else:
                # Zero padding for missing modalities
                batch_size = next(iter(observations.values())).shape[0]
                encoded.append(torch.zeros(batch_size, self.latent_dim))

        fused = torch.cat(encoded, dim=-1)
        return self.fusion(fused)


class StateEncoder(nn.Module):
    """
    State encoder with optional goal conditioning.
    
    Encodes state and goal into a unified representation.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int | None = None,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.goal_conditioned = goal_dim is not None

        # State encoder
        self.state_encoder = Encoder(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        # Goal encoder (if goal-conditioned)
        if self.goal_conditioned:
            self.goal_encoder = Encoder(
                input_dim=goal_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
            )

            # State-goal fusion
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode state with optional goal conditioning."""
        state_latent = self.state_encoder(state)

        if self.goal_conditioned and goal is not None:
            goal_latent = self.goal_encoder(goal)
            combined = torch.cat([state_latent, goal_latent], dim=-1)
            return self.fusion(combined)

        return state_latent


class SequenceEncoder(nn.Module):
    """
    Sequence encoder for temporal observations.
    
    Uses transformer or LSTM to encode sequence of observations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        sequence_length: int = 10,
        use_transformer: bool = True,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_transformer:
            # Positional encoding
            self.pos_encoding = nn.Parameter(
                torch.randn(1, sequence_length, hidden_dim) * 0.02
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        else:
            # LSTM encoder
            self.sequence_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )

        self.use_transformer = use_transformer

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence of observations.
        
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Latent representation (batch, latent_dim)
        """
        # Project input
        x = self.input_proj(x)

        if self.use_transformer:
            # Add positional encoding
            x = x + self.pos_encoding[:, : x.shape[1], :]
            # Transformer encoding
            x = self.sequence_encoder(x)
            # Use CLS token or mean pooling
            x = x.mean(dim=1)
        else:
            # LSTM encoding
            _, (h_n, _) = self.sequence_encoder(x)
            x = h_n[-1]

        return self.output_proj(x)
