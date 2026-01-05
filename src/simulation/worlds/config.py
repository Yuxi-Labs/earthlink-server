"""Shared configuration for worlds."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldConfig:
    """Base configuration shared across worlds."""
    
    # Timing
    step_delay_ms: float = 100.0
    
    # Observation
    observation_dim: int = 256
    action_dim: int = 64
    
    # Features
    max_features_per_observation: int = 10
    
    # Spawn
    max_spawn_attempts: int = 10


@dataclass 
class SpatialConfig:
    """Configuration for spatial/geographic worlds."""
    
    step_size_km: float = 1.0
    perception_radius_km: float = 5.0
    
    # Coordinate bounds
    min_lat: float = -90.0
    max_lat: float = 90.0
    min_lon: float = -180.0
    max_lon: float = 180.0

