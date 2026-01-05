"""Earth-specific configuration."""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class EarthConfig:
    """Configuration for Earth world."""
    
    # Identity
    world_id: str = "earth"
    name: str = "Earth"
    description: str = "Digital twin of Earth with real geography"
    
    # Movement
    step_size_km: float = 1.0
    perception_radius_km: float = 5.0
    
    # Observation dimensions
    observation_dim: int = 92
    action_dim: int = 5
    
    # Data source
    source_type: str = "postgis"
    
    # Spawn locations (lat, lon) - currently Australia
    spawn_locations: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-33.8688, 151.2093),  # Sydney
        (-37.8136, 144.9631),  # Melbourne
        (-27.4698, 153.0251),  # Brisbane
        (-31.9523, 115.8613),  # Perth
        (-34.9285, 138.6007),  # Adelaide
    ])
    
    # Geographic bounds for spawning (when not using specific locations)
    spawn_bounds: dict = field(default_factory=lambda: {
        "min_lat": -44.0,
        "max_lat": -10.0,
        "min_lon": 113.0,
        "max_lon": 154.0,
    })
    
    # Environment simulation
    enable_weather: bool = True           # Use real weather from Open-Meteo API
    enable_day_night: bool = True         # Track day/night cycles
    weather_cache_seconds: int = 300      # Cache weather for 5 minutes
    
    # Rewards
    step_penalty: float = -0.01
    stay_penalty: float = -0.05
    city_reward: float = 0.1
    city_population_threshold: int = 1_000_000

