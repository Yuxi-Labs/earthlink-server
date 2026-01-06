"""Earth-specific configuration."""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class EarthConfig:
    """Configuration for Earth world."""
    
    # Identity
    world_id: str = "earth"
    name: str = "Earth"
    description: str = "Digital twin of Great Britain with real geography"
    
    # Movement
    step_size_km: float = 1.0
    perception_radius_km: float = 5.0
    
    # Observation dimensions
    observation_dim: int = 92
    action_dim: int = 5
    
    # Data source
    source_type: str = "postgis"
    
    # Spawn locations (lat, lon) - Great Britain cities
    spawn_locations: List[Tuple[float, float]] = field(default_factory=lambda: [
        (51.5074456, -0.1277653),   # London
        (52.4796992, -1.9026911),   # Birmingham
        (53.4794892, -2.2451148),   # Manchester
        (55.9533456, -3.1883749),   # Edinburgh
        (51.4816546, -3.1791934),   # Cardiff
        (51.4538022, -2.5972985),   # Bristol
        (53.7974185, -1.5437941),   # Leeds
        (53.4071991, -2.99168),     # Liverpool
        (55.861155, -4.2501687),    # Glasgow
    ])
    
    # Geographic bounds for spawning (when not using specific locations)
    # Great Britain bounding box: 49.9°N to 58.7°N, -8.2°W to 1.8°E
    spawn_bounds: dict = field(default_factory=lambda: {
        "min_lat": 49.9,
        "max_lat": 58.7,
        "min_lon": -8.2,
        "max_lon": 1.8,
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

