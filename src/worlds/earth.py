"""
Earth digital twin world - agents live on real planet with actual geography.

The VW IS Earth:
- Real lat/lon coordinates (EPSG:4326)
- Geography from PostGIS (Natural Earth data)
- Agent positions are actual geographic locations
- Observations include nearby features from real world
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from src.simulation.world import World, WorldMetadata
from src.knowledge.geo import get_geo_source


class EarthWorld(World):
    """
    Digital twin of Earth using real geographic data.
    
    Extends simulation.World to work with the runner.
    Agents explore Earth using real coordinates and PostGIS geography.
    """
    
    def __init__(
        self,
        world_id: str = "earth",
        name: str = "Earth",
        step_size_km: float = 1.0,
        perception_radius_km: float = 5.0,
        db_session: Optional[AsyncSession] = None
    ):
        metadata = WorldMetadata(
            id=world_id,
            name=name,
            description="Digital twin of Earth with real geography from Natural Earth dataset",
            world_type="spatial",
            observation_dim=92,
            action_dim=5,
            is_deterministic=True,
            is_episodic=False,
            max_steps=None,
            source_type="postgis",
            source_config={
                "step_size_km": step_size_km,
                "perception_radius_km": perception_radius_km
            }
        )
        super().__init__(metadata)
        
        self.step_size_km = step_size_km
        self.perception_radius_km = perception_radius_km
        self.db_session = db_session
        self._is_loaded = False  # Track if world resources are loaded
        
        # Conversion: 1 degree ≈ 111 km
        self.lat_step = step_size_km / 111.0
        self.lon_step = step_size_km / 111.0
        
        # Current agent position
        self.current_position: Tuple[float, float] = (0.0, 0.0)
        
        # Spawn locations (major cities)
        self.spawn_locations = [
            (40.7128, -74.0060),   # New York
            (51.5074, -0.1278),    # London
            (35.6762, 139.6503),   # Tokyo
            (-33.8688, 151.2093),  # Sydney
            (48.8566, 2.3522),     # Paris
        ]
    
    async def load(self) -> None:
        """Load world resources."""
        self._is_loaded = True
    
    async def unload(self) -> None:
        """Unload world resources."""
        self._is_loaded = False
    
    async def reset(self) -> Dict[str, Any]:
        """Reset to random spawn location."""
        idx = np.random.randint(len(self.spawn_locations))
        self.current_position = self.spawn_locations[idx]
        self._current_step = 0
        return await self._get_observation()
    
    async def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute world step.
        
        Action: {"direction": 0-4} where 0=N, 1=S, 2=E, 3=W, 4=Stay
        """
        lat, lon = self.current_position
        
        # Parse action
        direction = action.get("direction", 4)
        if isinstance(direction, torch.Tensor):
            direction = int(direction.item())
        
        # Move agent
        if direction == 0:  # North
            lat = min(90.0, lat + self.lat_step)
        elif direction == 1:  # South
            lat = max(-90.0, lat - self.lat_step)
        elif direction == 2:  # East
            lon = (lon + self.lon_step + 180) % 360 - 180
        elif direction == 3:  # West
            lon = (lon - self.lon_step + 180) % 360 - 180
        
        self.current_position = (lat, lon)
        self._current_step += 1
        
        # Get new observation
        obs = await self._get_observation()
        
        # Compute reward
        reward = await self._compute_reward(direction)
        
        info = {
            "position": {"lat": lat, "lon": lon},
            "step": self._current_step
        }
        
        return obs, reward, False, info
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Return observation space spec."""
        return {
            "type": "Box",
            "shape": (92,),
            "dtype": "float32"
        }
    
    def get_action_space(self) -> Dict[str, Any]:
        """Return action space spec."""
        return {
            "type": "Discrete",
            "n": 5
        }
    
    async def _get_observation(self) -> Dict[str, Any]:
        """Build observation from current position."""
        lat, lon = self.current_position
        obs_vector = np.zeros(92, dtype=np.float32)
        
        # Position (normalized)
        obs_vector[0] = lat / 90.0
        obs_vector[1] = lon / 180.0
        
        # Get nearby features from PostGIS
        geo_source = get_geo_source()
        features = await geo_source.get_nearby_features(
            lat=lat,
            lon=lon,
            radius_meters=self.perception_radius_km * 1000,
            limit=10
        )
        
        # Encode features (simplified)
        for i, feature in enumerate(features[:10]):
            base_idx = 2 + (i * 8)
            distance_km = feature.get("distance_meters", 0) / 1000.0
            obs_vector[base_idx] = min(1.0, distance_km / self.perception_radius_km)
        
        return {
            "vector": obs_vector,
            "position": {"lat": lat, "lon": lon},
            "features": features
        }
    
    async def _compute_reward(self, action: int) -> float:
        """Compute exploration reward."""
        lat, lon = self.current_position
        reward = -0.01  # Small step penalty
        
        # Reward for nearby interesting features
        geo_source = get_geo_source()
        features = await geo_source.get_nearby_features(
            lat=lat, lon=lon, radius_meters=1000, limit=5
        )
        
        for feature in features:
            metadata = feature.get("metadata", {})
            pop = metadata.get("pop_est", 0) or 0
            if pop > 1000000:
                reward += 0.1
        
        # Penalty for staying still
        if action == 4:
            reward -= 0.05
        
        return float(reward)
