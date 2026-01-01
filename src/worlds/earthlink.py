"""
Earthlink virtual world built from real geography.

The VW mirrors Earth:
- Real lat/lon coordinates (EPSG:4326)
- Geography from PostGIS (Natural Earth data)
- Agent positions are actual geographic locations
- Observations include nearby features from real world
- Environment context (time/weather) derived from position
"""

from datetime import datetime, timedelta, timezone
from math import cos, pi
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from src.simulation.world import World, WorldMetadata
from src.knowledge.geo import get_geo_source


class EarthlinkWorld(World):
    """
    Digital twin using real geographic data.
    
    Extends simulation.World to work with the runner.
    Agents explore Earth using real coordinates and PostGIS geography.
    """

    def __init__(
        self,
        world_id: str = "earthlink",
        name: str = "Earthlink",
        step_size_km: float = 1.0,
        perception_radius_km: float = 5.0,
        db_session: Optional[AsyncSession] = None,
    ):
        metadata = WorldMetadata(
            id=world_id,
            name=name,
            description="Virtual world with real geography from Natural Earth/OSM data",
            world_type="spatial",
            observation_dim=92,
            action_dim=5,
            is_deterministic=True,
            is_episodic=False,
            max_steps=None,
            source_type="postgis",
            source_config={
                "step_size_km": step_size_km,
                "perception_radius_km": perception_radius_km,
            },
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

        # Spawn locations aligned to current ingested data (Australia)
        self.spawn_locations = [
            (-33.8688, 151.2093),  # Sydney
            (-37.8136, 144.9631),  # Melbourne
            (-27.4698, 153.0251),  # Brisbane
            (-31.9523, 115.8613),  # Perth
            (-34.9285, 138.6007),  # Adelaide
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
        action: Dict[str, Any],
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
            "step": self._current_step,
        }

        return obs, reward, False, info

    def get_observation_space(self) -> Dict[str, Any]:
        """Return observation space spec."""
        return {
            "type": "Box",
            "shape": (92,),
            "dtype": "float32",
        }

    def get_action_space(self) -> Dict[str, Any]:
        """Return action space spec."""
        return {
            "type": "Discrete",
            "n": 5,
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
            limit=10,
        )

        # Encode features (simplified)
        for i, feature in enumerate(features[:10]):
            base_idx = 2 + (i * 8)
            distance_km = feature.get("distance_meters", 0) / 1000.0
            obs_vector[base_idx] = min(1.0, distance_km / self.perception_radius_km)

        # Environment context (time/weather) mapped into unused tail of vector
        env = self._get_environment_context(lat, lon)
        obs_vector[82] = env["local_hour"] / 24.0
        obs_vector[83] = 1.0 if env["is_daytime"] else 0.0
        obs_vector[84] = np.clip((env["temperature_c"] + 30.0) / 70.0, 0.0, 1.0)
        obs_vector[85] = env["humidity"]
        obs_vector[86] = np.clip(env["wind_kph"] / 50.0, 0.0, 1.0)
        obs_vector[87] = env["daylight_fraction"]

        return {
            "vector": obs_vector,
            "position": {"lat": lat, "lon": lon},
            "features": features,
            "environment": env,
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

    def _get_environment_context(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Derive simple environment context from position (time-of-day, weather).
        
        Uses deterministic heuristics (no external API) so behavior matches location
        and can be surfaced later in the client.
        """
        utc_now = datetime.now(timezone.utc)
        local_offset_hours = lon / 15.0  # 360 deg / 24 h
        local_time = utc_now + timedelta(hours=local_offset_hours)
        local_hour = local_time.hour + local_time.minute / 60.0

        # Day/night and daylight fraction (cosine proxy for sun altitude)
        daylight_fraction = max(0.0, 0.5 + 0.5 * cos(2 * pi * (local_hour - 12) / 24))
        is_daytime = 6 <= local_hour <= 18

        # Southern-hemisphere aware season
        month = local_time.month
        if month in (12, 1, 2):
            season = "summer"
            seasonal_shift = 5.0 if lat < 0 else -5.0
        elif month in (3, 4, 5):
            season = "autumn"
            seasonal_shift = 0.0
        elif month in (6, 7, 8):
            season = "winter"
            seasonal_shift = -5.0 if lat < 0 else 5.0
        else:
            season = "spring"
            seasonal_shift = 0.0

        base_temp = 30.0 - (abs(lat) * 0.2)
        diurnal = (daylight_fraction - 0.5) * 10.0
        temperature_c = base_temp + seasonal_shift + diurnal

        # Humidity/wind simple scalars
        humidity = float(np.clip(0.4 + 0.4 * daylight_fraction, 0.0, 1.0))
        wind_kph = float(np.clip(5.0 + 20.0 * (1 - daylight_fraction), 0.0, 40.0))

        return {
            "local_time_iso": local_time.isoformat(),
            "local_hour": local_hour,
            "is_daytime": is_daytime,
            "season": season,
            "temperature_c": float(temperature_c),
            "humidity": humidity,
            "wind_kph": wind_kph,
            "daylight_fraction": float(daylight_fraction),
        }
