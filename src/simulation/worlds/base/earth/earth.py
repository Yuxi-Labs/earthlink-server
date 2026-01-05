"""Earth - the simulation's world state.

Digital twin of Earth using real geographic and weather data.
"""

from datetime import datetime, timedelta, timezone
from math import cos, pi
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.worlds.base.earth import get_geo_source
from src.data.sources.weather import get_weather_source, WeatherData
from src.data.sources.climate import get_climate_source, ClimateData
from .config import EarthConfig


class Earth:
    """
    Digital twin of Earth using real geographic data.
    
    This is the simulation's world state.
    Agents explore Earth using real coordinates and PostGIS geography.
    """

    def __init__(
        self,
        config: EarthConfig | None = None,
        db_session: Optional[AsyncSession] = None,
    ):
        self.config = config or EarthConfig()
        self.db_session = db_session
        
        # From config
        self.id = self.config.world_id
        self.name = self.config.name
        self.description = self.config.description
        self.world_type = "spatial"
        
        self.step_size_km = self.config.step_size_km
        self.perception_radius_km = self.config.perception_radius_km
        
        # State
        self._current_step = 0
        self._is_loaded = False

        # Conversion: 1 degree ≈ 111 km
        self.lat_step = self.step_size_km / 111.0
        self.lon_step = self.step_size_km / 111.0

        # Current position
        self.current_position: Tuple[float, float] = (0.0, 0.0)

    async def load(self) -> None:
        """Load world resources."""
        self._is_loaded = True

    async def unload(self) -> None:
        """Unload world resources."""
        self._is_loaded = False

    async def reset(self) -> Dict[str, Any]:
        """Reset to random spawn location."""
        spawn_locations = self.config.spawn_locations
        idx = np.random.randint(len(spawn_locations))
        self.current_position = spawn_locations[idx]
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

        direction = action.get("direction", 4)
        if isinstance(direction, torch.Tensor):
            direction = int(direction.item())

        # Move
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

        obs = await self._get_observation()
        reward = await self._compute_reward(direction)

        info = {
            "position": {"lat": lat, "lon": lon},
            "step": self._current_step,
        }

        return obs, reward, False, info

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "Box",
            "shape": (self.config.observation_dim,),
            "dtype": "float32",
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "Discrete",
            "n": self.config.action_dim,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "world_type": self.world_type,
            "observation_dim": self.config.observation_dim,
            "action_dim": self.config.action_dim,
            "is_deterministic": True,
            "is_episodic": False,
            "source_type": self.config.source_type,
        }

    async def _get_observation(self) -> Dict[str, Any]:
        """Build observation from current position."""
        lat, lon = self.current_position
        obs_vector = np.zeros(self.config.observation_dim, dtype=np.float32)

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

        # Encode features
        for i, feature in enumerate(features[:10]):
            base_idx = 2 + (i * 8)
            distance_km = feature.get("distance_meters", 0) / 1000.0
            obs_vector[base_idx] = min(1.0, distance_km / self.perception_radius_km)

        # Environment context (real weather if enabled)
        env: Dict[str, Any] = {}
        if self.config.enable_weather:
            env = await self._get_real_weather(lat, lon)
        elif self.config.enable_day_night:
            env = self._get_calculated_environment(lat, lon)
        
        if env:
            obs_vector[82] = env.get("local_hour", 12.0) / 24.0
            obs_vector[83] = 1.0 if env.get("is_day", True) else 0.0
            obs_vector[84] = np.clip((env.get("temperature_c", 20.0) + 30.0) / 70.0, 0.0, 1.0)
            obs_vector[85] = env.get("humidity_percent", 50.0) / 100.0
            obs_vector[86] = np.clip(env.get("wind_speed_kph", 10.0) / 50.0, 0.0, 1.0)
            obs_vector[87] = env.get("cloud_cover_percent", 0.0) / 100.0
            obs_vector[88] = env.get("precipitation_mm", 0.0) / 10.0  # Normalize to ~10mm
            obs_vector[89] = np.clip(env.get("pressure_hpa", 1013.25) / 1050.0, 0.9, 1.1) - 0.9

        # Get climate classification (cached, doesn't hit API every step)
        climate: Dict[str, Any] = {}
        if self.config.enable_weather:
            try:
                climate_source = get_climate_source()
                climate_data = await climate_source.get_climate(lat, lon)
                climate = climate_data.to_dict()
            except Exception:
                pass  # Climate is supplementary, don't fail on error
        
        return {
            "vector": obs_vector,
            "position": {"lat": lat, "lon": lon},
            "features": features,
            "environment": env,
            "climate": climate,
        }

    async def _compute_reward(self, action: int) -> float:
        """Compute exploration reward."""
        lat, lon = self.current_position
        reward = self.config.step_penalty

        geo_source = get_geo_source()
        features = await geo_source.get_nearby_features(
            lat=lat, lon=lon, radius_meters=1000, limit=5
        )

        for feature in features:
            metadata = feature.get("metadata", {})
            pop = metadata.get("pop_est", 0) or 0
            if pop > self.config.city_population_threshold:
                reward += self.config.city_reward

        if action == 4:
            reward += self.config.stay_penalty

        return float(reward)

    async def _get_real_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get real weather data from Open-Meteo API."""
        try:
            weather_source = get_weather_source()
            weather = await weather_source.get_current(lat, lon)
            
            # Parse local hour from local time
            local_hour = weather.time_local.hour + weather.time_local.minute / 60.0
            
            # Determine season from month
            month = weather.time_local.month
            if lat < 0:  # Southern hemisphere
                if month in (12, 1, 2):
                    season = "summer"
                elif month in (3, 4, 5):
                    season = "autumn"
                elif month in (6, 7, 8):
                    season = "winter"
                else:
                    season = "spring"
            else:  # Northern hemisphere
                if month in (12, 1, 2):
                    season = "winter"
                elif month in (3, 4, 5):
                    season = "spring"
                elif month in (6, 7, 8):
                    season = "summer"
                else:
                    season = "autumn"
            
            return {
                # Time
                "time_utc": weather.time_utc.isoformat(),
                "time_local": weather.time_local.isoformat(),
                "local_hour": local_hour,
                "timezone": weather.timezone,
                "is_day": weather.is_day,
                "season": season,
                
                # Temperature
                "temperature_c": weather.temperature_c,
                "feels_like_c": weather.feels_like_c,
                
                # Atmosphere
                "humidity_percent": weather.humidity_percent,
                "pressure_hpa": weather.pressure_hpa,
                "cloud_cover_percent": weather.cloud_cover_percent,
                
                # Wind
                "wind_speed_kph": weather.wind_speed_kph,
                "wind_direction_deg": weather.wind_direction_deg,
                "wind_gusts_kph": weather.wind_gusts_kph,
                
                # Precipitation
                "precipitation_mm": weather.precipitation_mm,
                "rain_mm": weather.rain_mm,
                "snowfall_cm": weather.snowfall_cm,
                
                # Conditions
                "weather_code": weather.weather_code,
                "weather_description": weather.weather_description,
                
                # Location
                "elevation_m": weather.elevation_m,
                
                # Source
                "source": "open-meteo",
            }
        except Exception as e:
            # Fallback to calculated if API fails
            return self._get_calculated_environment(lat, lon)
    
    def _get_calculated_environment(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fallback: derive environment context from position (no real data)."""
        utc_now = datetime.now(timezone.utc)
        local_offset_hours = lon / 15.0
        local_time = utc_now + timedelta(hours=local_offset_hours)
        local_hour = local_time.hour + local_time.minute / 60.0

        daylight_fraction = max(0.0, 0.5 + 0.5 * cos(2 * pi * (local_hour - 12) / 24))
        is_day = 6 <= local_hour <= 18

        month = local_time.month
        if lat < 0:  # Southern hemisphere
            if month in (12, 1, 2):
                season = "summer"
                seasonal_shift = 5.0
            elif month in (3, 4, 5):
                season = "autumn"
                seasonal_shift = 0.0
            elif month in (6, 7, 8):
                season = "winter"
                seasonal_shift = -5.0
            else:
                season = "spring"
                seasonal_shift = 0.0
        else:
            if month in (12, 1, 2):
                season = "winter"
                seasonal_shift = -5.0
            elif month in (3, 4, 5):
                season = "spring"
                seasonal_shift = 0.0
            elif month in (6, 7, 8):
                season = "summer"
                seasonal_shift = 5.0
            else:
                season = "autumn"
                seasonal_shift = 0.0

        base_temp = 30.0 - (abs(lat) * 0.2)
        diurnal = (daylight_fraction - 0.5) * 10.0
        temperature_c = base_temp + seasonal_shift + diurnal

        humidity_percent = float(np.clip(40.0 + 40.0 * daylight_fraction, 20.0, 90.0))
        wind_speed_kph = float(np.clip(5.0 + 20.0 * (1 - daylight_fraction), 0.0, 40.0))

        return {
            "time_local": local_time.isoformat(),
            "local_hour": local_hour,
            "is_day": is_day,
            "season": season,
            "temperature_c": float(temperature_c),
            "feels_like_c": float(temperature_c),
            "humidity_percent": humidity_percent,
            "pressure_hpa": 1013.25,
            "cloud_cover_percent": 30.0,
            "wind_speed_kph": wind_speed_kph,
            "wind_direction_deg": 0.0,
            "wind_gusts_kph": wind_speed_kph * 1.5,
            "precipitation_mm": 0.0,
            "rain_mm": 0.0,
            "snowfall_cm": 0.0,
            "weather_code": 0,
            "weather_description": "Clear sky",
            "source": "calculated",
        }
