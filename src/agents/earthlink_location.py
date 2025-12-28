"""Earthlink Location Awareness - Agents understanding where they are in the VW."""

from typing import Any
from uuid import UUID

import httpx


class EarthlinkLocationMixin:
    """
    Mixin for agent location awareness in Earthlink (VW).
    
    Provides methods for agents to:
    - Know their lat/lon position
    - Query surroundings via Earthlink API
    - Understand geographic context (city, neighborhood, buildings, roads)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._earthlink_api_base = "http://localhost:8000/api/v1/earthlink"
        self._http_client = None
    
    @property
    def latitude(self) -> float:
        """Current latitude in Earthlink (stored in location.x)."""
        return self.state.location.x
    
    @property
    def longitude(self) -> float:
        """Current longitude in Earthlink (stored in location.y)."""
        return self.state.location.y
    
    @property
    def altitude(self) -> float:
        """Current altitude in meters (stored in location.z)."""
        return self.state.location.z
    
    def set_earthlink_position(self, lat: float, lon: float, altitude: float = 0.0) -> None:
        """
        Set agent's position in Earthlink.
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            altitude: Altitude in meters (default 0)
        """
        self.state.location.x = lat
        self.state.location.y = lon
        self.state.location.z = altitude
    
    def move_earthlink_position(self, lat_delta: float, lon_delta: float, alt_delta: float = 0.0) -> None:
        """
        Move agent by delta amounts.
        
        Args:
            lat_delta: Latitude change (degrees)
            lon_delta: Longitude change (degrees)
            alt_delta: Altitude change (meters)
        """
        self.state.location.x += lat_delta
        self.state.location.y += lon_delta
        self.state.location.z += alt_delta
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for Earthlink API."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def earthlink_query_nearby(
        self,
        radius_meters: float = 500,
        limit: int = 10,
        feature_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query nearby features from Earthlink at current position.
        
        Args:
            radius_meters: Search radius (10-50000)
            limit: Max results per feature type (1-100)
            feature_types: Filter types (buildings, roads, places, pois, water, landcover)
        
        Returns:
            Dict with nearby buildings, roads, POIs, etc.
        """
        client = await self._get_http_client()
        
        params = {
            "lat": self.latitude,
            "lon": self.longitude,
            "radius_meters": radius_meters,
            "limit": limit,
        }
        
        if feature_types:
            params["types"] = ",".join(feature_types)
        
        response = await client.get(f"{self._earthlink_api_base}/nearby", params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def earthlink_get_context(
        self,
        radius_meters: float = 5000,
    ) -> dict[str, Any]:
        """
        Get comprehensive context about current location.
        
        Returns nearby features, containing regions (boundaries), and summary stats.
        
        Args:
            radius_meters: Search radius (100-50000)
        
        Returns:
            Dict with comprehensive location context
        """
        client = await self._get_http_client()
        
        params = {
            "lat": self.latitude,
            "lon": self.longitude,
            "radius_meters": radius_meters,
        }
        
        response = await client.get(f"{self._earthlink_api_base}/context", params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def earthlink_get_regions(self) -> dict[str, Any]:
        """
        Get administrative boundaries containing current position.
        
        Returns nested regions (country → state → city → neighborhood).
        
        Returns:
            Dict with containing regions
        """
        client = await self._get_http_client()
        
        params = {
            "lat": self.latitude,
            "lon": self.longitude,
        }
        
        response = await client.get(f"{self._earthlink_api_base}/regions", params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def earthlink_where_am_i(self) -> dict[str, Any]:
        """
        Answer the question: "Where am I?"
        
        Returns a human-readable description of current location including:
        - Coordinates
        - Administrative regions (city, state, country)
        - Nearby landmarks/buildings
        - Local roads
        
        Returns:
            Dict with comprehensive "where am I" information
        """
        # Get both regions and nearby features
        regions_data = await self.earthlink_get_regions()
        nearby_data = await self.earthlink_query_nearby(radius_meters=200, limit=5)
        
        # Extract key information
        regions = regions_data.get("regions", [])
        
        # Parse administrative hierarchy
        admin_context = {
            "neighborhood": None,
            "city": None,
            "state": None,
            "country": None,
        }
        
        for region in regions:
            name = region.get("name", "")
            region_type = region.get("type", "")
            
            # Simple heuristics for region classification
            if "neighborhood" in region_type.lower() or region.get("admin_level") == "10":
                admin_context["neighborhood"] = name
            elif "city" in region_type.lower() or region.get("admin_level") in ["8", "6"]:
                admin_context["city"] = name
            elif "state" in region_type.lower() or region.get("admin_level") == "4":
                admin_context["state"] = name
            elif "country" in region_type.lower() or region.get("admin_level") == "2":
                admin_context["country"] = name
        
        # Get notable nearby features
        nearby_buildings = nearby_data.get("by_category", {}).get("building", [])[:3]
        nearby_roads = nearby_data.get("by_category", {}).get("road", [])[:3]
        
        # Build human-readable description
        location_parts = []
        if admin_context["neighborhood"]:
            location_parts.append(admin_context["neighborhood"])
        if admin_context["city"]:
            location_parts.append(admin_context["city"])
        if admin_context["state"]:
            location_parts.append(admin_context["state"])
        if admin_context["country"]:
            location_parts.append(admin_context["country"])
        
        description = ", ".join(location_parts) if location_parts else "Unknown location"
        
        return {
            "position": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
            },
            "description": description,
            "administrative": admin_context,
            "nearby_buildings": [
                {
                    "name": b.get("name", "Unnamed"),
                    "type": b.get("type", "unknown"),
                    "distance_meters": b.get("distance_meters", 0),
                }
                for b in nearby_buildings
            ],
            "nearby_roads": [
                {
                    "name": r.get("name", "Unnamed"),
                    "type": r.get("type", "unknown"),
                    "distance_meters": r.get("distance_meters", 0),
                }
                for r in nearby_roads
            ],
        }
    
    async def earthlink_spawn_random_city(self, region: str = "south_america") -> dict[str, Any]:
        """
        Spawn agent at a random major city.
        
        Args:
            region: Geographic region (south_america, europe, north_america)
        
        Returns:
            Dict with spawn location and context
        """
        import random
        
        # Major cities by region (with Earthlink coverage)
        cities = {
            "south_america": [
                {"name": "São Paulo", "lat": -23.55, "lon": -46.63},
                {"name": "Rio de Janeiro", "lat": -22.91, "lon": -43.17},
                {"name": "Buenos Aires", "lat": -34.60, "lon": -58.38},
                {"name": "Lima", "lat": -12.05, "lon": -77.03},
                {"name": "Bogotá", "lat": 4.71, "lon": -74.07},
                {"name": "Santiago", "lat": -33.45, "lon": -70.67},
            ],
            "europe": [
                {"name": "London", "lat": 51.51, "lon": -0.13},
                {"name": "Paris", "lat": 48.86, "lon": 2.35},
                {"name": "Berlin", "lat": 52.52, "lon": 13.40},
                {"name": "Madrid", "lat": 40.42, "lon": -3.70},
                {"name": "Rome", "lat": 41.90, "lon": 12.50},
            ],
            "north_america": [
                {"name": "New York", "lat": 40.71, "lon": -74.01},
                {"name": "Los Angeles", "lat": 34.05, "lon": -118.24},
                {"name": "Mexico City", "lat": 19.43, "lon": -99.13},
                {"name": "Chicago", "lat": 41.88, "lon": -87.63},
                {"name": "Toronto", "lat": 43.65, "lon": -79.38},
            ],
        }
        
        city_list = cities.get(region.lower(), cities["south_america"])
        selected_city = random.choice(city_list)
        
        # Set position
        self.set_earthlink_position(selected_city["lat"], selected_city["lon"])
        
        # Get context
        context = await self.earthlink_where_am_i()
        
        return {
            "spawned_at": selected_city["name"],
            "region": region,
            "context": context,
        }
