"""Geospatial knowledge source - query Earth data via PostGIS."""

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db


class GeoSource:
    """
    Interface to geospatial data via PostGIS.
    
    Agents can query:
    - Points of interest near locations
    - Geographic regions and boundaries
    - Spatial relationships
    - Environmental data layers
    """

    async def get_nearby_features(
        self,
        lat: float,
        lon: float,
        radius_meters: float = 1000,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get geographic features near a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_meters: Search radius in meters
            limit: Maximum results
            
        Returns:
            List of features with distance, name, type
        """
        # Create point geometry
        point_wkt = f"POINT({lon} {lat})"
        
        async with get_db() as db:
            # Query spatial data (assumes a features table exists)
            query = text("""
                SELECT 
                    id,
                    name,
                    feature_type,
                    ST_AsGeoJSON(geom) as geometry,
                    ST_Distance(
                        geom::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography
                    ) as distance_meters
                FROM geo_features
                WHERE ST_DWithin(
                    geom::geography,
                    ST_GeomFromText(:point_wkt, 4326)::geography,
                    :radius
                )
                ORDER BY distance_meters
                LIMIT :limit
            """)
            
            result = await db.execute(
                query,
                {
                    "point_wkt": point_wkt,
                    "radius": radius_meters,
                    "limit": limit,
                }
            )
            
            rows = result.fetchall()
            
            features = []
            for row in rows:
                features.append({
                    "id": str(row.id),
                    "name": row.name,
                    "type": row.feature_type,
                    "geometry": row.geometry,
                    "distance_meters": float(row.distance_meters),
                })
            
            return features

    async def query_region(
        self,
        name: str,
    ) -> dict[str, Any] | None:
        """
        Query information about a geographic region.
        
        Args:
            name: Region name (city, country, etc.)
            
        Returns:
            Region data including boundaries, area, centroid
        """
        async with get_db() as db:
            query = text("""
                SELECT 
                    id,
                    name,
                    region_type,
                    ST_AsGeoJSON(geom) as geometry,
                    ST_Area(geom::geography) / 1000000 as area_km2,
                    ST_AsText(ST_Centroid(geom)) as centroid
                FROM geo_regions
                WHERE LOWER(name) = LOWER(:name)
                LIMIT 1
            """)
            
            result = await db.execute(query, {"name": name})
            row = result.fetchone()
            
            if not row:
                return None
            
            return {
                "id": str(row.id),
                "name": row.name,
                "type": row.region_type,
                "geometry": row.geometry,
                "area_km2": float(row.area_km2),
                "centroid": row.centroid,
            }

    async def spatial_query(
        self,
        query_type: str,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute custom spatial query.
        
        Supported query types:
        - "within": Find features within polygon
        - "intersects": Find features intersecting geometry
        - "contains": Find regions containing point
        - "nearest": Find N nearest features to point
        
        Args:
            query_type: Type of spatial query
            params: Query parameters
            
        Returns:
            List of results
        """
        if query_type == "within":
            return await self._query_within(params)
        elif query_type == "intersects":
            return await self._query_intersects(params)
        elif query_type == "contains":
            return await self._query_contains(params)
        elif query_type == "nearest":
            return await self._query_nearest(params)
        else:
            return []

    async def _query_within(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Find features within a polygon."""
        polygon_wkt = params.get("polygon_wkt")
        limit = params.get("limit", 100)
        
        async with get_db() as db:
            query = text("""
                SELECT 
                    id,
                    name,
                    feature_type,
                    ST_AsGeoJSON(geom) as geometry
                FROM geo_features
                WHERE ST_Within(
                    geom,
                    ST_GeomFromText(:polygon_wkt, 4326)
                )
                LIMIT :limit
            """)
            
            result = await db.execute(
                query,
                {"polygon_wkt": polygon_wkt, "limit": limit}
            )
            
            rows = result.fetchall()
            
            return [
                {
                    "id": str(row.id),
                    "name": row.name,
                    "type": row.feature_type,
                    "geometry": row.geometry,
                }
                for row in rows
            ]

    async def _query_intersects(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Find features intersecting a geometry."""
        geometry_wkt = params.get("geometry_wkt")
        limit = params.get("limit", 100)
        
        async with get_db() as db:
            query = text("""
                SELECT 
                    id,
                    name,
                    feature_type,
                    ST_AsGeoJSON(geom) as geometry
                FROM geo_features
                WHERE ST_Intersects(
                    geom,
                    ST_GeomFromText(:geometry_wkt, 4326)
                )
                LIMIT :limit
            """)
            
            result = await db.execute(
                query,
                {"geometry_wkt": geometry_wkt, "limit": limit}
            )
            
            rows = result.fetchall()
            
            return [
                {
                    "id": str(row.id),
                    "name": row.name,
                    "type": row.feature_type,
                    "geometry": row.geometry,
                }
                for row in rows
            ]

    async def _query_contains(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Find regions containing a point."""
        lat = params.get("lat")
        lon = params.get("lon")
        
        point_wkt = f"POINT({lon} {lat})"
        
        async with get_db() as db:
            query = text("""
                SELECT 
                    id,
                    name,
                    region_type,
                    ST_AsGeoJSON(geom) as geometry
                FROM geo_regions
                WHERE ST_Contains(
                    geom,
                    ST_GeomFromText(:point_wkt, 4326)
                )
                ORDER BY ST_Area(geom)
            """)
            
            result = await db.execute(query, {"point_wkt": point_wkt})
            
            rows = result.fetchall()
            
            return [
                {
                    "id": str(row.id),
                    "name": row.name,
                    "type": row.region_type,
                    "geometry": row.geometry,
                }
                for row in rows
            ]

    async def _query_nearest(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Find N nearest features to a point."""
        lat = params.get("lat")
        lon = params.get("lon")
        n = params.get("n", 5)
        
        point_wkt = f"POINT({lon} {lat})"
        
        async with get_db() as db:
            query = text("""
                SELECT 
                    id,
                    name,
                    feature_type,
                    ST_AsGeoJSON(geom) as geometry,
                    ST_Distance(
                        geom::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography
                    ) as distance_meters
                FROM geo_features
                ORDER BY distance_meters
                LIMIT :n
            """)
            
            result = await db.execute(
                query,
                {"point_wkt": point_wkt, "n": n}
            )
            
            rows = result.fetchall()
            
            return [
                {
                    "id": str(row.id),
                    "name": row.name,
                    "type": row.feature_type,
                    "geometry": row.geometry,
                    "distance_meters": float(row.distance_meters),
                }
                for row in rows
            ]

    async def get_location_context(
        self,
        lat: float,
        lon: float,
    ) -> dict[str, Any]:
        """
        Get comprehensive context about a location.
        
        Returns nearby features, containing regions, etc.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Complete location context
        """
        # Parallel queries
        import asyncio
        
        nearby_features_task = self.get_nearby_features(lat, lon, radius_meters=5000, limit=20)
        containing_regions_task = self.spatial_query(
            "contains",
            {"lat": lat, "lon": lon}
        )
        
        nearby_features, containing_regions = await asyncio.gather(
            nearby_features_task,
            containing_regions_task,
        )
        
        return {
            "lat": lat,
            "lon": lon,
            "nearby_features": nearby_features,
            "containing_regions": containing_regions,
        }


# Singleton instance
_geo_source = None


def get_geo_source() -> GeoSource:
    """Get or create GeoSource singleton."""
    global _geo_source
    if _geo_source is None:
        _geo_source = GeoSource()
    return _geo_source
