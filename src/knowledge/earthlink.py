"""
Earthlink Geospatial Source - Query the VW's Earth data.

Queries Earthlink (the VW) data tables:
- buildings, roads, places, pois, water_features, land_cover, boundaries
"""

from typing import Any

from sqlalchemy import text

from ..db.database import async_session_maker


class EarthlinkGeoSource:
    """
    Interface to Earthlink (VW) geospatial data.
    
    Agents can query 43+ million features:
    - Buildings (22M) - footprints, heights, types
    - Roads (14M) - network with lanes, speed limits
    - Places (500K) - cities, towns, villages
    - POIs (1.3M) - amenities, shops, tourism
    - Water (2.8M) - rivers, lakes, coastlines
    - Land Cover (2.9M) - forests, grassland, urban
    - Boundaries (71K) - administrative regions
    """

    async def get_nearby_features(
        self,
        lat: float,
        lon: float,
        radius_meters: float = 1000,
        limit: int = 10,
        feature_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get geographic features near a point from Earth.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_meters: Search radius in meters
            limit: Maximum results per feature type
            feature_types: Filter by types: ['buildings', 'roads', 'places', 'pois', 'water', 'landcover']
            
        Returns:
            List of features with distance, name, type, properties
        """
        point_wkt = f"POINT({lon} {lat})"
        
        # Default to all types if not specified
        if not feature_types:
            feature_types = ['buildings', 'roads', 'places', 'pois', 'water']
        
        all_features = []
        
        async with async_session_maker() as db:
            # Buildings
            if 'buildings' in feature_types:
                query = text("""
                    SELECT 
                        'building' as feature_type,
                        id,
                        name,
                        building_type,
                        height,
                        levels,
                        ST_Distance(
                            footprint::geography,
                            ST_GeomFromText(:point_wkt, 4326)::geography
                        ) as distance_meters,
                        ST_AsGeoJSON(footprint) as geometry,
                        properties
                    FROM buildings
                    WHERE ST_DWithin(
                        footprint::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography,
                        :radius
                    )
                    ORDER BY distance_meters
                    LIMIT :limit
                """)
                
                result = await db.execute(query, {
                    "point_wkt": point_wkt,
                    "radius": radius_meters,
                    "limit": limit,
                })
                
                for row in result.fetchall():
                    all_features.append({
                        "category": "building",
                        "id": row.id,
                        "name": row.name,
                        "type": row.building_type,
                        "height_meters": float(row.height) if row.height else None,
                        "levels": row.levels,
                        "distance_meters": float(row.distance_meters),
                        "geometry": row.geometry,
                        "properties": row.properties,
                    })
            
            # Roads
            if 'roads' in feature_types:
                query = text("""
                    SELECT 
                        'road' as feature_type,
                        id,
                        name,
                        road_type,
                        surface,
                        lanes,
                        max_speed_kph,
                        oneway,
                        ST_Distance(
                            geom::geography,
                            ST_GeomFromText(:point_wkt, 4326)::geography
                        ) as distance_meters,
                        ST_AsGeoJSON(geom) as geometry,
                        properties
                    FROM roads
                    WHERE ST_DWithin(
                        geom::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography,
                        :radius
                    )
                    ORDER BY distance_meters
                    LIMIT :limit
                """)
                
                result = await db.execute(query, {
                    "point_wkt": point_wkt,
                    "radius": radius_meters,
                    "limit": limit,
                })
                
                for row in result.fetchall():
                    all_features.append({
                        "category": "road",
                        "id": row.id,
                        "name": row.name,
                        "type": row.road_type,
                        "surface": row.surface,
                        "lanes": row.lanes,
                        "max_speed_kph": row.max_speed_kph,
                        "oneway": row.oneway,
                        "distance_meters": float(row.distance_meters),
                        "geometry": row.geometry,
                        "properties": row.properties,
                    })
            
            # Places (cities, towns, villages)
            if 'places' in feature_types:
                query = text("""
                    SELECT 
                        'place' as feature_type,
                        id,
                        name,
                        place_type,
                        population,
                        ST_Distance(
                            geom::geography,
                            ST_GeomFromText(:point_wkt, 4326)::geography
                        ) as distance_meters,
                        ST_AsGeoJSON(geom) as geometry,
                        properties
                    FROM places
                    WHERE ST_DWithin(
                        geom::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography,
                        :radius
                    )
                    ORDER BY distance_meters
                    LIMIT :limit
                """)
                
                result = await db.execute(query, {
                    "point_wkt": point_wkt,
                    "radius": radius_meters,
                    "limit": limit,
                })
                
                for row in result.fetchall():
                    all_features.append({
                        "category": "place",
                        "id": row.id,
                        "name": row.name,
                        "type": row.place_type,
                        "population": row.population,
                        "distance_meters": float(row.distance_meters),
                        "geometry": row.geometry,
                        "properties": row.properties,
                    })
            
            # POIs (amenities, shops, tourism)
            if 'pois' in feature_types:
                query = text("""
                    SELECT 
                        'poi' as feature_type,
                        id,
                        name,
                        poi_type,
                        category,
                        ST_Distance(
                            geom::geography,
                            ST_GeomFromText(:point_wkt, 4326)::geography
                        ) as distance_meters,
                        ST_AsGeoJSON(geom) as geometry,
                        properties
                    FROM pois
                    WHERE ST_DWithin(
                        geom::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography,
                        :radius
                    )
                    ORDER BY distance_meters
                    LIMIT :limit
                """)
                
                result = await db.execute(query, {
                    "point_wkt": point_wkt,
                    "radius": radius_meters,
                    "limit": limit,
                })
                
                for row in result.fetchall():
                    all_features.append({
                        "category": "poi",
                        "id": row.id,
                        "name": row.name,
                        "type": row.poi_type,
                        "poi_category": row.category,
                        "distance_meters": float(row.distance_meters),
                        "geometry": row.geometry,
                        "properties": row.properties,
                    })
            
            # Water features
            if 'water' in feature_types:
                query = text("""
                    SELECT 
                        'water' as feature_type,
                        id,
                        name,
                        water_type,
                        ST_Distance(
                            geom::geography,
                            ST_GeomFromText(:point_wkt, 4326)::geography
                        ) as distance_meters,
                        ST_AsGeoJSON(geom) as geometry,
                        properties
                    FROM water_features
                    WHERE ST_DWithin(
                        geom::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography,
                        :radius
                    )
                    ORDER BY distance_meters
                    LIMIT :limit
                """)
                
                result = await db.execute(query, {
                    "point_wkt": point_wkt,
                    "radius": radius_meters,
                    "limit": limit,
                })
                
                for row in result.fetchall():
                    all_features.append({
                        "category": "water",
                        "id": row.id,
                        "name": row.name,
                        "type": row.water_type,
                        "distance_meters": float(row.distance_meters),
                        "geometry": row.geometry,
                        "properties": row.properties,
                    })
            
            # Land cover
            if 'landcover' in feature_types:
                query = text("""
                    SELECT 
                        'landcover' as feature_type,
                        id,
                        name,
                        land_type,
                        ST_Distance(
                            geom::geography,
                            ST_GeomFromText(:point_wkt, 4326)::geography
                        ) as distance_meters,
                        ST_AsGeoJSON(geom) as geometry,
                        properties
                    FROM land_cover
                    WHERE ST_DWithin(
                        geom::geography,
                        ST_GeomFromText(:point_wkt, 4326)::geography,
                        :radius
                    )
                    ORDER BY distance_meters
                    LIMIT :limit
                """)
                
                result = await db.execute(query, {
                    "point_wkt": point_wkt,
                    "radius": radius_meters,
                    "limit": limit,
                })
                
                for row in result.fetchall():
                    all_features.append({
                        "category": "landcover",
                        "id": row.id,
                        "name": row.name,
                        "type": row.land_type,
                        "distance_meters": float(row.distance_meters),
                        "geometry": row.geometry,
                        "properties": row.properties,
                    })
        
        # Sort by distance
        all_features.sort(key=lambda x: x["distance_meters"])
        
        return all_features[:limit * 2]  # Return up to 2x limit across all types

    async def get_containing_regions(
        self,
        lat: float,
        lon: float,
    ) -> list[dict[str, Any]]:
        """
        Find administrative boundaries containing a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            List of containing regions (country, state, city, etc.)
        """
        point_wkt = f"POINT({lon} {lat})"
        
        async with async_session_maker() as db:
            query = text("""
                SELECT 
                    id,
                    name,
                    boundary_type,
                    admin_level,
                    ST_AsGeoJSON(geom) as geometry,
                    ST_Area(geom::geography) / 1000000 as area_km2
                FROM boundaries
                WHERE ST_Contains(
                    geom,
                    ST_GeomFromText(:point_wkt, 4326)
                )
                ORDER BY admin_level, area_km2
            """)
            
            result = await db.execute(query, {"point_wkt": point_wkt})
            
            regions = []
            for row in result.fetchall():
                regions.append({
                    "id": row.id,
                    "name": row.name,
                    "type": row.boundary_type,
                    "admin_level": row.admin_level,
                    "area_km2": float(row.area_km2) if row.area_km2 else None,
                    "geometry": row.geometry,
                })
            
            return regions

    async def get_location_context(
        self,
        lat: float,
        lon: float,
        radius_meters: float = 5000,
    ) -> dict[str, Any]:
        """
        Get comprehensive context about a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_meters: Search radius for nearby features
            
        Returns:
            Complete location context with all feature types
        """
        import asyncio
        
        # Parallel queries
        nearby_task = self.get_nearby_features(lat, lon, radius_meters, limit=20)
        regions_task = self.get_containing_regions(lat, lon)
        
        nearby_features, containing_regions = await asyncio.gather(
            nearby_task,
            regions_task,
        )
        
        # Summarize by category
        summary = {}
        for feature in nearby_features:
            category = feature["category"]
            summary[category] = summary.get(category, 0) + 1
        
        return {
            "lat": lat,
            "lon": lon,
            "nearby_features": nearby_features,
            "containing_regions": containing_regions,
            "summary": summary,
            "total_features": len(nearby_features),
        }
    
    async def get_containing_regions(
        self,
        lat: float,
        lon: float,
    ) -> list[dict[str, Any]]:
        """
        Get administrative boundaries containing a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            List of containing regions (country → state → city → neighborhood)
        """
        point_wkt = f"POINT({lon} {lat})"
        
        async with async_session_maker() as db:
            query = text("""
                SELECT
                    id,
                    name,
                    boundary_type,
                    admin_level,
                    ST_AsGeoJSON(geom) as geometry,
                    ST_Area(geom::geography) / 1000000 as area_km2
                FROM boundaries
                WHERE ST_Contains(
                    geom,
                    ST_GeomFromText(:point_wkt, 4326)
                )
                ORDER BY admin_level, area_km2
            """)
            
            result = await db.execute(query, {"point_wkt": point_wkt})
            
            regions = []
            for row in result.fetchall():
                regions.append({
                    "id": row.id,
                    "name": row.name,
                    "type": row.boundary_type,
                    "admin_level": row.admin_level,
                    "geometry": row.geometry,
                    "area_km2": float(row.area_km2) if row.area_km2 else None,
                })
            
            return regions

    async def find_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        mode: str = "car",
    ) -> dict[str, Any]:
        """
        Find route between two points using road network.
        
        Args:
            start_lat: Start latitude
            start_lon: Start longitude
            end_lat: End latitude
            end_lon: End longitude
            mode: Transport mode (car, bike, walk)
            
        Returns:
            Route with waypoints, distance, estimated time
            
        Note: Requires pgRouting extension - placeholder for now
        """
        return {
            "status": "not_implemented",
            "message": "Road network routing requires pgRouting extension",
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
            "mode": mode,
        }


# Singleton
_earthlink_geo_source = None


def get_earthlink_geo_source() -> EarthlinkGeoSource:
    """Get or create EarthlinkGeoSource singleton."""
    global _earthlink_geo_source
    if _earthlink_geo_source is None:
        _earthlink_geo_source = EarthlinkGeoSource()
    return _earthlink_geo_source
