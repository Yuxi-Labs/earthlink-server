"""Earthlink API endpoints - Query the VW's Earth data."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..knowledge.earthlink import get_earthlink_geo_source


router = APIRouter()


@router.get("/nearby")
async def get_nearby_features(
    lat: float = Query(..., description="Latitude", ge=-90, le=90),
    lon: float = Query(..., description="Longitude", ge=-180, le=180),
    radius_meters: float = Query(1000, description="Search radius in meters", ge=10, le=50000),
    limit: int = Query(10, description="Maximum results per feature type", ge=1, le=100),
    types: str = Query(
        None,
        description="Comma-separated feature types: buildings,roads,places,pois,water,landcover"
    ),
) -> dict[str, Any]:
    """
    Get features near a point from Earthlink (VW with 43M+ OSM features).
    
    Returns nearby buildings, roads, places, POIs, water, and land cover.
    
    Example: `/earthlink/nearby?lat=-23.5506507&lon=-46.6333824&radius_meters=1000`
    """
    geo = get_earthlink_geo_source()
    
    feature_types = types.split(',') if types else None
    
    try:
        features = await geo.get_nearby_features(
            lat, lon, radius_meters, limit, feature_types
        )
        
        # Organize by category
        by_category = {}
        for feature in features:
            category = feature["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        return {
            "lat": lat,
            "lon": lon,
            "radius_meters": radius_meters,
            "features": features,
            "by_category": by_category,
            "total": len(features),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying Earth data: {str(e)}"
        )


@router.get("/context")
async def get_location_context(
    lat: float = Query(..., description="Latitude", ge=-90, le=90),
    lon: float = Query(..., description="Longitude", ge=-180, le=180),
    radius_meters: float = Query(5000, description="Search radius in meters", ge=100, le=50000),
) -> dict[str, Any]:
    """
    Get comprehensive context about a location from Earthlink.
    
    Returns nearby features, containing regions (boundaries), and summary stats.
    
    Example: `/earthlink/context?lat=-23.5506507&lon=-46.6333824`
    """
    geo = get_earthlink_geo_source()
    
    try:
        context = await geo.get_location_context(lat, lon, radius_meters)
        return context
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting location context: {str(e)}"
        )


@router.get("/regions")
async def get_containing_regions(
    lat: float = Query(..., description="Latitude", ge=-90, le=90),
    lon: float = Query(..., description="Longitude", ge=-180, le=180),
) -> dict[str, Any]:
    """
    Get administrative boundaries containing a point.
    
    Returns nested regions (country → state → city → neighborhood).
    
    Example: `/earthlink/regions?lat=-23.5506507&lon=-46.6333824`
    """
    geo = get_earthlink_geo_source()
    
    try:
        regions = await geo.get_containing_regions(lat, lon)
        return {
            "lat": lat,
            "lon": lon,
            "regions": regions,
            "count": len(regions),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding containing regions: {str(e)}"
        )


@router.get("/stats")
async def get_earthlink_stats() -> dict[str, Any]:
    """
    Get statistics about the Earthlink dataset (VW).
    
    Returns feature counts and coverage information.
    """
    from ..db.database import async_session_maker
    from sqlalchemy import text
    
    try:
        async with async_session_maker() as db:
            # Get counts for each table
            tables = {
                "buildings": "SELECT COUNT(*) FROM buildings",
                "roads": "SELECT COUNT(*) FROM roads",
                "places": "SELECT COUNT(*) FROM places",
                "pois": "SELECT COUNT(*) FROM pois",
                "water_features": "SELECT COUNT(*) FROM water_features",
                "land_cover": "SELECT COUNT(*) FROM land_cover",
                "boundaries": "SELECT COUNT(*) FROM boundaries",
            }
            
            counts = {}
            for table, query_str in tables.items():
                result = await db.execute(text(query_str))
                counts[table] = result.scalar()
            
            # Get bounding box of data
            bbox_query = text("""
                SELECT 
                    ST_XMin(extent) as min_lon,
                    ST_YMin(extent) as min_lat,
                    ST_XMax(extent) as max_lon,
                    ST_YMax(extent) as max_lat
                FROM (
                    SELECT ST_Extent(footprint::geometry) as extent FROM buildings
                ) subquery
            """)
            
            result = await db.execute(bbox_query)
            bbox = result.fetchone()
            
            return {
                "feature_counts": counts,
                "total_features": sum(counts.values()),
                "coverage": {
                    "region": "South America",
                    "bounding_box": {
                        "min_lat": float(bbox.min_lat) if bbox and bbox.min_lat else None,
                        "max_lat": float(bbox.max_lat) if bbox and bbox.max_lat else None,
                        "min_lon": float(bbox.min_lon) if bbox and bbox.min_lon else None,
                        "max_lon": float(bbox.max_lon) if bbox and bbox.max_lon else None,
                    } if bbox else None,
                },
                "data_source": "OpenStreetMap via Geofabrik",
                "last_updated": "2025-12-27",
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )


@router.get("/route")
async def find_route(
    start_lat: float = Query(..., description="Start latitude", ge=-90, le=90),
    start_lon: float = Query(..., description="Start longitude", ge=-180, le=180),
    end_lat: float = Query(..., description="End latitude", ge=-90, le=90),
    end_lon: float = Query(..., description="End longitude", ge=-180, le=180),
    mode: str = Query("car", description="Transport mode: car, bike, walk"),
) -> dict[str, Any]:
    """
    Find route between two points (requires pgRouting - not yet implemented).
    
    Future enhancement for navigation between coordinates.
    """
    geo = get_earthlink_geo_source()
    
    try:
        route = await geo.find_route(start_lat, start_lon, end_lat, end_lon, mode)
        return route
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding route: {str(e)}"
        )
