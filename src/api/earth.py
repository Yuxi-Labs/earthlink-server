"""Earth API endpoints.

The simulation owns Earth as its world state. This API provides:
- Simulation state (observation/action spaces, reset)
- Geographic data queries (nearby features, regions, context)
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy import text

from ..data.worlds.base.earth import get_geo_source

router = APIRouter()


# -------------------------------------------------------------------------
# Dependency: Get simulation runner
# -------------------------------------------------------------------------

def get_simulation():
    """Get the global simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.get("/", response_model=dict[str, Any])
async def get_earth(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Get Earth state and metadata."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")
    
    if simulation.world is None:
        raise HTTPException(status_code=503, detail="Earth not initialized")
    
    return simulation.get_world_state()


@router.get("/state", response_model=dict[str, Any])
async def get_earth_state(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Get current Earth state."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")
    
    if simulation.world is None:
        return {"loaded": False}
    
    return {
        "loaded": simulation.world._is_loaded,
        "metadata": simulation.world.get_metadata(),
        "current_step": simulation.world._current_step,
    }


@router.get("/observation-space", response_model=dict[str, Any])
async def get_observation_space(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Get observation space specification."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")
    
    if simulation.world is None:
        raise HTTPException(status_code=503, detail="Earth not initialized")
    
    return simulation.world.get_observation_space()


@router.get("/action-space", response_model=dict[str, Any])
async def get_action_space(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Get action space specification."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")
    
    if simulation.world is None:
        raise HTTPException(status_code=503, detail="Earth not initialized")
    
    return simulation.world.get_action_space()


@router.post("/reset", response_model=dict[str, Any])
async def reset_earth(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Reset Earth to initial state."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")
    
    if simulation.world is None:
        raise HTTPException(status_code=503, detail="Earth not initialized")
    
    observation = await simulation.world.reset()
    return {
        "message": "Earth reset",
        "initial_observation": observation,
    }


# -------------------------------------------------------------------------
# Geographic Data Queries
# -------------------------------------------------------------------------

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
    Get geographic features near a point (43M+ OSM features).
    
    Returns nearby buildings, roads, places, POIs, water, and land cover.
    
    Example: `/earth/nearby?lat=-23.5506507&lon=-46.6333824&radius_meters=1000`
    """
    geo = get_geo_source()
    
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
    Get comprehensive context about a location.
    
    Returns nearby features, containing regions (boundaries), and summary stats.
    
    Example: `/earth/context?lat=-23.5506507&lon=-46.6333824`
    """
    geo = get_geo_source()
    
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
    
    Example: `/earth/regions?lat=-23.5506507&lon=-46.6333824`
    """
    geo = get_geo_source()
    
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
async def get_earth_stats() -> dict[str, Any]:
    """
    Get statistics about Earth's geographic dataset.
    
    Returns feature counts and coverage information.
    """
    from ..db.database import async_session_maker
    
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
    geo = get_geo_source()
    
    try:
        route = await geo.find_route(start_lat, start_lon, end_lat, end_lon, mode)
        return route
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding route: {str(e)}"
        )
