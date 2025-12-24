"""Geospatial API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ..knowledge import get_geo_source


router = APIRouter()


@router.get("/nearby")
async def get_nearby_features(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    radius_meters: float = Query(1000, description="Search radius in meters", ge=10, le=100000),
    limit: int = Query(10, description="Maximum results", ge=1, le=100),
) -> dict[str, Any]:
    """
    Get geographic features near a point.
    
    Returns nearby landmarks, POIs, and geographic features.
    """
    geo = get_geo_source()
    
    try:
        features = await geo.get_nearby_features(lat, lon, radius_meters, limit)
        return {
            "lat": lat,
            "lon": lon,
            "radius_meters": radius_meters,
            "features": features,
            "count": len(features),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying nearby features: {str(e)}")


@router.get("/region/{region_name}")
async def get_region(region_name: str) -> dict[str, Any]:
    """
    Get information about a geographic region.
    
    Returns boundaries, area, centroid, and metadata.
    """
    geo = get_geo_source()
    
    try:
        region = await geo.query_region(region_name)
        
        if not region:
            raise HTTPException(status_code=404, detail=f"Region '{region_name}' not found")
        
        return region
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying region: {str(e)}")


@router.get("/context")
async def get_location_context(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
) -> dict[str, Any]:
    """
    Get comprehensive context about a location.
    
    Returns nearby features, containing regions, and geographic context.
    """
    geo = get_geo_source()
    
    try:
        context = await geo.get_location_context(lat, lon)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting location context: {str(e)}")


@router.post("/query")
async def spatial_query(
    query_type: str = Query(..., description="Query type: within, intersects, contains, nearest"),
    params: dict[str, Any] = ...,
) -> dict[str, Any]:
    """
    Execute custom spatial query.
    
    Supported types:
    - within: Find features within polygon
    - intersects: Find features intersecting geometry
    - contains: Find regions containing point
    - nearest: Find N nearest features to point
    """
    geo = get_geo_source()
    
    valid_types = ["within", "intersects", "contains", "nearest"]
    if query_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid query type. Must be one of: {', '.join(valid_types)}"
        )
    
    try:
        results = await geo.spatial_query(query_type, params)
        return {
            "query_type": query_type,
            "params": params,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing spatial query: {str(e)}")


@router.get("/agent/{agent_id}/location")
async def get_agent_location(agent_id: UUID) -> dict[str, Any]:
    """
    Get current location and geographic context for an agent.
    
    Returns agent position and nearby features.
    """
    # TODO: Get agent state from simulation
    # For now, return placeholder
    raise HTTPException(status_code=501, detail="Not implemented - need agent state integration")
