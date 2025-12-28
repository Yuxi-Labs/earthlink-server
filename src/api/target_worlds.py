"""Target worlds management API."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_db


router = APIRouter(prefix="/api/v1/target-worlds", tags=["Target Worlds"])


class TargetWorldCreate(BaseModel):
    """Request to create a new target world."""
    
    name: str = Field(..., min_length=1, max_length=255)
    world_type: str = Field(..., pattern="^(earth_region|virtual_world|game_world|simulation)$")
    description: str | None = None
    region_code: str | None = Field(None, max_length=10)
    bounding_box: dict[str, float] | None = Field(
        None,
        description="Geographic bounds: {min_lat, max_lat, min_lon, max_lon}"
    )
    complexity_score: float | None = Field(None, ge=0.0, le=1.0)
    feature_count: int | None = Field(None, ge=0)
    required_knowledge_domains: list[str] | None = None
    deployment_status: str = Field(default="pending", pattern="^(pending|active|inactive|archived)$")
    metadata: dict[str, Any] | None = None


class TargetWorldResponse(BaseModel):
    """Target world details."""
    
    id: str
    name: str
    world_type: str
    description: str | None
    region_code: str | None
    bounding_box: dict[str, float] | None
    complexity_score: float | None
    feature_count: int | None
    required_knowledge_domains: list[str] | None
    deployment_status: str
    metadata: dict[str, Any] | None
    created_at: str
    updated_at: str


class TargetWorldUpdate(BaseModel):
    """Update target world fields."""
    
    name: str | None = None
    description: str | None = None
    complexity_score: float | None = Field(None, ge=0.0, le=1.0)
    deployment_status: str | None = Field(None, pattern="^(pending|active|inactive|archived)$")
    metadata: dict[str, Any] | None = None


@router.post("", response_model=TargetWorldResponse, status_code=201)
async def create_target_world(
    world: TargetWorldCreate,
    db: AsyncSession = Depends(get_db),
) -> TargetWorldResponse:
    """
    Create a new target world for agent deployment.
    
    Target worlds are environments where trained agents can be deployed:
    - earth_region: Real-world geographic areas
    - virtual_world: Simulated 3D environments
    - game_world: Game engines or metaverses
    - simulation: Custom simulation environments
    """
    world_id = uuid4()
    
    # Build bounding box JSON
    bbox_json = None
    if world.bounding_box:
        bbox_json = world.bounding_box
    
    # Build knowledge domains JSON
    domains_json = None
    if world.required_knowledge_domains:
        domains_json = world.required_knowledge_domains
    
    # Build metadata JSON
    metadata_json = None
    if world.metadata:
        metadata_json = world.metadata
    
    insert_query = text("""
        INSERT INTO target_worlds (
            id, name, world_type, description, region_code,
            bounding_box, complexity_score, feature_count,
            required_knowledge_domains, deployment_status, metadata
        ) VALUES (
            :id, :name, :world_type, :description, :region_code,
            :bounding_box, :complexity_score, :feature_count,
            :required_knowledge_domains, :deployment_status, :metadata
        )
        RETURNING id, name, world_type, description, region_code,
                  bounding_box, complexity_score, feature_count,
                  required_knowledge_domains, deployment_status, metadata,
                  created_at, updated_at
    """)
    
    try:
        result = await db.execute(insert_query, {
            "id": str(world_id),
            "name": world.name,
            "world_type": world.world_type,
            "description": world.description,
            "region_code": world.region_code,
            "bounding_box": bbox_json,
            "complexity_score": world.complexity_score,
            "feature_count": world.feature_count,
            "required_knowledge_domains": domains_json,
            "deployment_status": world.deployment_status,
            "metadata": metadata_json,
        })
        
        row = result.fetchone()
        await db.commit()
        
        return TargetWorldResponse(
            id=str(row.id),
            name=row.name,
            world_type=row.world_type,
            description=row.description,
            region_code=row.region_code,
            bounding_box=row.bounding_box,
            complexity_score=row.complexity_score,
            feature_count=row.feature_count,
            required_knowledge_domains=row.required_knowledge_domains,
            deployment_status=row.deployment_status,
            metadata=row.metadata,
            created_at=row.created_at.isoformat(),
            updated_at=row.updated_at.isoformat(),
        )
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create target world: {str(e)}"
        )


@router.get("", response_model=list[TargetWorldResponse])
async def list_target_worlds(
    world_type: str | None = Query(None, pattern="^(earth_region|virtual_world|game_world|simulation)$"),
    deployment_status: str | None = Query(None, pattern="^(pending|active|inactive|archived)$"),
    region_code: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[TargetWorldResponse]:
    """List target worlds with optional filtering."""
    # Build dynamic WHERE clause
    conditions = []
    params = {"limit": limit, "offset": offset}
    
    if world_type:
        conditions.append("world_type = :world_type")
        params["world_type"] = world_type
    
    if deployment_status:
        conditions.append("deployment_status = :deployment_status")
        params["deployment_status"] = deployment_status
    
    if region_code:
        conditions.append("region_code = :region_code")
        params["region_code"] = region_code
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = text(f"""
        SELECT 
            id, name, world_type, description, region_code,
            bounding_box, complexity_score, feature_count,
            required_knowledge_domains, deployment_status, metadata,
            created_at, updated_at
        FROM target_worlds
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """)
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    return [
        TargetWorldResponse(
            id=str(row.id),
            name=row.name,
            world_type=row.world_type,
            description=row.description,
            region_code=row.region_code,
            bounding_box=row.bounding_box,
            complexity_score=row.complexity_score,
            feature_count=row.feature_count,
            required_knowledge_domains=row.required_knowledge_domains,
            deployment_status=row.deployment_status,
            metadata=row.metadata,
            created_at=row.created_at.isoformat(),
            updated_at=row.updated_at.isoformat(),
        )
        for row in rows
    ]


@router.get("/{world_id}", response_model=TargetWorldResponse)
async def get_target_world(
    world_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> TargetWorldResponse:
    """Get details of a specific target world."""
    query = text("""
        SELECT 
            id, name, world_type, description, region_code,
            bounding_box, complexity_score, feature_count,
            required_knowledge_domains, deployment_status, metadata,
            created_at, updated_at
        FROM target_worlds
        WHERE id = :world_id
    """)
    
    result = await db.execute(query, {"world_id": str(world_id)})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Target world {world_id} not found"
        )
    
    return TargetWorldResponse(
        id=str(row.id),
        name=row.name,
        world_type=row.world_type,
        description=row.description,
        region_code=row.region_code,
        bounding_box=row.bounding_box,
        complexity_score=row.complexity_score,
        feature_count=row.feature_count,
        required_knowledge_domains=row.required_knowledge_domains,
        deployment_status=row.deployment_status,
        metadata=row.metadata,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
    )


@router.patch("/{world_id}", response_model=TargetWorldResponse)
async def update_target_world(
    world_id: UUID,
    updates: TargetWorldUpdate,
    db: AsyncSession = Depends(get_db),
) -> TargetWorldResponse:
    """Update target world fields."""
    # Build dynamic SET clause
    set_clauses = []
    params = {"world_id": str(world_id)}
    
    if updates.name is not None:
        set_clauses.append("name = :name")
        params["name"] = updates.name
    
    if updates.description is not None:
        set_clauses.append("description = :description")
        params["description"] = updates.description
    
    if updates.complexity_score is not None:
        set_clauses.append("complexity_score = :complexity_score")
        params["complexity_score"] = updates.complexity_score
    
    if updates.deployment_status is not None:
        set_clauses.append("deployment_status = :deployment_status")
        params["deployment_status"] = updates.deployment_status
    
    if updates.metadata is not None:
        set_clauses.append("metadata = :metadata")
        params["metadata"] = updates.metadata
    
    if not set_clauses:
        raise HTTPException(
            status_code=400,
            detail="No fields to update"
        )
    
    set_clauses.append("updated_at = NOW()")
    set_clause = ", ".join(set_clauses)
    
    update_query = text(f"""
        UPDATE target_worlds
        SET {set_clause}
        WHERE id = :world_id
        RETURNING id, name, world_type, description, region_code,
                  bounding_box, complexity_score, feature_count,
                  required_knowledge_domains, deployment_status, metadata,
                  created_at, updated_at
    """)
    
    try:
        result = await db.execute(update_query, params)
        row = result.fetchone()
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Target world {world_id} not found"
            )
        
        await db.commit()
        
        return TargetWorldResponse(
            id=str(row.id),
            name=row.name,
            world_type=row.world_type,
            description=row.description,
            region_code=row.region_code,
            bounding_box=row.bounding_box,
            complexity_score=row.complexity_score,
            feature_count=row.feature_count,
            required_knowledge_domains=row.required_knowledge_domains,
            deployment_status=row.deployment_status,
            metadata=row.metadata,
            created_at=row.created_at.isoformat(),
            updated_at=row.updated_at.isoformat(),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update target world: {str(e)}"
        )


@router.delete("/{world_id}", status_code=204)
async def delete_target_world(
    world_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a target world (soft delete - archive)."""
    update_query = text("""
        UPDATE target_worlds
        SET deployment_status = 'archived',
            updated_at = NOW()
        WHERE id = :world_id
    """)
    
    result = await db.execute(update_query, {"world_id": str(world_id)})
    
    if result.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Target world {world_id} not found"
        )
    
    await db.commit()
