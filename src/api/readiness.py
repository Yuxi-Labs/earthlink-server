"""Readiness Dashboard API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_db
from src.explorer.readiness import ReadinessCalculator


router = APIRouter(prefix="/api/v1/readiness", tags=["Explorer Readiness"])


class ReadinessResponse(BaseModel):
    """Readiness evaluation response."""
    
    agent_id: str
    target_world_id: str | None = None
    readiness_score: float = Field(ge=0.0, le=1.0)
    is_ready: bool
    readiness_threshold: float
    component_scores: dict[str, float]
    training_metrics: dict[str, Any]
    evaluated_at: str


class ReadinessSummary(BaseModel):
    """Summary of agent readiness."""
    
    agent_id: str
    overall_readiness: float
    is_ready: bool
    world_readiness: list[dict[str, Any]]
    last_evaluated: str | None = None


class WorldReadiness(BaseModel):
    """World deployment readiness summary."""
    
    world_id: str
    world_name: str
    ready_agents: int
    total_agents: int
    avg_readiness_score: float


@router.post("/calculate/{agent_id}", response_model=ReadinessResponse)
async def calculate_agent_readiness(
    agent_id: UUID,
    target_world_id: UUID | None = None,
    readiness_threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    save: bool = Query(default=True, description="Save results to database"),
    db: AsyncSession = Depends(get_db),
) -> ReadinessResponse:
    """
    Calculate and optionally save agent readiness metrics.
    
    Evaluates agent across 6 dimensions:
    - Knowledge coverage
    - Exploration depth
    - Spatial competence
    - Curiosity level
    - Collaboration score
    - Goal achievement
    """
    calculator = ReadinessCalculator(readiness_threshold=readiness_threshold)
    
    try:
        readiness_data = await calculator.calculate_readiness(
            agent_id=agent_id,
            target_world_id=target_world_id,
        )
        
        if save:
            await calculator.save_readiness(readiness_data)
        
        return ReadinessResponse(**readiness_data)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate readiness: {str(e)}"
        )


@router.get("/agents/{agent_id}", response_model=ReadinessSummary)
async def get_agent_readiness_summary(
    agent_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ReadinessSummary:
    """
    Get comprehensive readiness summary for agent.
    
    Includes overall readiness and world-specific scores.
    """
    agent_id_str = str(agent_id)
    
    # Get all readiness records for agent
    query = text("""
        SELECT 
            er.target_world_id,
            tw.name as world_name,
            er.readiness_score,
            er.is_ready,
            er.knowledge_coverage,
            er.exploration_depth,
            er.spatial_competence,
            er.curiosity_level,
            er.collaboration_score,
            er.goal_achievement,
            er.last_evaluated_at
        FROM explorer_readiness er
        LEFT JOIN target_worlds tw ON er.target_world_id = tw.id
        WHERE er.agent_id = :agent_id
        ORDER BY er.readiness_score DESC
    """)
    
    result = await db.execute(query, {"agent_id": agent_id_str})
    rows = result.fetchall()
    
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No readiness data found for agent {agent_id}"
        )
    
    # Overall readiness (general deployment or average)
    general_readiness = next(
        (r for r in rows if r.target_world_id is None),
        None
    )
    
    if general_readiness:
        overall_score = general_readiness.readiness_score
        is_ready = general_readiness.is_ready
        last_evaluated = general_readiness.last_evaluated_at.isoformat() if general_readiness.last_evaluated_at else None
    else:
        # Calculate average from world-specific scores
        overall_score = sum(r.readiness_score for r in rows) / len(rows)
        is_ready = overall_score >= 0.7
        last_evaluated = max(r.last_evaluated_at for r in rows if r.last_evaluated_at).isoformat() if any(r.last_evaluated_at for r in rows) else None
    
    # World-specific readiness
    world_readiness = []
    for row in rows:
        if row.target_world_id:
            world_readiness.append({
                "world_id": str(row.target_world_id),
                "world_name": row.world_name,
                "readiness_score": row.readiness_score,
                "is_ready": row.is_ready,
                "component_scores": {
                    "knowledge_coverage": row.knowledge_coverage,
                    "exploration_depth": row.exploration_depth,
                    "spatial_competence": row.spatial_competence,
                    "curiosity_level": row.curiosity_level,
                    "collaboration_score": row.collaboration_score,
                    "goal_achievement": row.goal_achievement,
                },
            })
    
    return ReadinessSummary(
        agent_id=agent_id_str,
        overall_readiness=round(overall_score, 3),
        is_ready=is_ready,
        world_readiness=world_readiness,
        last_evaluated=last_evaluated,
    )


@router.get("/agents/{agent_id}/worlds/{world_id}", response_model=ReadinessResponse)
async def get_world_specific_readiness(
    agent_id: UUID,
    world_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get agent readiness for specific target world."""
    agent_id_str = str(agent_id)
    world_id_str = str(world_id)
    
    query = text("""
        SELECT 
            er.agent_id,
            er.target_world_id,
            er.readiness_score,
            er.is_ready,
            er.readiness_threshold,
            er.knowledge_coverage,
            er.exploration_depth,
            er.spatial_competence,
            er.curiosity_level,
            er.collaboration_score,
            er.goal_achievement,
            er.total_steps,
            er.knowledge_items_acquired,
            er.unique_topics_explored,
            er.last_evaluated_at
        FROM explorer_readiness er
        WHERE er.agent_id = :agent_id
          AND er.target_world_id = :world_id
    """)
    
    result = await db.execute(query, {
        "agent_id": agent_id_str,
        "world_id": world_id_str,
    })
    
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No readiness data for agent {agent_id} in world {world_id}"
        )
    
    return {
        "agent_id": row.agent_id,
        "target_world_id": row.target_world_id,
        "readiness_score": row.readiness_score,
        "is_ready": row.is_ready,
        "readiness_threshold": row.readiness_threshold,
        "component_scores": {
            "knowledge_coverage": row.knowledge_coverage,
            "exploration_depth": row.exploration_depth,
            "spatial_competence": row.spatial_competence,
            "curiosity_level": row.curiosity_level,
            "collaboration_score": row.collaboration_score,
            "goal_achievement": row.goal_achievement,
        },
        "training_metrics": {
            "total_steps": row.total_steps,
            "knowledge_items_acquired": row.knowledge_items_acquired,
            "unique_topics_explored": row.unique_topics_explored,
        },
        "evaluated_at": row.last_evaluated_at.isoformat() if row.last_evaluated_at else None,
    }


@router.get("/worlds/{world_id}", response_model=WorldReadiness)
async def get_world_readiness(
    world_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> WorldReadiness:
    """
    Get deployment readiness summary for target world.
    
    Shows how many agents are ready for this world.
    """
    world_id_str = str(world_id)
    
    # Get world details
    world_query = text("""
        SELECT name FROM target_worlds WHERE id = :world_id
    """)
    
    world_result = await db.execute(world_query, {"world_id": world_id_str})
    world_row = world_result.fetchone()
    
    if not world_row:
        raise HTTPException(
            status_code=404,
            detail=f"World {world_id} not found"
        )
    
    # Get readiness stats
    stats_query = text("""
        SELECT 
            COUNT(*) as total_agents,
            SUM(CASE WHEN is_ready THEN 1 ELSE 0 END) as ready_agents,
            AVG(readiness_score) as avg_score
        FROM explorer_readiness
        WHERE target_world_id = :world_id
    """)
    
    stats_result = await db.execute(stats_query, {"world_id": world_id_str})
    stats_row = stats_result.fetchone()
    
    return WorldReadiness(
        world_id=world_id_str,
        world_name=world_row.name,
        ready_agents=stats_row.ready_agents or 0,
        total_agents=stats_row.total_agents or 0,
        avg_readiness_score=round(stats_row.avg_score, 3) if stats_row.avg_score else 0.0,
    )


@router.get("/dashboard")
async def get_readiness_dashboard(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get comprehensive readiness dashboard data.
    
    Provides overview of all agents and worlds.
    """
    # Overall stats
    overall_query = text("""
        SELECT 
            COUNT(DISTINCT agent_id) as total_agents,
            SUM(CASE WHEN is_ready THEN 1 ELSE 0 END) as ready_agents,
            AVG(readiness_score) as avg_readiness
        FROM explorer_readiness
        WHERE target_world_id IS NULL OR target_world_id IN (
            SELECT id FROM target_worlds WHERE deployment_status = 'active'
        )
    """)
    
    overall_result = await db.execute(overall_query)
    overall_row = overall_result.fetchone()
    
    # World-specific stats
    worlds_query = text("""
        SELECT 
            tw.id,
            tw.name,
            tw.world_type,
            tw.deployment_status,
            COUNT(er.agent_id) as evaluated_agents,
            SUM(CASE WHEN er.is_ready THEN 1 ELSE 0 END) as ready_agents,
            AVG(er.readiness_score) as avg_readiness
        FROM target_worlds tw
        LEFT JOIN explorer_readiness er ON tw.id = er.target_world_id
        WHERE tw.deployment_status IN ('active', 'pending')
        GROUP BY tw.id, tw.name, tw.world_type, tw.deployment_status
        ORDER BY avg_readiness DESC NULLS LAST
    """)
    
    worlds_result = await db.execute(worlds_query)
    worlds_rows = worlds_result.fetchall()
    
    worlds_summary = [
        {
            "world_id": str(row.id),
            "world_name": row.name,
            "world_type": row.world_type,
            "deployment_status": row.deployment_status,
            "evaluated_agents": row.evaluated_agents or 0,
            "ready_agents": row.ready_agents or 0,
            "avg_readiness": round(row.avg_readiness, 3) if row.avg_readiness else 0.0,
        }
        for row in worlds_rows
    ]
    
    # Top performing agents
    top_agents_query = text("""
        SELECT 
            agent_id,
            AVG(readiness_score) as avg_score,
            MAX(last_evaluated_at) as last_evaluated
        FROM explorer_readiness
        GROUP BY agent_id
        ORDER BY avg_score DESC
        LIMIT 10
    """)
    
    top_result = await db.execute(top_agents_query)
    top_rows = top_result.fetchall()
    
    top_agents = [
        {
            "agent_id": row.agent_id,
            "avg_readiness": round(row.avg_score, 3),
            "last_evaluated": row.last_evaluated.isoformat() if row.last_evaluated else None,
        }
        for row in top_rows
    ]
    
    return {
        "overview": {
            "total_agents": overall_row.total_agents or 0,
            "ready_agents": overall_row.ready_agents or 0,
            "avg_readiness": round(overall_row.avg_readiness, 3) if overall_row.avg_readiness else 0.0,
        },
        "worlds": worlds_summary,
        "top_agents": top_agents,
    }
