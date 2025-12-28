"""Analytics API endpoints for agent learning insights."""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy import text
from src.db.database import async_session_maker

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/knowledge-acquisition")
async def get_knowledge_acquisition(
    agent_id: Optional[str] = None,
    source: Optional[str] = None,
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get knowledge acquisition logs with full context.
    
    Track WHAT agents learned, WHEN, WHY, WHERE, and quality metrics.
    """
    async with async_session_maker() as db:
        query = """
            SELECT 
                id,
                agent_id,
                source,
                topic,
                content_summary,
                knowledge_count,
                goal_context,
                curiosity_signal,
                lat,
                lon,
                relevance_score,
                novelty_score,
                agent_state,
                metadata,
                created_at
            FROM knowledge_acquisition_log
            WHERE created_at >= NOW() - INTERVAL :hours HOUR
        """
        
        params = {"hours": hours}
        
        if agent_id:
            query += " AND agent_id = :agent_id"
            params["agent_id"] = agent_id
        
        if source:
            query += " AND source LIKE :source"
            params["source"] = f"%{source}%"
        
        query += " ORDER BY created_at DESC LIMIT :limit"
        params["limit"] = limit
        
        result = await db.execute(text(query), params)
        rows = result.fetchall()
        
        return {
            "total": len(rows),
            "acquisitions": [
                {
                    "id": row.id,
                    "agent_id": str(row.agent_id),
                    "what": {
                        "topic": row.topic,
                        "source": row.source,
                        "knowledge_count": row.knowledge_count,
                        "content_summary": row.content_summary[:200] if row.content_summary else None,
                    },
                    "when": row.created_at.isoformat() if row.created_at else None,
                    "why": {
                        "goal_context": row.goal_context,
                        "curiosity_signal": row.curiosity_signal,
                    },
                    "where": {
                        "lat": row.lat,
                        "lon": row.lon,
                    },
                    "quality": {
                        "relevance_score": row.relevance_score,
                        "novelty_score": row.novelty_score,
                    },
                    "context": {
                        "agent_state": row.agent_state,
                        "metadata": row.metadata,
                    },
                }
                for row in rows
            ],
        }


@router.get("/knowledge-stats")
async def get_knowledge_stats(
    agent_id: Optional[str] = None,
    hours: int = Query(default=24, ge=1, le=168),
):
    """Get aggregated knowledge acquisition statistics."""
    async with async_session_maker() as db:
        query = """
            SELECT 
                COUNT(*) as total_acquisitions,
                SUM(knowledge_count) as total_knowledge_items,
                AVG(knowledge_count) as avg_items_per_acquisition,
                AVG(curiosity_signal) as avg_curiosity,
                AVG(relevance_score) as avg_relevance,
                AVG(novelty_score) as avg_novelty,
                COUNT(DISTINCT source) as unique_sources,
                COUNT(DISTINCT topic) as unique_topics
            FROM knowledge_acquisition_log
            WHERE created_at >= NOW() - INTERVAL :hours HOUR
        """
        
        params = {"hours": hours}
        
        if agent_id:
            query += " AND agent_id = :agent_id"
            params["agent_id"] = agent_id
        
        result = await db.execute(text(query), params)
        row = result.fetchone()
        
        if not row:
            return {"error": "No data found"}
        
        return {
            "period_hours": hours,
            "agent_id": agent_id,
            "total_acquisitions": row.total_acquisitions or 0,
            "total_knowledge_items": row.total_knowledge_items or 0,
            "avg_items_per_acquisition": float(row.avg_items_per_acquisition or 0),
            "avg_curiosity": float(row.avg_curiosity or 0),
            "avg_relevance": float(row.avg_relevance or 0),
            "avg_novelty": float(row.avg_novelty or 0),
            "unique_sources": row.unique_sources or 0,
            "unique_topics": row.unique_topics or 0,
        }


@router.get("/topic-distribution")
async def get_topic_distribution(
    agent_id: Optional[str] = None,
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Get distribution of topics explored by agents."""
    async with async_session_maker() as db:
        query = """
            SELECT 
                topic,
                COUNT(*) as exploration_count,
                SUM(knowledge_count) as total_items,
                AVG(curiosity_signal) as avg_curiosity,
                AVG(novelty_score) as avg_novelty
            FROM knowledge_acquisition_log
            WHERE created_at >= NOW() - INTERVAL :hours HOUR
              AND topic IS NOT NULL
        """
        
        params = {"hours": hours}
        
        if agent_id:
            query += " AND agent_id = :agent_id"
            params["agent_id"] = agent_id
        
        query += """
            GROUP BY topic
            ORDER BY exploration_count DESC
            LIMIT :limit
        """
        params["limit"] = limit
        
        result = await db.execute(text(query), params)
        rows = result.fetchall()
        
        return {
            "topics": [
                {
                    "topic": row.topic,
                    "exploration_count": row.exploration_count,
                    "total_items": row.total_items,
                    "avg_curiosity": float(row.avg_curiosity or 0),
                    "avg_novelty": float(row.avg_novelty or 0),
                }
                for row in rows
            ]
        }


@router.get("/exploration-timeline")
async def get_exploration_timeline(
    agent_id: str,
    hours: int = Query(default=24, ge=1, le=168),
):
    """Get chronological timeline of agent's exploration journey."""
    async with async_session_maker() as db:
        query = text("""
            SELECT 
                topic,
                source,
                knowledge_count,
                goal_context,
                curiosity_signal,
                lat,
                lon,
                agent_state,
                created_at
            FROM knowledge_acquisition_log
            WHERE agent_id = :agent_id
              AND created_at >= NOW() - INTERVAL :hours HOUR
            ORDER BY created_at ASC
        """)
        
        result = await db.execute(query, {"agent_id": agent_id, "hours": hours})
        rows = result.fetchall()
        
        return {
            "agent_id": agent_id,
            "timeline": [
                {
                    "timestamp": row.created_at.isoformat(),
                    "topic": row.topic,
                    "source": row.source,
                    "knowledge_count": row.knowledge_count,
                    "why": row.goal_context,
                    "curiosity": row.curiosity_signal,
                    "location": {"lat": row.lat, "lon": row.lon} if row.lat and row.lon else None,
                    "state": row.agent_state,
                }
                for row in rows
            ],
        }
