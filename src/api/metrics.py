"""Metrics API endpoints."""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_db
from src.metrics import MetricsCollector

router = APIRouter()


# -------------------------------------------------------------------------
# Request/Response Schemas
# -------------------------------------------------------------------------

class MetricRecord(BaseModel):
    """Schema for recording a metric."""
    metric_name: str
    metric_value: float
    metadata: dict[str, Any] | None = None


class MetricsBatch(BaseModel):
    """Schema for batch metric recording."""
    metrics: dict[str, float]
    metadata: dict[str, Any] | None = None


class ReadinessMetricRecord(BaseModel):
    """Schema for recording readiness metric."""
    target_world_id: UUID
    metric_name: str
    metric_value: float
    metadata: dict[str, Any] | None = None


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.post("/agents/{agent_id}/metrics")
async def record_agent_metric(
    agent_id: UUID,
    metric: MetricRecord,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Record a single agent metric."""
    collector = MetricsCollector(db)
    await collector.record_agent_metric(
        agent_id=agent_id,
        metric_name=metric.metric_name,
        metric_value=metric.metric_value,
        metadata=metric.metadata,
    )
    return {"message": "Metric recorded"}


@router.post("/agents/{agent_id}/metrics/batch")
async def record_agent_metrics_batch(
    agent_id: UUID,
    batch: MetricsBatch,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Record multiple agent metrics at once."""
    collector = MetricsCollector(db)
    await collector.record_agent_metrics_batch(
        agent_id=agent_id,
        metrics=batch.metrics,
        metadata=batch.metadata,
    )
    return {"message": f"{len(batch.metrics)} metrics recorded"}


@router.get("/agents/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: UUID,
    metric_names: list[str] = Query(None),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(1000, le=10000),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get agent metrics over time."""
    collector = MetricsCollector(db)
    return await collector.get_agent_metrics(
        agent_id=agent_id,
        metric_names=metric_names,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


@router.get("/agents/{agent_id}/metrics/{metric_name}/aggregated")
async def get_agent_metrics_aggregated(
    agent_id: UUID,
    metric_name: str,
    interval: str = Query("1 hour", description="Time bucket interval (e.g., '1 hour', '5 minutes')"),
    agg_function: str = Query("avg", description="Aggregation function: avg, min, max, sum, count"),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get aggregated agent metrics (e.g., hourly averages)."""
    if agg_function not in ["avg", "min", "max", "sum", "count"]:
        raise HTTPException(status_code=400, detail="Invalid aggregation function")
    
    collector = MetricsCollector(db)
    return await collector.get_agent_metrics_aggregated(
        agent_id=agent_id,
        metric_name=metric_name,
        interval=interval,
        agg_function=agg_function,
        start_time=start_time,
        end_time=end_time,
    )


@router.post("/simulation/metrics")
async def record_simulation_metric(
    metric: MetricRecord,
    simulation_id: UUID | None = None,
    step: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Record a simulation-level metric."""
    collector = MetricsCollector(db)
    await collector.record_simulation_metric(
        metric_name=metric.metric_name,
        metric_value=metric.metric_value,
        simulation_id=simulation_id,
        step=step,
        metadata=metric.metadata,
    )
    return {"message": "Metric recorded"}


@router.get("/simulation/metrics")
async def get_simulation_metrics(
    metric_names: list[str] = Query(None),
    simulation_id: UUID | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(1000, le=10000),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get simulation metrics over time."""
    collector = MetricsCollector(db)
    return await collector.get_simulation_metrics(
        metric_names=metric_names,
        simulation_id=simulation_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


@router.post("/worlds/{world_id}/metrics")
async def record_world_metric(
    world_id: UUID,
    metric: MetricRecord,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Record a world metric."""
    collector = MetricsCollector(db)
    await collector.record_world_metric(
        world_id=world_id,
        metric_name=metric.metric_name,
        metric_value=metric.metric_value,
        metadata=metric.metadata,
    )
    return {"message": "Metric recorded"}


@router.post("/agents/{agent_id}/readiness")
async def record_readiness_metric(
    agent_id: UUID,
    metric: ReadinessMetricRecord,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Record an agent readiness metric for a target world."""
    collector = MetricsCollector(db)
    await collector.record_readiness_metric(
        agent_id=agent_id,
        target_world_id=metric.target_world_id,
        metric_name=metric.metric_name,
        metric_value=metric.metric_value,
        metadata=metric.metadata,
    )
    return {"message": "Readiness metric recorded"}


@router.get("/agents/{agent_id}/readiness/{target_world_id}")
async def get_readiness_score(
    agent_id: UUID,
    target_world_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get agent readiness score for a target world."""
    collector = MetricsCollector(db)
    score = await collector.get_readiness_score(agent_id, target_world_id)
    
    # Determine deployment readiness
    status = "not_ready"
    if score >= 0.85:
        status = "deployment_ready"
    elif score >= 0.70:
        status = "testing_ready"
    elif score >= 0.50:
        status = "in_progress"
    
    return {
        "agent_id": str(agent_id),
        "target_world_id": str(target_world_id),
        "readiness_score": score,
        "status": status,
        "thresholds": {
            "testing_ready": 0.70,
            "deployment_ready": 0.85,
        },
    }
