"""Metrics collection and time-series storage."""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and stores time-series metrics in TimescaleDB.
    
    Handles:
    - Agent performance metrics (learning rate, rewards, etc.)
    - Simulation-level metrics (step count, throughput, etc.)
    - World metrics (activity, resource usage)
    - Readiness metrics (for target world preparation)
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def record_agent_metric(
        self,
        agent_id: UUID,
        metric_name: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a single agent metric."""
        try:
            query = text("""
                INSERT INTO agent_metrics_ts (time, agent_id, metric_name, metric_value, metadata)
                VALUES (NOW(), :agent_id, :metric_name, :metric_value, :metadata)
            """)
            
            await self.db.execute(
                query,
                {
                    "agent_id": agent_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "metadata": metadata or {},
                },
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to record agent metric: {e}")
            await self.db.rollback()

    async def record_agent_metrics_batch(
        self,
        agent_id: UUID,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record multiple agent metrics at once."""
        for metric_name, metric_value in metrics.items():
            await self.record_agent_metric(agent_id, metric_name, metric_value, metadata)

    async def record_simulation_metric(
        self,
        metric_name: str,
        metric_value: float,
        simulation_id: UUID | None = None,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a simulation-level metric."""
        try:
            query = text("""
                INSERT INTO simulation_metrics_ts 
                (time, simulation_id, metric_name, metric_value, step, metadata)
                VALUES (NOW(), :simulation_id, :metric_name, :metric_value, :step, :metadata)
            """)
            
            await self.db.execute(
                query,
                {
                    "simulation_id": simulation_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "step": step,
                    "metadata": metadata or {},
                },
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to record simulation metric: {e}")
            await self.db.rollback()

    async def record_world_metric(
        self,
        world_id: UUID,
        metric_name: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a world metric."""
        try:
            query = text("""
                INSERT INTO world_metrics_ts (time, world_id, metric_name, metric_value, metadata)
                VALUES (NOW(), :world_id, :metric_name, :metric_value, :metadata)
            """)
            
            await self.db.execute(
                query,
                {
                    "world_id": world_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "metadata": metadata or {},
                },
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to record world metric: {e}")
            await self.db.rollback()

    async def record_readiness_metric(
        self,
        agent_id: UUID,
        target_world_id: UUID,
        metric_name: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an agent readiness metric for a target world."""
        try:
            query = text("""
                INSERT INTO readiness_metrics_ts 
                (time, agent_id, target_world_id, metric_name, metric_value, metadata)
                VALUES (NOW(), :agent_id, :target_world_id, :metric_name, :metric_value, :metadata)
            """)
            
            await self.db.execute(
                query,
                {
                    "agent_id": agent_id,
                    "target_world_id": target_world_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "metadata": metadata or {},
                },
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to record readiness metric: {e}")
            await self.db.rollback()

    async def get_agent_metrics(
        self,
        agent_id: UUID,
        metric_names: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get agent metrics over time."""
        conditions = ["agent_id = :agent_id"]
        params: dict[str, Any] = {"agent_id": agent_id}
        
        if metric_names:
            conditions.append("metric_name = ANY(:metric_names)")
            params["metric_names"] = metric_names
        
        if start_time:
            conditions.append("time >= :start_time")
            params["start_time"] = start_time
        
        if end_time:
            conditions.append("time <= :end_time")
            params["end_time"] = end_time
        
        where_clause = " AND ".join(conditions)
        
        query = text(f"""
            SELECT time, agent_id, metric_name, metric_value, metadata
            FROM agent_metrics_ts
            WHERE {where_clause}
            ORDER BY time DESC
            LIMIT :limit
        """)
        
        params["limit"] = limit
        
        result = await self.db.execute(query, params)
        rows = result.fetchall()
        
        return [
            {
                "time": row[0],
                "agent_id": str(row[1]),
                "metric_name": row[2],
                "metric_value": row[3],
                "metadata": row[4] or {},
            }
            for row in rows
        ]

    async def get_agent_metrics_aggregated(
        self,
        agent_id: UUID,
        metric_name: str,
        interval: str = "1 hour",
        agg_function: str = "avg",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated agent metrics (e.g., hourly averages)."""
        conditions = ["agent_id = :agent_id", "metric_name = :metric_name"]
        params: dict[str, Any] = {
            "agent_id": agent_id,
            "metric_name": metric_name,
            "interval": interval,
        }
        
        if start_time:
            conditions.append("time >= :start_time")
            params["start_time"] = start_time
        
        if end_time:
            conditions.append("time <= :end_time")
            params["end_time"] = end_time
        
        where_clause = " AND ".join(conditions)
        
        agg_func = {
            "avg": "AVG",
            "min": "MIN",
            "max": "MAX",
            "sum": "SUM",
            "count": "COUNT",
        }.get(agg_function, "AVG")
        
        query = text(f"""
            SELECT 
                time_bucket(:interval::interval, time) AS bucket,
                {agg_func}(metric_value) AS value
            FROM agent_metrics_ts
            WHERE {where_clause}
            GROUP BY bucket
            ORDER BY bucket DESC
        """)
        
        result = await self.db.execute(query, params)
        rows = result.fetchall()
        
        return [
            {
                "time": row[0],
                "value": float(row[1]) if row[1] is not None else 0.0,
            }
            for row in rows
        ]

    async def get_simulation_metrics(
        self,
        metric_names: list[str] | None = None,
        simulation_id: UUID | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get simulation metrics over time."""
        conditions = []
        params: dict[str, Any] = {}
        
        if simulation_id:
            conditions.append("simulation_id = :simulation_id")
            params["simulation_id"] = simulation_id
        
        if metric_names:
            conditions.append("metric_name = ANY(:metric_names)")
            params["metric_names"] = metric_names
        
        if start_time:
            conditions.append("time >= :start_time")
            params["start_time"] = start_time
        
        if end_time:
            conditions.append("time <= :end_time")
            params["end_time"] = end_time
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = text(f"""
            SELECT time, simulation_id, metric_name, metric_value, step, metadata
            FROM simulation_metrics_ts
            WHERE {where_clause}
            ORDER BY time DESC
            LIMIT :limit
        """)
        
        params["limit"] = limit
        
        result = await self.db.execute(query, params)
        rows = result.fetchall()
        
        return [
            {
                "time": row[0],
                "simulation_id": str(row[1]) if row[1] else None,
                "metric_name": row[2],
                "metric_value": row[3],
                "step": row[4],
                "metadata": row[5] or {},
            }
            for row in rows
        ]

    async def get_readiness_score(
        self,
        agent_id: UUID,
        target_world_id: UUID,
    ) -> float:
        """
        Calculate readiness score for an agent/world combination.
        
        Based on plan.md formula:
        readiness_score = weighted_average(
            knowledge_coverage * 0.25,
            novel_scenario_success * 0.25,
            adaptation_speed * 0.20,
            uncertainty_handling * 0.15,
            goal_persistence * 0.15
        )
        """
        # Get latest values for each metric
        metrics_query = text("""
            SELECT DISTINCT ON (metric_name) metric_name, metric_value
            FROM readiness_metrics_ts
            WHERE agent_id = :agent_id AND target_world_id = :target_world_id
            ORDER BY metric_name, time DESC
        """)
        
        result = await self.db.execute(
            metrics_query,
            {"agent_id": agent_id, "target_world_id": target_world_id},
        )
        rows = result.fetchall()
        
        metrics = {row[0]: row[1] for row in rows}
        
        # Calculate weighted score
        score = (
            metrics.get("knowledge_coverage", 0.0) * 0.25
            + metrics.get("novel_scenario_success", 0.0) * 0.25
            + metrics.get("adaptation_speed", 0.0) * 0.20
            + metrics.get("uncertainty_handling", 0.0) * 0.15
            + metrics.get("goal_persistence", 0.0) * 0.15
        )
        
        return score

    async def get_knowledge_logs(
        self,
        agent_id: UUID | None = None,
        sources: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch knowledge acquisition log entries for auditing learning events."""
        conditions = []
        params: dict[str, Any] = {}

        if agent_id:
            conditions.append("agent_id = :agent_id")
            params["agent_id"] = agent_id

        if sources:
            conditions.append("source = ANY(:sources)")
            params["sources"] = sources

        if start_time:
            conditions.append("timestamp >= :start_time")
            params["start_time"] = start_time

        if end_time:
            conditions.append("timestamp <= :end_time")
            params["end_time"] = end_time

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = text(f"""
            SELECT id, agent_id, timestamp, source, topic, content_summary, knowledge_count, metadata
            FROM knowledge_acquisition_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT :limit
        """)

        params["limit"] = limit

        result = await self.db.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "id": str(row[0]),
                "agent_id": str(row[1]),
                "timestamp": row[2],
                "source": row[3],
                "topic": row[4],
                "content_summary": row[5],
                "knowledge_count": row[6],
                "metadata": row[7] or {},
            }
            for row in rows
        ]
