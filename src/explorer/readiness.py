"""Explorer readiness calculation and evaluation."""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import async_session_maker


class ReadinessCalculator:
    """
    Calculate agent readiness scores for deployment.
    
    Evaluates agent fitness across multiple dimensions:
    - Knowledge coverage: Breadth of topics explored
    - Exploration depth: Thoroughness of learning
    - Spatial competence: Navigation ability
    - Curiosity level: Active learning drive
    - Collaboration: Multi-agent interaction
    - Goal achievement: Task completion rate
    """
    
    def __init__(self, readiness_threshold: float = 0.7):
        """
        Initialize calculator.
        
        Args:
            readiness_threshold: Minimum score (0-1) for deployment readiness
        """
        self.readiness_threshold = readiness_threshold
    
    async def calculate_readiness(
        self,
        agent_id: UUID | str,
        target_world_id: UUID | str | None = None,
    ) -> dict[str, Any]:
        """
        Calculate comprehensive readiness score for agent.
        
        Args:
            agent_id: Agent UUID
            target_world_id: Optional world-specific readiness
            
        Returns:
            Readiness metrics dictionary
        """
        agent_id_str = str(agent_id)
        
        async with async_session_maker() as db:
            # Get agent training metrics
            training_metrics = await self._get_training_metrics(db, agent_id_str)
            
            # Calculate component scores
            knowledge_coverage = self._calculate_knowledge_coverage(training_metrics)
            exploration_depth = self._calculate_exploration_depth(training_metrics)
            spatial_competence = self._calculate_spatial_competence(training_metrics)
            curiosity_level = self._calculate_curiosity_level(training_metrics)
            collaboration_score = self._calculate_collaboration_score(training_metrics)
            goal_achievement = self._calculate_goal_achievement(training_metrics)
            
            # Weighted overall score
            component_scores = {
                "knowledge_coverage": knowledge_coverage,
                "exploration_depth": exploration_depth,
                "spatial_competence": spatial_competence,
                "curiosity_level": curiosity_level,
                "collaboration_score": collaboration_score,
                "goal_achievement": goal_achievement,
            }
            
            # Weights for overall score
            weights = {
                "knowledge_coverage": 0.25,
                "exploration_depth": 0.20,
                "spatial_competence": 0.15,
                "curiosity_level": 0.15,
                "collaboration_score": 0.10,
                "goal_achievement": 0.15,
            }
            
            readiness_score = sum(
                component_scores[k] * weights[k]
                for k in weights.keys()
            )
            
            is_ready = readiness_score >= self.readiness_threshold
            
            return {
                "agent_id": agent_id_str,
                "target_world_id": str(target_world_id) if target_world_id else None,
                "readiness_score": round(readiness_score, 3),
                "is_ready": is_ready,
                "readiness_threshold": self.readiness_threshold,
                "component_scores": {k: round(v, 3) for k, v in component_scores.items()},
                "training_metrics": training_metrics,
                "evaluated_at": datetime.utcnow().isoformat(),
            }
    
    async def _get_training_metrics(
        self,
        db: AsyncSession,
        agent_id: str,
    ) -> dict[str, Any]:
        """Get agent's training metrics from database."""
        # Knowledge acquisition metrics
        knowledge_query = text("""
            SELECT 
                COUNT(*) as total_acquisitions,
                SUM(knowledge_count) as total_knowledge_items,
                COUNT(DISTINCT topic) as unique_topics,
                AVG(curiosity_signal) as avg_curiosity,
                COUNT(DISTINCT DATE(created_at)) as active_days
            FROM knowledge_acquisition_log
            WHERE agent_id = :agent_id
        """)
        
        result = await db.execute(knowledge_query, {"agent_id": agent_id})
        row = result.fetchone()
        
        if not row or row.total_acquisitions == 0:
            return {
                "total_acquisitions": 0,
                "total_knowledge_items": 0,
                "unique_topics": 0,
                "avg_curiosity": 0.0,
                "active_days": 0,
                "spatial_variance": 0.0,
                "goal_completions": 0,
            }
        
        # Spatial movement metrics (variance in locations = exploration breadth)
        spatial_query = text("""
            SELECT 
                VARIANCE(lat) as lat_variance,
                VARIANCE(lon) as lon_variance
            FROM knowledge_acquisition_log
            WHERE agent_id = :agent_id
              AND lat IS NOT NULL
              AND lon IS NOT NULL
        """)
        
        spatial_result = await db.execute(spatial_query, {"agent_id": agent_id})
        spatial_row = spatial_result.fetchone()
        
        # Calculate spatial variance (higher = more exploration)
        lat_var = spatial_row.lat_variance if spatial_row and spatial_row.lat_variance else 0.0
        lon_var = spatial_row.lon_variance if spatial_row and spatial_row.lon_variance else 0.0
        spatial_variance = float(np.sqrt(lat_var + lon_var)) if (lat_var or lon_var) else 0.0
        
        # Goal achievements (approximate from goal_context presence)
        goal_query = text("""
            SELECT COUNT(DISTINCT goal_context) as goal_completions
            FROM knowledge_acquisition_log
            WHERE agent_id = :agent_id
              AND goal_context IS NOT NULL
              AND goal_context != ''
        """)
        
        goal_result = await db.execute(goal_query, {"agent_id": agent_id})
        goal_row = goal_result.fetchone()
        
        return {
            "total_acquisitions": row.total_acquisitions or 0,
            "total_knowledge_items": row.total_knowledge_items or 0,
            "unique_topics": row.unique_topics or 0,
            "avg_curiosity": float(row.avg_curiosity) if row.avg_curiosity else 0.0,
            "active_days": row.active_days or 0,
            "spatial_variance": spatial_variance,
            "goal_completions": goal_row.goal_completions if goal_row else 0,
        }
    
    def _calculate_knowledge_coverage(self, metrics: dict[str, Any]) -> float:
        """
        Calculate knowledge coverage score (0-1).
        
        Based on unique topics explored and total knowledge items.
        """
        unique_topics = metrics["unique_topics"]
        total_items = metrics["total_knowledge_items"]
        
        # Score based on topic diversity + volume
        # 50+ topics = excellent coverage
        # 100+ items = good volume
        topic_score = min(unique_topics / 50.0, 1.0)
        volume_score = min(total_items / 100.0, 1.0)
        
        return (topic_score * 0.6 + volume_score * 0.4)
    
    def _calculate_exploration_depth(self, metrics: dict[str, Any]) -> float:
        """
        Calculate exploration depth score (0-1).
        
        Based on items per topic and active exploration days.
        """
        total_items = metrics["total_knowledge_items"]
        unique_topics = metrics["unique_topics"]
        active_days = metrics["active_days"]
        
        # Depth = items per topic
        if unique_topics == 0:
            depth_score = 0.0
        else:
            items_per_topic = total_items / unique_topics
            depth_score = min(items_per_topic / 5.0, 1.0)  # 5+ items/topic = deep
        
        # Consistency = active days
        consistency_score = min(active_days / 7.0, 1.0)  # 7+ days = consistent
        
        return (depth_score * 0.7 + consistency_score * 0.3)
    
    def _calculate_spatial_competence(self, metrics: dict[str, Any]) -> float:
        """
        Calculate spatial competence score (0-1).
        
        Based on geographic variance (exploration breadth).
        """
        spatial_variance = metrics["spatial_variance"]
        
        # Higher variance = explored more locations
        # Normalize to 0-1 scale (variance > 10 = excellent)
        return min(spatial_variance / 10.0, 1.0)
    
    def _calculate_curiosity_level(self, metrics: dict[str, Any]) -> float:
        """
        Calculate curiosity level score (0-1).
        
        Based on average curiosity signal from exploration.
        """
        avg_curiosity = metrics["avg_curiosity"]
        
        # Curiosity signal is already 0-1
        return max(0.0, min(avg_curiosity, 1.0))
    
    def _calculate_collaboration_score(self, metrics: dict[str, Any]) -> float:
        """
        Calculate collaboration score (0-1).
        
        Currently not tracked - placeholder for future multi-agent features.
        """
        # TODO: Track agent-to-agent messages, shared knowledge, joint goals
        return 0.5  # Neutral score
    
    def _calculate_goal_achievement(self, metrics: dict[str, Any]) -> float:
        """
        Calculate goal achievement score (0-1).
        
        Based on number of distinct goals pursued.
        """
        goal_completions = metrics["goal_completions"]
        
        # 10+ goals = high achievement
        return min(goal_completions / 10.0, 1.0)
    
    async def save_readiness(
        self,
        readiness_data: dict[str, Any],
    ) -> None:
        """
        Save readiness evaluation to database.
        
        Args:
            readiness_data: Output from calculate_readiness()
        """
        async with async_session_maker() as db:
            # Check if record exists
            check_query = text("""
                SELECT id FROM explorer_readiness
                WHERE agent_id = :agent_id
                  AND (target_world_id = :target_world_id 
                       OR (target_world_id IS NULL AND :target_world_id IS NULL))
            """)
            
            result = await db.execute(check_query, {
                "agent_id": readiness_data["agent_id"],
                "target_world_id": readiness_data["target_world_id"],
            })
            
            existing = result.fetchone()
            
            if existing:
                # Update existing record
                update_query = text("""
                    UPDATE explorer_readiness
                    SET readiness_score = :readiness_score,
                        knowledge_coverage = :knowledge_coverage,
                        exploration_depth = :exploration_depth,
                        spatial_competence = :spatial_competence,
                        curiosity_level = :curiosity_level,
                        collaboration_score = :collaboration_score,
                        goal_achievement = :goal_achievement,
                        total_steps = :total_steps,
                        knowledge_items_acquired = :knowledge_items_acquired,
                        unique_topics_explored = :unique_topics_explored,
                        is_ready = :is_ready,
                        readiness_threshold = :readiness_threshold,
                        last_evaluated_at = NOW(),
                        updated_at = NOW()
                    WHERE id = :id
                """)
                
                await db.execute(update_query, {
                    "id": existing.id,
                    "readiness_score": readiness_data["readiness_score"],
                    "knowledge_coverage": readiness_data["component_scores"]["knowledge_coverage"],
                    "exploration_depth": readiness_data["component_scores"]["exploration_depth"],
                    "spatial_competence": readiness_data["component_scores"]["spatial_competence"],
                    "curiosity_level": readiness_data["component_scores"]["curiosity_level"],
                    "collaboration_score": readiness_data["component_scores"]["collaboration_score"],
                    "goal_achievement": readiness_data["component_scores"]["goal_achievement"],
                    "total_steps": 0,  # TODO: Get from agent state
                    "knowledge_items_acquired": readiness_data["training_metrics"]["total_knowledge_items"],
                    "unique_topics_explored": readiness_data["training_metrics"]["unique_topics"],
                    "is_ready": readiness_data["is_ready"],
                    "readiness_threshold": readiness_data["readiness_threshold"],
                })
            else:
                # Insert new record
                insert_query = text("""
                    INSERT INTO explorer_readiness (
                        agent_id, target_world_id, readiness_score,
                        knowledge_coverage, exploration_depth, spatial_competence,
                        curiosity_level, collaboration_score, goal_achievement,
                        total_steps, knowledge_items_acquired, unique_topics_explored,
                        is_ready, readiness_threshold, last_evaluated_at
                    ) VALUES (
                        :agent_id, :target_world_id, :readiness_score,
                        :knowledge_coverage, :exploration_depth, :spatial_competence,
                        :curiosity_level, :collaboration_score, :goal_achievement,
                        :total_steps, :knowledge_items_acquired, :unique_topics_explored,
                        :is_ready, :readiness_threshold, NOW()
                    )
                """)
                
                await db.execute(insert_query, {
                    "agent_id": readiness_data["agent_id"],
                    "target_world_id": readiness_data["target_world_id"],
                    "readiness_score": readiness_data["readiness_score"],
                    "knowledge_coverage": readiness_data["component_scores"]["knowledge_coverage"],
                    "exploration_depth": readiness_data["component_scores"]["exploration_depth"],
                    "spatial_competence": readiness_data["component_scores"]["spatial_competence"],
                    "curiosity_level": readiness_data["component_scores"]["curiosity_level"],
                    "collaboration_score": readiness_data["component_scores"]["collaboration_score"],
                    "goal_achievement": readiness_data["component_scores"]["goal_achievement"],
                    "total_steps": 0,
                    "knowledge_items_acquired": readiness_data["training_metrics"]["total_knowledge_items"],
                    "unique_topics_explored": readiness_data["training_metrics"]["unique_topics"],
                    "is_ready": readiness_data["is_ready"],
                    "readiness_threshold": readiness_data["readiness_threshold"],
                })
            
            await db.commit()
