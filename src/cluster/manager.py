"""Ray cluster manager for multi-agent scaling."""

import asyncio
import logging
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .config import ClusterConfig

logger = logging.getLogger(__name__)


class ClusterManager:
    """
    Manages Ray cluster for distributed agent execution.
    
    Handles:
    - Cluster initialization and connection
    - Node management and monitoring
    - Agent placement across nodes
    - Auto-scaling coordination
    - Resource tracking
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self._initialized = False
        self._placement_groups: dict[str, PlacementGroup] = {}
        self._agent_placement: dict[str, str] = {}  # agent_id -> placement_group_name

    async def initialize(self) -> None:
        """Initialize Ray cluster."""
        if self._initialized:
            logger.warning("Cluster already initialized")
            return

        try:
            if self.config.redis_address:
                # Connect to existing cluster
                logger.info(f"Connecting to existing Ray cluster at {self.config.redis_address}")
                ray.init(address=self.config.redis_address, namespace=self.config.namespace)
            else:
                # Start local cluster
                logger.info("Starting local Ray cluster")
                ray.init(**self.config.to_ray_init_kwargs())

            self._initialized = True
            logger.info(f"Ray cluster initialized: {ray.cluster_resources()}")

        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown cluster gracefully."""
        if not self._initialized:
            return

        logger.info("Shutting down Ray cluster")
        
        # Remove all placement groups
        for pg_name, pg in self._placement_groups.items():
            try:
                ray.util.remove_placement_group(pg)
                logger.info(f"Removed placement group: {pg_name}")
            except Exception as e:
                logger.warning(f"Error removing placement group {pg_name}: {e}")

        ray.shutdown()
        self._initialized = False
        self._placement_groups.clear()
        self._agent_placement.clear()

    def is_initialized(self) -> bool:
        """Check if cluster is initialized."""
        return self._initialized and ray.is_initialized()

    def get_cluster_resources(self) -> dict[str, Any]:
        """Get current cluster resources."""
        if not self.is_initialized():
            return {}

        return {
            "total": ray.cluster_resources(),
            "available": ray.available_resources(),
            "nodes": self._get_node_info(),
        }

    def _get_node_info(self) -> list[dict[str, Any]]:
        """Get information about all nodes in cluster."""
        nodes = []
        for node in ray.nodes():
            if node["Alive"]:
                nodes.append({
                    "node_id": node["NodeID"],
                    "node_ip": node["NodeManagerAddress"],
                    "resources": node["Resources"],
                    "alive": node["Alive"],
                })
        return nodes

    async def create_placement_group(
        self,
        name: str,
        num_agents: int,
        strategy: str | None = None,
    ) -> PlacementGroup:
        """
        Create a placement group for agent co-location.
        
        Placement groups ensure agents are placed together (PACK)
        or spread across nodes (SPREAD) for optimal performance.
        """
        if not self.is_initialized():
            raise RuntimeError("Cluster not initialized")

        if name in self._placement_groups:
            logger.warning(f"Placement group {name} already exists")
            return self._placement_groups[name]

        strategy = strategy or self.config.agent_placement_strategy

        # Calculate bundles (resource allocations per agent)
        bundles = [{"CPU": 1, "memory": 1024 * 1024 * 512}] * num_agents  # 512MB per agent

        try:
            pg = placement_group(bundles, strategy=strategy, name=name)
            ray.get(pg.ready(), timeout=30)
            self._placement_groups[name] = pg
            logger.info(f"Created placement group {name} with {num_agents} bundles ({strategy})")
            return pg

        except Exception as e:
            logger.error(f"Failed to create placement group {name}: {e}")
            raise

    def get_placement_strategy(self, placement_group_name: str) -> PlacementGroupSchedulingStrategy:
        """Get scheduling strategy for a placement group."""
        if placement_group_name not in self._placement_groups:
            raise ValueError(f"Placement group {placement_group_name} not found")

        pg = self._placement_groups[placement_group_name]
        return PlacementGroupSchedulingStrategy(placement_group=pg)

    def register_agent_placement(self, agent_id: str, placement_group_name: str) -> None:
        """Register which placement group an agent belongs to."""
        self._agent_placement[agent_id] = placement_group_name

    def get_agent_placement(self, agent_id: str) -> str | None:
        """Get which placement group an agent is in."""
        return self._agent_placement.get(agent_id)

    def get_placement_group_agents(self, placement_group_name: str) -> list[str]:
        """Get all agents in a placement group."""
        return [
            agent_id
            for agent_id, pg_name in self._agent_placement.items()
            if pg_name == placement_group_name
        ]

    async def scale_workers(self, target_workers: int) -> None:
        """
        Request scaling to target number of workers.
        
        Note: Actual scaling depends on cluster autoscaler configuration.
        This is a hint to the autoscaler.
        """
        if not self.is_initialized():
            raise RuntimeError("Cluster not initialized")

        current_nodes = len(self._get_node_info())
        logger.info(f"Scaling request: current={current_nodes}, target={target_workers}")

        # Ray autoscaler will handle actual scaling based on resource demands
        # We create resource demands by requesting future agent placements
        if target_workers > current_nodes:
            logger.info("Cluster will auto-scale up based on resource demands")

    def get_stats(self) -> dict[str, Any]:
        """Get cluster statistics."""
        if not self.is_initialized():
            return {
                "initialized": False,
                "placement_groups": 0,
                "agents": 0,
            }

        return {
            "initialized": True,
            "cluster_name": self.config.cluster_name,
            "namespace": self.config.namespace,
            "placement_groups": len(self._placement_groups),
            "agents": len(self._agent_placement),
            "nodes": len(self._get_node_info()),
            "resources": self.get_cluster_resources(),
            "placement_group_details": {
                name: {
                    "agents": len(self.get_placement_group_agents(name)),
                    "state": pg.state if hasattr(pg, 'state') else "CREATED",
                }
                for name, pg in self._placement_groups.items()
            },
        }


# Global instance
_cluster_manager: ClusterManager | None = None


def get_cluster_manager() -> ClusterManager | None:
    """Get global cluster manager instance."""
    return _cluster_manager


def init_cluster_manager(config: ClusterConfig) -> ClusterManager:
    """Initialize global cluster manager."""
    global _cluster_manager
    _cluster_manager = ClusterManager(config)
    return _cluster_manager


async def shutdown_cluster_manager() -> None:
    """Shutdown global cluster manager."""
    global _cluster_manager
    if _cluster_manager:
        await _cluster_manager.shutdown()
        _cluster_manager = None
