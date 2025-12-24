"""Ray cluster configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeConfig:
    """Configuration for a Ray cluster node."""

    # Node identification
    node_type: str = "worker"  # head or worker
    node_ip: str = "127.0.0.1"
    
    # Resources
    num_cpus: int = 0  # 0 means auto-detect
    num_gpus: int = 0
    memory: int = 0  # bytes, 0 means auto-detect
    
    # Object store
    object_store_memory: int = 0  # bytes, 0 means auto-detect
    
    # Custom resources for agent placement
    resources: dict[str, float] = field(default_factory=dict)
    
    # Labels for filtering
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class ClusterConfig:
    """Configuration for Ray cluster."""

    # Cluster identification
    cluster_name: str = "earthlink-cluster"
    namespace: str = "earthlink"
    
    # Head node configuration
    head_node: NodeConfig = field(default_factory=lambda: NodeConfig(node_type="head"))
    
    # Worker nodes
    worker_nodes: list[NodeConfig] = field(default_factory=list)
    
    # Ray runtime configuration
    runtime_env: dict[str, Any] = field(default_factory=dict)
    
    # Auto-scaling
    autoscaling_enabled: bool = True
    min_workers: int = 1
    max_workers: int = 10
    idle_timeout_minutes: int = 5
    
    # Agent placement strategy
    agent_placement_strategy: str = "SPREAD"  # SPREAD, PACK, or STRICT_SPREAD
    agents_per_node: int = 100  # Target agents per node
    
    # Monitoring
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8265
    metrics_export_port: int = 8080
    
    # Redis address for existing cluster
    redis_address: str | None = None
    redis_password: str | None = None
    
    def to_ray_init_kwargs(self) -> dict[str, Any]:
        """Convert to Ray init kwargs."""
        kwargs: dict[str, Any] = {
            "namespace": self.namespace,
            "runtime_env": self.runtime_env,
            "dashboard_host": self.dashboard_host,
            "dashboard_port": self.dashboard_port,
        }
        
        if self.redis_address:
            kwargs["address"] = self.redis_address
            if self.redis_password:
                kwargs["_redis_password"] = self.redis_password
        
        # Head node resources
        if self.head_node.num_cpus > 0:
            kwargs["num_cpus"] = self.head_node.num_cpus
        if self.head_node.num_gpus > 0:
            kwargs["num_gpus"] = self.head_node.num_gpus
        if self.head_node.memory > 0:
            kwargs["memory"] = self.head_node.memory
        if self.head_node.object_store_memory > 0:
            kwargs["object_store_memory"] = self.head_node.object_store_memory
        if self.head_node.resources:
            kwargs["resources"] = self.head_node.resources
        
        return kwargs
