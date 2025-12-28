"""Health check and service monitoring endpoints."""

from typing import Any
from datetime import datetime

from fastapi import APIRouter, Depends
import ray
import psutil


router = APIRouter()


def get_simulation():
    """Get simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe - is the service running?"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready")
async def readiness(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Kubernetes readiness probe - is the service ready to accept traffic?"""
    
    checks = {
        "simulation": simulation is not None,
        "ray": ray.is_initialized() if simulation else False,
    }
    
    # Check database
    try:
        from src.db.database import async_session_maker
        async with async_session_maker() as db:
            await db.execute("SELECT 1")
        checks["database"] = True
    except Exception:
        checks["database"] = False
    
    all_ready = all(checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/status")
async def health_status(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Comprehensive health status."""
    
    status = {
        "service": "earthlink-api",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": 0,  # TODO: Track from startup
    }
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    status["system"] = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024**3),
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free / (1024**3),
    }
    
    # Simulation status
    if simulation:
        status["simulation"] = {
            "state": simulation.state.name,
            "total_steps": simulation._total_steps,
            "active_agents": len(simulation._agents),
            "active_worlds": len(simulation.world_registry._worlds),
        }
        
        # Ray cluster info
        if ray.is_initialized():
            try:
                cluster_resources = ray.cluster_resources()
                status["ray"] = {
                    "initialized": True,
                    "cpu_available": cluster_resources.get("CPU", 0),
                    "memory_gb": cluster_resources.get("memory", 0) / (1024**3),
                    "gpu_available": cluster_resources.get("GPU", 0),
                }
            except Exception:
                status["ray"] = {"initialized": True, "error": "Failed to get resources"}
        else:
            status["ray"] = {"initialized": False}
    else:
        status["simulation"] = {"state": "not_initialized"}
        status["ray"] = {"initialized": False}
    
    # Database status
    try:
        from src.db.database import async_session_maker
        from sqlalchemy import text
        
        async with async_session_maker() as db:
            # Check basic connectivity
            await db.execute(text("SELECT 1"))
            
            # Get table counts
            agent_count = await db.execute(text("SELECT COUNT(*) FROM agents"))
            world_count = await db.execute(text("SELECT COUNT(*) FROM worlds"))
            knowledge_count = await db.execute(text("SELECT COUNT(*) FROM knowledge_acquisition_log"))
            
            status["database"] = {
                "connected": True,
                "agents": agent_count.scalar(),
                "worlds": world_count.scalar(),
                "knowledge_items": knowledge_count.scalar(),
            }
    except Exception as e:
        status["database"] = {
            "connected": False,
            "error": str(e),
        }
    
    # Services status
    status["services"] = {}
    
    # ChromaDB
    try:
        import requests
        from src.config import settings
        chroma_url = f"http://{settings.chroma_host}:{settings.chroma_port}/api/v1/heartbeat"
        response = requests.get(chroma_url, timeout=2)
        status["services"]["chromadb"] = {"status": "up" if response.ok else "down"}
    except Exception:
        status["services"]["chromadb"] = {"status": "down"}
    
    # Redis
    try:
        from src.config import settings
        import redis
        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        status["services"]["redis"] = {"status": "up"}
    except Exception:
        status["services"]["redis"] = {"status": "down"}
    
    # Ollama
    try:
        import requests
        from src.config import settings
        ollama_url = f"{settings.ollama_base_url}/api/tags"
        response = requests.get(ollama_url, timeout=2)
        status["services"]["ollama"] = {"status": "up" if response.ok else "down"}
    except Exception:
        status["services"]["ollama"] = {"status": "down"}
    
    return status


@router.get("/health/metrics")
async def health_metrics(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Prometheus-style metrics."""
    
    metrics = {}
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    metrics["system_cpu_percent"] = cpu_percent
    metrics["system_memory_percent"] = memory.percent
    metrics["system_memory_used_bytes"] = memory.used
    metrics["system_memory_available_bytes"] = memory.available
    
    # Simulation metrics
    if simulation:
        metrics["simulation_total_steps"] = simulation._total_steps
        metrics["simulation_active_agents"] = len(simulation._agents)
        metrics["simulation_active_worlds"] = len(simulation.world_registry._worlds)
        metrics["simulation_state"] = 1 if simulation.state.name == "RUNNING" else 0
        
        # Episode metrics
        if hasattr(simulation, 'episode_manager'):
            stats = simulation.episode_manager.get_stats()
            metrics["episodes_total"] = stats["total_episodes"]
            metrics["episodes_successful"] = stats["successful_episodes"]
            metrics["episodes_failed"] = stats["failed_episodes"]
            metrics["episodes_current"] = stats["current_episodes"]
    
    # Database metrics
    try:
        from src.db.database import async_session_maker
        from sqlalchemy import text
        
        async with async_session_maker() as db:
            agent_count = await db.execute(text("SELECT COUNT(*) FROM agents"))
            knowledge_count = await db.execute(text("SELECT COUNT(*) FROM knowledge_acquisition_log"))
            
            metrics["db_agents_total"] = agent_count.scalar()
            metrics["db_knowledge_items_total"] = knowledge_count.scalar()
            metrics["db_connected"] = 1
    except Exception:
        metrics["db_connected"] = 0
    
    # Ray metrics
    if simulation and ray.is_initialized():
        try:
            cluster_resources = ray.cluster_resources()
            metrics["ray_cpu_available"] = cluster_resources.get("CPU", 0)
            metrics["ray_memory_bytes"] = cluster_resources.get("memory", 0)
            metrics["ray_initialized"] = 1
        except Exception:
            metrics["ray_initialized"] = 0
    else:
        metrics["ray_initialized"] = 0
    
    return metrics
