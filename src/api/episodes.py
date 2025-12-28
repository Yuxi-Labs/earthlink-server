"""Episode management API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends


router = APIRouter()


def get_simulation():
    """Get simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


@router.get("/stats")
async def get_episode_stats(
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get episode statistics."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    return simulation.episode_manager.get_stats()


@router.get("/recent")
async def get_recent_episodes(
    limit: int = 10,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get recent episodes."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    episodes = simulation.episode_manager.get_recent_episodes(limit)
    
    return {
        "episodes": [ep.to_dict() for ep in episodes],
        "count": len(episodes),
    }


@router.get("/best")
async def get_best_episodes(
    limit: int = 10,
    by: str = "reward",
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get best performing episodes."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    episodes = simulation.episode_manager.get_best_episodes(limit, by=by)
    
    return {
        "episodes": [ep.to_dict() for ep in episodes],
        "count": len(episodes),
        "sorted_by": by,
    }


@router.get("/agent/{agent_id}")
async def get_agent_episodes(
    agent_id: UUID,
    limit: int = 10,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get episodes for a specific agent."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    episodes = simulation.episode_manager.get_agent_episodes(agent_id, limit)
    
    return {
        "agent_id": str(agent_id),
        "episodes": [ep.to_dict() for ep in episodes],
        "count": len(episodes),
    }


@router.get("/world/{world_id}")
async def get_world_episodes(
    world_id: str,
    limit: int = 10,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get episodes for a specific world."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    episodes = simulation.episode_manager.get_world_episodes(world_id, limit)
    
    return {
        "world_id": world_id,
        "episodes": [ep.to_dict() for ep in episodes],
        "count": len(episodes),
    }


@router.delete("/agent/{agent_id}")
async def reset_agent_episodes(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, str]:
    """Reset/clear episodes for an agent."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    simulation.episode_manager.reset_agent(agent_id)
    
    return {"message": f"Episodes cleared for agent {agent_id}"}


@router.delete("/all")
async def clear_all_episodes(
    simulation=Depends(get_simulation),
) -> dict[str, str]:
    """Clear all episodes."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    simulation.episode_manager.clear_all()
    
    return {"message": "All episodes cleared"}
