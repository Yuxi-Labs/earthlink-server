"""Simulation control API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter()


# -------------------------------------------------------------------------
# Request/Response Schemas
# -------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """Schema for simulation configuration."""

    steps_per_second: float = 10.0
    max_steps: int | None = None
    speed_multiplier: float = 1.0
    train_every_n_steps: int = 4
    batch_size: int = 32
    checkpoint_every_n_steps: int = 1000


# -------------------------------------------------------------------------
# Dependency: Get simulation runner
# -------------------------------------------------------------------------

def get_simulation():
    """Get the global simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.get("/status")
async def get_status(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Get simulation status and statistics."""
    if simulation is None:
        return {
            "status": "not_initialized",
            "message": "Simulation not running",
        }

    return {
        "status": simulation.state.name,
        "stats": simulation.get_stats(),
    }


@router.post("/start")
async def start_simulation(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Start the simulation loop."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")

    import asyncio
    asyncio.create_task(simulation.run())

    return {
        "message": "Simulation started",
        "status": "RUNNING",
    }


@router.post("/pause")
async def pause_simulation(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Pause the simulation."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")

    simulation.pause()
    return {
        "message": "Simulation paused",
        "status": "PAUSED",
    }


@router.post("/resume")
async def resume_simulation(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Resume a paused simulation."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")

    simulation.resume()
    return {
        "message": "Simulation resumed",
        "status": "RUNNING",
    }


@router.post("/stop")
async def stop_simulation(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Stop the simulation."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")

    simulation.stop()
    return {
        "message": "Simulation stopped",
        "status": "STOPPED",
    }


@router.post("/speed/set")
async def set_simulation_speed(
    multiplier: float,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Set simulation speed multiplier.
    
    Args:
        multiplier: Speed multiplier (0.1 = slow motion, 1.0 = normal, 10.0 = fast forward)
    """
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    if multiplier <= 0:
        raise HTTPException(status_code=400, detail="Speed multiplier must be positive")
    
    simulation.config.speed_multiplier = multiplier
    
    return {
        "message": f"Simulation speed set to {multiplier}x",
        "speed_multiplier": multiplier,
        "effective_sps": simulation.config.steps_per_second * multiplier,
    }


@router.get("/speed")
async def get_simulation_speed(
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get current simulation speed."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    return {
        "speed_multiplier": simulation.config.speed_multiplier,
        "base_sps": simulation.config.steps_per_second,
        "effective_sps": simulation.config.steps_per_second * simulation.config.speed_multiplier,
    }


@router.post("/speed/slow")
async def slow_motion(
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Set simulation to slow motion (0.25x speed)."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    simulation.config.speed_multiplier = 0.25
    return {"message": "Slow motion activated", "speed_multiplier": 0.25}


@router.post("/speed/normal")
async def normal_speed(
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Set simulation to normal speed (1.0x)."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    simulation.config.speed_multiplier = 1.0
    return {"message": "Normal speed restored", "speed_multiplier": 1.0}


@router.post("/speed/fast")
async def fast_forward(
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Set simulation to fast forward (5.0x speed)."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    simulation.config.speed_multiplier = 5.0
    return {"message": "Fast forward activated", "speed_multiplier": 5.0}


@router.get("/events")
async def get_events(
    event_type: str | None = None,
    limit: int = 100,
    simulation=Depends(get_simulation),
) -> list[dict[str, Any]]:
    """Get recent simulation events."""
    if simulation is None:
        return []

    from src.simulation.events import EventType

    if event_type:
        try:
            et = EventType[event_type.upper()]
            events = simulation.event_bus.get_history(event_type=et, limit=limit)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown event type: {event_type}")
    else:
        events = simulation.event_bus.get_history(limit=limit)

    return [e.to_dict() for e in events]


@router.post("/checkpoint")
async def save_all_checkpoints(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Save checkpoints for all agents."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")

    await simulation._save_checkpoints()
    return {
        "message": "Checkpoints saved",
        "step": simulation._total_steps,
    }


@router.get("/cluster")
async def get_cluster_stats(simulation=Depends(get_simulation)) -> dict[str, Any]:
    """Get Ray cluster statistics and node information."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    return simulation.get_cluster_stats()


class PlacementGroupRequest(BaseModel):
    """Request to create a placement group."""
    name: str
    num_agents: int
    strategy: str | None = None  # SPREAD, PACK, or STRICT_SPREAD


@router.post("/cluster/placement-groups")
async def create_placement_group(
    request: PlacementGroupRequest,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Create a placement group for agent distribution."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    success = await simulation.create_agent_placement_group(
        request.name,
        request.num_agents,
        request.strategy,
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to create placement group. Cluster not enabled or error occurred."
        )
    
    return {
        "message": f"Placement group '{request.name}' created",
        "name": request.name,
        "num_agents": request.num_agents,
        "strategy": request.strategy,
    }
