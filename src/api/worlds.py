"""World management API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

router = APIRouter()


# -------------------------------------------------------------------------
# Request/Response Schemas
# -------------------------------------------------------------------------

class WorldCreate(BaseModel):
    """Schema for creating a world."""

    world_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    world_type: str = Field(default="text")  # text, spatial, etc.
    description: str = ""
    observation_dim: int = Field(default=256, ge=1)
    action_dim: int = Field(default=64, ge=1)


class WorldUpdate(BaseModel):
    """Schema for updating a world."""

    name: str | None = None
    description: str | None = None


class TextWorldData(BaseModel):
    """Schema for adding text data to a world."""

    documents: list[str]


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

@router.get("/", response_model=list[dict[str, Any]])
async def list_worlds(simulation=Depends(get_simulation)) -> list[dict[str, Any]]:
    """List all worlds."""
    if simulation is None:
        return []
    return simulation.list_worlds()


@router.post("/", response_model=dict[str, Any])
async def create_world(
    world: WorldCreate,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Create a new world."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world_id = simulation.create_world(
        world_id=world.world_id,
        name=world.name,
        world_type=world.world_type,
        description=world.description,
        observation_dim=world.observation_dim,
        action_dim=world.action_dim,
    )

    return {
        "id": world_id,
        "name": world.name,
        "world_type": world.world_type,
        "message": "World created successfully",
    }


@router.get("/{world_id}", response_model=dict[str, Any])
async def get_world(
    world_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get world by ID."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world = simulation.get_world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")

    return world.get_metadata()


@router.delete("/{world_id}")
async def delete_world(
    world_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Delete a world."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    success = simulation.world_registry.remove_world(world_id)
    if not success:
        raise HTTPException(status_code=404, detail="World not found")

    return {"message": "World deleted", "id": world_id}


@router.post("/{world_id}/load")
async def load_world(
    world_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Load world resources."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world = simulation.get_world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")

    await world.load()
    return {"message": "World loaded", "id": world_id}


@router.post("/{world_id}/unload")
async def unload_world(
    world_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Unload world resources."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world = simulation.get_world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")

    await world.unload()
    return {"message": "World unloaded", "id": world_id}


@router.post("/{world_id}/reset")
async def reset_world(
    world_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Reset world to initial state."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world = simulation.get_world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")

    observation = await world.reset()
    return {
        "message": "World reset",
        "id": world_id,
        "initial_observation": observation,
    }


@router.post("/{world_id}/documents")
async def add_documents(
    world_id: str,
    data: TextWorldData,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Add documents to a text world."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world = simulation.get_world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")

    if not hasattr(world, "add_documents"):
        raise HTTPException(status_code=400, detail="World does not support documents")

    world.add_documents(data.documents)
    return {
        "message": f"Added {len(data.documents)} documents",
        "world_id": world_id,
    }


@router.get("/{world_id}/observation-space")
async def get_observation_space(
    world_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get observation space specification."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world = simulation.get_world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")

    return world.get_observation_space()


@router.get("/{world_id}/action-space")
async def get_action_space(
    world_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get action space specification."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    world = simulation.get_world(world_id)
    if world is None:
        raise HTTPException(status_code=404, detail="World not found")

    return world.get_action_space()
