"""Agent management API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

router = APIRouter()


# -------------------------------------------------------------------------
# Request/Response Schemas
# -------------------------------------------------------------------------

class AgentCreate(BaseModel):
    """Schema for creating a new agent."""

    name: str = Field(..., min_length=1, max_length=100)
    config: dict[str, Any] = Field(default_factory=dict)
    placement_group: str | None = Field(None, description="Placement group for cluster distribution")


class AgentResponse(BaseModel):
    """Schema for agent response."""

    id: str
    name: str
    lifecycle: str
    target_world: str | None
    metrics: dict[str, Any]


class AgentAssignWorld(BaseModel):
    """Schema for assigning agent to world."""

    world_id: str


class AgentCheckpoint(BaseModel):
    """Schema for checkpoint operations."""

    path: str


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
async def list_agents(simulation=Depends(get_simulation)) -> list[dict[str, Any]]:
    """List all agents."""
    if simulation is None:
        return []
    return await simulation.list_agents()


@router.post("/", response_model=dict[str, Any])
async def create_agent(
    agent: AgentCreate,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Create and spawn a new agent."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    agent_id = await simulation.spawn_agent(
        name=agent.name,
        config=agent.config,
        placement_group=agent.placement_group,
    )

    return {
        "id": str(agent_id),
        "name": agent.name,
        "lifecycle": "TRAINING",
        "placement_group": agent.placement_group,
        "message": "Agent spawned successfully",
    }


@router.get("/{agent_id}", response_model=dict[str, Any])
async def get_agent(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent by ID."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    state = await simulation.get_agent_state(agent_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    return state


@router.delete("/{agent_id}")
async def destroy_agent(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Destroy an agent."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    success = await simulation.destroy_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"message": "Agent destroyed", "id": str(agent_id)}


@router.post("/{agent_id}/assign-world")
async def assign_world(
    agent_id: UUID,
    assignment: AgentAssignWorld,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Assign agent to explore a world."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    success = await simulation.assign_agent_to_world(agent_id, assignment.world_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign agent to world")

    return {
        "agent_id": str(agent_id),
        "world_id": assignment.world_id,
        "message": "Agent assigned to world",
    }


@router.get("/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent metrics and learning progress."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    state = await simulation.get_agent_state(agent_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {
        "agent_id": str(agent_id),
        "metrics": state.get("metrics", {}),
    }


@router.post("/{agent_id}/checkpoint/save")
async def save_checkpoint(
    agent_id: UUID,
    checkpoint: AgentCheckpoint,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Save agent checkpoint."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    if agent_id not in simulation._agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_ref = simulation._agents[agent_id]
    await agent_ref.save_checkpoint.remote(checkpoint.path)

    return {
        "agent_id": str(agent_id),
        "path": checkpoint.path,
        "message": "Checkpoint saved",
    }


@router.post("/{agent_id}/checkpoint/load")
async def load_checkpoint(
    agent_id: UUID,
    checkpoint: AgentCheckpoint,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Load agent checkpoint."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")

    if agent_id not in simulation._agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_ref = simulation._agents[agent_id]
    await agent_ref.load_checkpoint.remote(checkpoint.path)

    return {
        "agent_id": str(agent_id),
        "path": checkpoint.path,
        "message": "Checkpoint loaded",
    }

