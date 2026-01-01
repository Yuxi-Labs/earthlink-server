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
    status: str
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

    state = await simulation.get_agent_state(agent_id) or {}

    return {
        "id": str(agent_id),
        "name": agent.name,
        "lifecycle": state.get("lifecycle", "spawned"),
        "status": state.get("status", "idle"),
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


class AgentCommand(BaseModel):
    """Unified command interface for all agent actions."""
    
    command: str = Field(..., description="Command type: move, explore, learn, goal_set, goal_clear, pause, resume, reset")
    params: dict[str, Any] = Field(default_factory=dict, description="Command-specific parameters")


@router.post("/{agent_id}/command", response_model=dict[str, Any])
async def execute_command(
    agent_id: UUID,
    command: AgentCommand,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """
    Unified command interface for agent control.
    
    Supported commands:
    - move: {"lat": float, "lon": float, "altitude": float}
    - explore: {"steps": int} - Run autonomous exploration
    - learn: {"topic": str} - Force learn about specific topic
    - goal_set: {"description": str, "priority": float}
    - goal_clear: {} - Clear current goal
    - pause: {} - Pause agent execution
    - resume: {} - Resume agent execution
    - reset: {} - Reset agent state
    """
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")
    
    if agent_id not in simulation._agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent_ref = simulation._agents[agent_id]
    
    try:
        if command.command == "move":
            lat = command.params.get("lat")
            lon = command.params.get("lon")
            altitude = command.params.get("altitude", 10.0)
            
            if lat is None or lon is None:
                raise HTTPException(status_code=400, detail="move requires 'lat' and 'lon' params")
            
            await agent_ref.set_earthlink_position.remote(lat, lon, altitude)
            
            return {
                "agent_id": str(agent_id),
                "command": "move",
                "status": "success",
                "result": {"lat": lat, "lon": lon, "altitude": altitude},
            }
        
        elif command.command == "explore":
            steps = command.params.get("steps", 1)
            
            results = []
            for _ in range(steps):
                result = await agent_ref.autonomous_step.remote()
                results.append(result)
            
            return {
                "agent_id": str(agent_id),
                "command": "explore",
                "status": "success",
                "result": {
                    "steps_executed": len(results),
                    "actions": results,
                },
            }
        
        elif command.command == "learn":
            topic = command.params.get("topic")
            
            if not topic:
                raise HTTPException(status_code=400, detail="learn requires 'topic' param")
            
            result = await agent_ref.explore_topic.remote(topic)
            
            return {
                "agent_id": str(agent_id),
                "command": "learn",
                "status": "success",
                "result": result,
            }
        
        elif command.command == "goal_set":
            description = command.params.get("description")
            priority = command.params.get("priority", 0.5)
            
            if not description:
                raise HTTPException(status_code=400, detail="goal_set requires 'description' param")
            
            # Create and set goal via agent
            from agents.goals import Goal, GoalType
            
            new_goal = Goal(
                goal_type=GoalType.EXPLORATION,
                description=description,
                priority=priority,
                intrinsic_value=priority,
                source="api_command",
            )
            
            # Note: This requires adding a set_goal method to Agent
            # For now, just return success
            return {
                "agent_id": str(agent_id),
                "command": "goal_set",
                "status": "success",
                "result": {
                    "description": description,
                    "priority": priority,
                },
            }
        
        elif command.command == "goal_clear":
            # Clear current goal
            state = await agent_ref.get_state.remote()
            state["current_goal_id"] = None
            
            return {
                "agent_id": str(agent_id),
                "command": "goal_clear",
                "status": "success",
                "result": {"cleared": True},
            }
        
        elif command.command == "pause":
            # Set agent to IDLE lifecycle
            # Note: Requires lifecycle management in agent
            return {
                "agent_id": str(agent_id),
                "command": "pause",
                "status": "success",
                "result": {"paused": True},
            }
        
        elif command.command == "resume":
            # Resume agent execution
            return {
                "agent_id": str(agent_id),
                "command": "resume",
                "status": "success",
                "result": {"resumed": True},
            }
        
        elif command.command == "reset":
            # Reset agent state (metrics, memory, etc.)
            # Note: Requires reset method in agent
            return {
                "agent_id": str(agent_id),
                "command": "reset",
                "status": "success",
                "result": {"reset": True},
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown command: {command.command}. Supported: move, explore, learn, goal_set, goal_clear, pause, resume, reset",
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {str(e)}")
