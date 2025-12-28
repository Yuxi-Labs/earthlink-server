"""Unified command interface - consolidates all agent control methods."""

from enum import Enum
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field


router = APIRouter()


# -------------------------------------------------------------------------
# Command Types
# -------------------------------------------------------------------------

class CommandType(str, Enum):
    """Types of commands agents can execute."""
    
    # Movement
    MOVE_TO = "move_to"
    EXPLORE_RANDOM = "explore_random"
    NAVIGATE_TO_POI = "navigate_to_poi"
    
    # Knowledge
    EXPLORE_TOPIC = "explore_topic"
    QUERY_KNOWLEDGE = "query_knowledge"
    LEARN_FROM_OBSERVATION = "learn_from_observation"
    
    # Communication
    SEND_MESSAGE = "send_message"
    BROADCAST_MESSAGE = "broadcast_message"
    
    # World Interaction
    QUERY_NEARBY = "query_nearby"
    OBSERVE_ENVIRONMENT = "observe_environment"
    
    # Control
    SET_GOAL = "set_goal"
    AUTONOMOUS_STEP = "autonomous_step"
    RESET = "reset"
    
    # Lifecycle
    PAUSE = "pause"
    RESUME = "resume"
    SHUTDOWN = "shutdown"


# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------

class Command(BaseModel):
    """Unified command structure."""
    
    type: CommandType
    parameters: dict[str, Any] = Field(default_factory=dict)
    agent_id: UUID | None = None
    world_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CommandResponse(BaseModel):
    """Response from command execution."""
    
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchCommand(BaseModel):
    """Execute multiple commands."""
    
    commands: list[Command]
    sequential: bool = True  # If False, execute in parallel


# -------------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------------

def get_simulation():
    """Get simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


# -------------------------------------------------------------------------
# Command Execution
# -------------------------------------------------------------------------

async def execute_command(
    command: Command,
    simulation,
) -> CommandResponse:
    """Execute a single command."""
    
    if simulation is None:
        return CommandResponse(
            success=False,
            error="Simulation not initialized",
        )
    
    agent_id = command.agent_id
    if agent_id is None:
        return CommandResponse(
            success=False,
            error="agent_id required",
        )
    
    agent_ref = simulation._agents.get(agent_id)
    if agent_ref is None:
        return CommandResponse(
            success=False,
            error=f"Agent {agent_id} not found",
        )
    
    try:
        # Route command to appropriate handler
        if command.type == CommandType.MOVE_TO:
            lat = command.parameters.get("lat")
            lon = command.parameters.get("lon")
            altitude = command.parameters.get("altitude", 10.0)
            
            if lat is None or lon is None:
                raise ValueError("lat and lon required for move_to")
            
            result = await agent_ref.move_to.remote(lat, lon, altitude)
            
        elif command.type == CommandType.EXPLORE_RANDOM:
            max_distance_km = command.parameters.get("max_distance_km", 10.0)
            result = await agent_ref.explore_random_location.remote(max_distance_km)
            
        elif command.type == CommandType.NAVIGATE_TO_POI:
            poi_type = command.parameters.get("poi_type")
            if poi_type is None:
                raise ValueError("poi_type required for navigate_to_poi")
            
            result = await agent_ref.navigate_to_poi.remote(poi_type)
            
        elif command.type == CommandType.EXPLORE_TOPIC:
            topic = command.parameters.get("topic")
            if topic is None:
                raise ValueError("topic required for explore_topic")
            
            result = await agent_ref.explore_topic.remote(topic)
            
        elif command.type == CommandType.QUERY_KNOWLEDGE:
            query = command.parameters.get("query")
            source = command.parameters.get("source", "wikipedia")
            
            if query is None:
                raise ValueError("query required for query_knowledge")
            
            result = await agent_ref.query_knowledge.remote(query, source)
            
        elif command.type == CommandType.SEND_MESSAGE:
            target_id = command.parameters.get("target_id")
            content = command.parameters.get("content")
            
            if target_id is None or content is None:
                raise ValueError("target_id and content required for send_message")
            
            target_uuid = UUID(target_id) if isinstance(target_id, str) else target_id
            result = await agent_ref.send_message.remote(target_uuid, content)
            
        elif command.type == CommandType.BROADCAST_MESSAGE:
            content = command.parameters.get("content")
            if content is None:
                raise ValueError("content required for broadcast_message")
            
            result = await agent_ref.broadcast_message.remote(content)
            
        elif command.type == CommandType.QUERY_NEARBY:
            feature_type = command.parameters.get("feature_type")
            radius_km = command.parameters.get("radius_km", 1.0)
            limit = command.parameters.get("limit", 10)
            
            result = await agent_ref.query_nearby_features.remote(
                feature_type, radius_km, limit
            )
            
        elif command.type == CommandType.OBSERVE_ENVIRONMENT:
            result = await agent_ref.observe_environment.remote()
            
        elif command.type == CommandType.SET_GOAL:
            goal_type = command.parameters.get("goal_type")
            goal_params = command.parameters.get("goal_params", {})
            
            if goal_type is None:
                raise ValueError("goal_type required for set_goal")
            
            result = await agent_ref.set_goal.remote(goal_type, goal_params)
            
        elif command.type == CommandType.AUTONOMOUS_STEP:
            result = await agent_ref.autonomous_step.remote()
            
        elif command.type == CommandType.RESET:
            result = await agent_ref.reset.remote()
            
        elif command.type == CommandType.PAUSE:
            # Pause agent (implementation needed in agent)
            result = {"message": "Pause not yet implemented"}
            
        elif command.type == CommandType.RESUME:
            # Resume agent (implementation needed in agent)
            result = {"message": "Resume not yet implemented"}
            
        elif command.type == CommandType.SHUTDOWN:
            # Graceful shutdown
            await simulation.destroy_agent(agent_id)
            result = {"message": "Agent shutdown"}
            
        else:
            raise ValueError(f"Unknown command type: {command.type}")
        
        return CommandResponse(
            success=True,
            result=result if isinstance(result, dict) else {"data": result},
            metadata=command.metadata,
        )
        
    except Exception as e:
        return CommandResponse(
            success=False,
            error=str(e),
            metadata=command.metadata,
        )


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@router.post("/execute", response_model=CommandResponse)
async def execute_single_command(
    command: Command,
    simulation=Depends(get_simulation),
) -> CommandResponse:
    """Execute a single command."""
    return await execute_command(command, simulation)


@router.post("/batch", response_model=list[CommandResponse])
async def execute_batch_commands(
    batch: BatchCommand,
    simulation=Depends(get_simulation),
) -> list[CommandResponse]:
    """Execute multiple commands."""
    
    if batch.sequential:
        # Execute sequentially
        responses = []
        for cmd in batch.commands:
            response = await execute_command(cmd, simulation)
            responses.append(response)
        return responses
    else:
        # Execute in parallel
        import asyncio
        tasks = [execute_command(cmd, simulation) for cmd in batch.commands]
        responses = await asyncio.gather(*tasks)
        return list(responses)


@router.get("/types")
async def get_command_types() -> dict[str, list[str]]:
    """Get all available command types."""
    return {
        "command_types": [ct.value for ct in CommandType],
        "categories": {
            "movement": ["move_to", "explore_random", "navigate_to_poi"],
            "knowledge": ["explore_topic", "query_knowledge", "learn_from_observation"],
            "communication": ["send_message", "broadcast_message"],
            "world": ["query_nearby", "observe_environment"],
            "control": ["set_goal", "autonomous_step", "reset"],
            "lifecycle": ["pause", "resume", "shutdown"],
        },
    }


@router.post("/agent/{agent_id}/quick/{command_type}")
async def quick_command(
    agent_id: UUID,
    command_type: CommandType,
    parameters: dict[str, Any] = {},
    simulation=Depends(get_simulation),
) -> CommandResponse:
    """Quick command execution (simplified endpoint)."""
    
    command = Command(
        type=command_type,
        parameters=parameters,
        agent_id=agent_id,
    )
    
    return await execute_command(command, simulation)
