"""WebSocket command streaming - real-time bidirectional agent control."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio

router = APIRouter()


class CommandStreamManager:
    """Manages WebSocket command streaming for agents."""
    
    def __init__(self):
        self.active_streams: dict[str, WebSocket] = {}
        self.agent_streams: dict[UUID, str] = {}  # agent_id -> client_id
    
    async def connect(self, websocket: WebSocket, client_id: str, agent_id: UUID | None = None):
        """Connect a WebSocket for command streaming."""
        await websocket.accept()
        self.active_streams[client_id] = websocket
        
        if agent_id:
            self.agent_streams[agent_id] = client_id
        
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "agent_id": str(agent_id) if agent_id else None,
        })
    
    def disconnect(self, client_id: str):
        """Disconnect a client."""
        self.active_streams.pop(client_id, None)
        
        # Remove from agent streams
        agent_ids_to_remove = [
            aid for aid, cid in self.agent_streams.items() if cid == client_id
        ]
        for aid in agent_ids_to_remove:
            self.agent_streams.pop(aid, None)
    
    async def send_to_client(self, client_id: str, message: dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.active_streams:
            try:
                await self.active_streams[client_id].send_json(message)
            except Exception:
                self.disconnect(client_id)
    
    async def send_to_agent_controller(self, agent_id: UUID, message: dict[str, Any]):
        """Send message to the client controlling an agent."""
        client_id = self.agent_streams.get(agent_id)
        if client_id:
            await self.send_to_client(client_id, message)
    
    async def stream_result(self, client_id: str, result: dict[str, Any]):
        """Stream command execution result."""
        await self.send_to_client(client_id, {
            "type": "command_result",
            "result": result,
        })
    
    async def stream_error(self, client_id: str, error: str):
        """Stream command error."""
        await self.send_to_client(client_id, {
            "type": "command_error",
            "error": error,
        })


# Global stream manager
stream_manager = CommandStreamManager()


@router.websocket("/commands/stream")
async def websocket_command_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agent commands.
    
    Protocol:
    - Client sends: {"type": "connect", "agent_id": "..."}
    - Client sends: {"type": "command", "command": {"type": "move_to", "parameters": {...}}}
    - Server sends: {"type": "command_result", "result": {...}}
    - Server sends: {"type": "command_error", "error": "..."}
    - Server sends: {"type": "agent_event", "event": {...}}
    """
    client_id = str(id(websocket))
    agent_id = None
    
    try:
        # Wait for connection message with optional agent_id
        data = await websocket.receive_json()
        
        if data.get("type") == "connect":
            agent_id_str = data.get("agent_id")
            if agent_id_str:
                try:
                    agent_id = UUID(agent_id_str)
                except ValueError:
                    await websocket.close(code=1003, reason="Invalid agent_id")
                    return
        
        await stream_manager.connect(websocket, client_id, agent_id)
        
        # Get simulation runner
        from src.main import get_simulation_runner
        simulation = get_simulation_runner()
        
        if simulation is None:
            await websocket.send_json({
                "type": "error",
                "error": "Simulation not initialized"
            })
            await websocket.close()
            return
        
        # Main command loop
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "unknown")
            
            if message_type == "command":
                # Execute command
                command_data = data.get("command", {})
                command_type = command_data.get("type")
                parameters = command_data.get("parameters", {})
                target_agent_id = data.get("agent_id", agent_id)
                
                if not target_agent_id:
                    await stream_manager.stream_error(client_id, "agent_id required")
                    continue
                
                if isinstance(target_agent_id, str):
                    try:
                        target_agent_id = UUID(target_agent_id)
                    except ValueError:
                        await stream_manager.stream_error(client_id, "Invalid agent_id")
                        continue
                
                # Get agent
                agent_ref = simulation._agents.get(target_agent_id)
                if not agent_ref:
                    await stream_manager.stream_error(
                        client_id, 
                        f"Agent {target_agent_id} not found"
                    )
                    continue
                
                # Execute command based on type
                try:
                    result = await execute_streamed_command(
                        agent_ref, 
                        command_type, 
                        parameters
                    )
                    
                    await stream_manager.stream_result(client_id, {
                        "command_type": command_type,
                        "agent_id": str(target_agent_id),
                        "data": result,
                    })
                    
                except Exception as e:
                    await stream_manager.stream_error(client_id, str(e))
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif message_type == "disconnect":
                break
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        stream_manager.disconnect(client_id)


async def execute_streamed_command(
    agent_ref,
    command_type: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Execute a command and return result."""
    
    # Map command types to agent methods
    if command_type == "move_to":
        lat = parameters.get("lat")
        lon = parameters.get("lon")
        altitude = parameters.get("altitude", 10.0)
        result = await agent_ref.move_to.remote(lat, lon, altitude)
    
    elif command_type == "explore_random":
        max_distance_km = parameters.get("max_distance_km", 10.0)
        result = await agent_ref.explore_random_location.remote(max_distance_km)
    
    elif command_type == "explore_topic":
        topic = parameters.get("topic")
        result = await agent_ref.explore_topic.remote(topic)
    
    elif command_type == "autonomous_step":
        result = await agent_ref.autonomous_step.remote()
    
    elif command_type == "get_state":
        result = await agent_ref.get_state.remote()
    
    elif command_type == "query_nearby":
        feature_type = parameters.get("feature_type")
        radius_km = parameters.get("radius_km", 1.0)
        limit = parameters.get("limit", 10)
        result = await agent_ref.query_nearby_features.remote(
            feature_type, radius_km, limit
        )
    
    elif command_type == "send_message":
        target_id = UUID(parameters.get("target_id"))
        content = parameters.get("content")
        result = await agent_ref.send_message.remote(target_id, content)
    
    elif command_type == "navigate_to_poi":
        poi_type = parameters.get("poi_type")
        result = await agent_ref.navigate_to_poi.remote(poi_type)
    
    else:
        raise ValueError(f"Unknown command type: {command_type}")
    
    return result if isinstance(result, dict) else {"data": result}


@router.get("/commands/stream/connections")
async def get_stream_connections() -> dict[str, Any]:
    """Get active command stream connections."""
    return {
        "active_connections": len(stream_manager.active_streams),
        "agent_streams": {
            str(agent_id): client_id 
            for agent_id, client_id in stream_manager.agent_streams.items()
        },
    }
