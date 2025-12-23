"""WebSocket endpoints for real-time communication."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str) -> None:
        """Remove a connection."""
        self.active_connections.pop(client_id, None)

    async def send_to_client(self, client_id: str, message: dict[str, Any]) -> None:
        """Send message to specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for streaming agent events."""
    client_id = str(id(websocket))
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()
            # Handle incoming messages
            message_type = data.get("type", "unknown")

            if message_type == "subscribe":
                # Subscribe to agent events
                agent_id = data.get("agent_id")
                await manager.send_to_client(
                    client_id,
                    {"type": "subscribed", "agent_id": agent_id},
                )

            elif message_type == "command":
                # Forward command to agent
                await manager.send_to_client(
                    client_id,
                    {"type": "command_received", "data": data},
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)


@router.websocket("/agent/{agent_id}")
async def websocket_agent(websocket: WebSocket, agent_id: UUID) -> None:
    """WebSocket endpoint for specific agent communication."""
    client_id = f"agent_{agent_id}_{id(websocket)}"
    await manager.connect(websocket, client_id)

    try:
        # Send initial agent state
        await manager.send_to_client(
            client_id,
            {
                "type": "agent.status",
                "agent_id": str(agent_id),
                "state": "connected",
            },
        )

        while True:
            data = await websocket.receive_json()
            # Process agent-specific commands
            await manager.send_to_client(
                client_id,
                {"type": "agent.response", "agent_id": str(agent_id), "data": data},
            )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
