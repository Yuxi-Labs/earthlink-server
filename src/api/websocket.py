"""WebSocket endpoints for real-time communication."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class SubscriptionChannel(str, Enum):
    """Available subscription channels."""

    ALL = "all"  # All events
    AGENTS = "agents"  # All agent events
    WORLDS = "worlds"  # All world events
    SIMULATION = "simulation"  # Simulation control events
    METRICS = "metrics"  # Metric updates


@dataclass
class Subscription:
    """Client subscription."""

    channels: set[SubscriptionChannel] = field(default_factory=set)
    agent_ids: set[UUID] = field(default_factory=set)  # Specific agents to watch
    world_ids: set[str] = field(default_factory=set)  # Specific worlds to watch


class ConnectionManager:
    """Manages WebSocket connections with subscriptions."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self.subscriptions: dict[str, Subscription] = {}
        self._broadcast_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._running = False

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = Subscription()

    def disconnect(self, client_id: str) -> None:
        """Remove a connection."""
        self.active_connections.pop(client_id, None)
        self.subscriptions.pop(client_id, None)

    def subscribe(
        self,
        client_id: str,
        channels: list[str] | None = None,
        agent_ids: list[str] | None = None,
        world_ids: list[str] | None = None,
    ) -> None:
        """Update client subscriptions."""
        if client_id not in self.subscriptions:
            return

        sub = self.subscriptions[client_id]

        if channels:
            for ch in channels:
                try:
                    sub.channels.add(SubscriptionChannel(ch))
                except ValueError:
                    pass  # Invalid channel, skip

        if agent_ids:
            for aid in agent_ids:
                try:
                    sub.agent_ids.add(UUID(aid))
                except ValueError:
                    pass

        if world_ids:
            sub.world_ids.update(world_ids)

    def unsubscribe(
        self,
        client_id: str,
        channels: list[str] | None = None,
        agent_ids: list[str] | None = None,
        world_ids: list[str] | None = None,
    ) -> None:
        """Remove client subscriptions."""
        if client_id not in self.subscriptions:
            return

        sub = self.subscriptions[client_id]

        if channels:
            for ch in channels:
                sub.channels.discard(SubscriptionChannel(ch))

        if agent_ids:
            for aid in agent_ids:
                try:
                    sub.agent_ids.discard(UUID(aid))
                except ValueError:
                    pass

        if world_ids:
            for wid in world_ids:
                sub.world_ids.discard(wid)

    def _should_receive(self, client_id: str, message: dict[str, Any]) -> bool:
        """Check if client should receive this message based on subscriptions."""
        if client_id not in self.subscriptions:
            return False

        sub = self.subscriptions[client_id]

        # No subscriptions = receive nothing
        if not sub.channels and not sub.agent_ids and not sub.world_ids:
            return False

        # ALL channel receives everything
        if SubscriptionChannel.ALL in sub.channels:
            return True

        msg_type = message.get("type", "")
        agent_id = message.get("agent_id")
        world_id = message.get("world_id")

        # Check specific agent subscriptions
        if agent_id and sub.agent_ids:
            try:
                if UUID(agent_id) in sub.agent_ids:
                    return True
            except ValueError:
                pass

        # Check specific world subscriptions
        if world_id and world_id in sub.world_ids:
            return True

        # Check channel subscriptions
        if msg_type.startswith("agent.") and SubscriptionChannel.AGENTS in sub.channels:
            return True
        if msg_type.startswith("world.") and SubscriptionChannel.WORLDS in sub.channels:
            return True
        if msg_type.startswith("simulation.") and SubscriptionChannel.SIMULATION in sub.channels:
            return True
        if msg_type.startswith("metric") and SubscriptionChannel.METRICS in sub.channels:
            return True

        return False

    async def send_to_client(self, client_id: str, message: dict[str, Any]) -> None:
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception:
                # Client disconnected
                self.disconnect(client_id)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all subscribed clients."""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            if self._should_receive(client_id, message):
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

    async def broadcast_to_channel(
        self, channel: SubscriptionChannel, message: dict[str, Any]
    ) -> None:
        """Broadcast to specific channel subscribers."""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            sub = self.subscriptions.get(client_id)
            if sub and (channel in sub.channels or SubscriptionChannel.ALL in sub.channels):
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(client_id)

        for client_id in disconnected:
            self.disconnect(client_id)

    async def start_broadcast_worker(self) -> None:
        """Start background worker for queued broadcasts."""
        self._running = True
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._broadcast_queue.get(), timeout=1.0
                )
                await self.broadcast(message)
            except asyncio.TimeoutError:
                continue
            except Exception:
                pass

    def stop_broadcast_worker(self) -> None:
        """Stop the broadcast worker."""
        self._running = False

    def queue_broadcast(self, message: dict[str, Any]) -> None:
        """Queue a message for broadcast (non-async)."""
        try:
            self._broadcast_queue.put_nowait(message)
        except asyncio.QueueFull:
            pass  # Drop message if queue is full

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_client_subscriptions(self, client_id: str) -> dict[str, Any] | None:
        """Get client's current subscriptions."""
        if client_id not in self.subscriptions:
            return None

        sub = self.subscriptions[client_id]
        return {
            "channels": [ch.value for ch in sub.channels],
            "agent_ids": [str(aid) for aid in sub.agent_ids],
            "world_ids": list(sub.world_ids),
        }


# Global manager instance
manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for streaming events.

    Message protocol:
    - Client sends: {"type": "subscribe", "channels": ["agents", "simulation"], "agent_ids": [...]}
    - Client sends: {"type": "unsubscribe", "channels": ["agents"]}
    - Client sends: {"type": "ping"}
    - Server sends: {"type": "pong"}
    - Server sends: {"type": "agent.action", "agent_id": "...", "data": {...}}
    """
    client_id = str(id(websocket))
    await manager.connect(websocket, client_id)

    # Send connection confirmation
    await manager.send_to_client(
        client_id,
        {
            "type": "connected",
            "client_id": client_id,
            "available_channels": [ch.value for ch in SubscriptionChannel],
        },
    )

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "unknown")

            if message_type == "subscribe":
                manager.subscribe(
                    client_id,
                    channels=data.get("channels"),
                    agent_ids=data.get("agent_ids"),
                    world_ids=data.get("world_ids"),
                )
                await manager.send_to_client(
                    client_id,
                    {
                        "type": "subscribed",
                        "subscriptions": manager.get_client_subscriptions(client_id),
                    },
                )

            elif message_type == "unsubscribe":
                manager.unsubscribe(
                    client_id,
                    channels=data.get("channels"),
                    agent_ids=data.get("agent_ids"),
                    world_ids=data.get("world_ids"),
                )
                await manager.send_to_client(
                    client_id,
                    {
                        "type": "unsubscribed",
                        "subscriptions": manager.get_client_subscriptions(client_id),
                    },
                )

            elif message_type == "ping":
                await manager.send_to_client(client_id, {"type": "pong"})

            elif message_type == "get_subscriptions":
                await manager.send_to_client(
                    client_id,
                    {
                        "type": "subscriptions",
                        "subscriptions": manager.get_client_subscriptions(client_id),
                    },
                )

            else:
                await manager.send_to_client(
                    client_id,
                    {"type": "error", "message": f"Unknown message type: {message_type}"},
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)


@router.websocket("/agent/{agent_id}")
async def websocket_agent(websocket: WebSocket, agent_id: UUID) -> None:
    """WebSocket endpoint for specific agent communication.

    Automatically subscribes to this agent's events.
    """
    client_id = f"agent_{agent_id}_{id(websocket)}"
    await manager.connect(websocket, client_id)

    # Auto-subscribe to this agent
    manager.subscribe(client_id, agent_ids=[str(agent_id)])

    # Send initial connection info
    await manager.send_to_client(
        client_id,
        {
            "type": "agent.connected",
            "agent_id": str(agent_id),
            "subscriptions": manager.get_client_subscriptions(client_id),
        },
    )

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "unknown")

            if message_type == "command":
                # Agent commands will be handled by command router
                await manager.send_to_client(
                    client_id,
                    {
                        "type": "agent.command_received",
                        "agent_id": str(agent_id),
                        "command": data.get("command"),
                    },
                )

            elif message_type == "ping":
                await manager.send_to_client(client_id, {"type": "pong"})

            else:
                await manager.send_to_client(
                    client_id,
                    {"type": "error", "message": f"Unknown message type: {message_type}"},
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)


@router.websocket("/simulation")
async def websocket_simulation(websocket: WebSocket) -> None:
    """WebSocket endpoint for simulation events.

    Automatically subscribes to simulation channel.
    """
    client_id = f"sim_{id(websocket)}"
    await manager.connect(websocket, client_id)

    # Auto-subscribe to simulation events
    manager.subscribe(client_id, channels=["simulation", "agents"])

    await manager.send_to_client(
        client_id,
        {
            "type": "simulation.connected",
            "subscriptions": manager.get_client_subscriptions(client_id),
        },
    )

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "unknown")

            if message_type == "ping":
                await manager.send_to_client(client_id, {"type": "pong"})

            elif message_type == "subscribe":
                manager.subscribe(
                    client_id,
                    channels=data.get("channels"),
                    agent_ids=data.get("agent_ids"),
                )
                await manager.send_to_client(
                    client_id,
                    {
                        "type": "subscribed",
                        "subscriptions": manager.get_client_subscriptions(client_id),
                    },
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)


@router.websocket("/agents/{agent_id}/stream")
async def stream_agent_events(websocket: WebSocket, agent_id: str):
    """Stream real-time learning events for a specific agent."""
    from sqlalchemy import text
    from db.database import async_session_maker
    from datetime import UTC, datetime
    
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connection_established",
            "agent_id": agent_id,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        
        last_event_id = 0
        
        while True:
            try:
                async with async_session_maker() as db:
                    query = text("""
                        SELECT id, agent_id, source, topic, content_summary,
                                knowledge_count, goal_context, curiosity_signal,
                                lat, lon, agent_state, created_at
                        FROM knowledge_acquisition_log
                        WHERE agent_id = :agent_id AND id > :last_id
                        ORDER BY created_at DESC LIMIT 10
                    """)
                    
                    result = await db.execute(query, {"agent_id": agent_id, "last_id": last_event_id})
                    rows = result.fetchall()
                    
                    for row in reversed(rows):
                        await websocket.send_json({
                            "type": "knowledge_acquisition",
                            "agent_id": str(row.agent_id),
                            "event_id": row.id,
                            "data": {
                                "source": row.source,
                                "topic": row.topic,
                                "summary": row.content_summary,
                                "knowledge_count": row.knowledge_count,
                                "goal_context": row.goal_context,
                                "curiosity_signal": row.curiosity_signal,
                                "location": {"lat": row.lat, "lon": row.lon} if row.lat and row.lon else None,
                                "agent_state": row.agent_state,
                            },
                            "timestamp": row.created_at.isoformat(),
                        })
                        last_event_id = max(last_event_id, row.id)
                
                await asyncio.sleep(0.5)
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                })
                await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@router.websocket("/analytics/live")
async def stream_analytics(websocket: WebSocket):
    """Stream live analytics aggregations every 2 seconds."""
    from sqlalchemy import text
    from db.database import async_session_maker
    from datetime import UTC, datetime
    
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connection_established",
            "stream": "analytics",
            "timestamp": datetime.now(UTC).isoformat(),
        })
        
        while True:
            try:
                async with async_session_maker() as db:
                    stats_query = text("""
                        SELECT COUNT(DISTINCT agent_id) as active_agents,
                                COUNT(*) as total_acquisitions,
                                SUM(knowledge_count) as total_knowledge_items,
                                COUNT(DISTINCT topic) as unique_topics,
                                AVG(curiosity_signal) as avg_curiosity,
                                MAX(created_at) as last_activity
                        FROM knowledge_acquisition_log
                        WHERE created_at > NOW() - INTERVAL '1 hour'
                    """)
                    
                    result = await db.execute(stats_query)
                    row = result.fetchone()
                    
                    topics_query = text("""
                        SELECT topic, COUNT(*) as exploration_count
                        FROM knowledge_acquisition_log
                        WHERE created_at > NOW() - INTERVAL '1 hour' AND topic IS NOT NULL
                        GROUP BY topic
                        ORDER BY exploration_count DESC LIMIT 5
                    """)
                    
                    topics_result = await db.execute(topics_query)
                    top_topics = [{"topic": r.topic, "count": r.exploration_count} for r in topics_result.fetchall()]
                    
                    await websocket.send_json({
                        "type": "analytics_update",
                        "data": {
                            "active_agents": row.active_agents or 0,
                            "total_acquisitions": row.total_acquisitions or 0,
                            "total_knowledge_items": row.total_knowledge_items or 0,
                            "unique_topics": row.unique_topics or 0,
                            "avg_curiosity": float(row.avg_curiosity) if row.avg_curiosity else 0.0,
                            "last_activity": row.last_activity.isoformat() if row.last_activity else None,
                            "top_topics": top_topics,
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                
                await asyncio.sleep(2)
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                })
                await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
