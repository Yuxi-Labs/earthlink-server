"""Bridge between simulation events and WebSocket broadcasting."""

from typing import TYPE_CHECKING

from src.simulation.events import Event, EventType

if TYPE_CHECKING:
    from src.api.websocket import ConnectionManager


class EventBridge:
    """Bridges EventBus events to WebSocket connections.

    Translates simulation events into WebSocket messages and broadcasts
    them to subscribed clients.
    """

    def __init__(self, ws_manager: "ConnectionManager") -> None:
        self.ws_manager = ws_manager
        self._event_type_map = self._build_event_type_map()

    def _build_event_type_map(self) -> dict[EventType, str]:
        """Map EventType to WebSocket message type prefixes."""
        return {
            # Agent lifecycle
            EventType.AGENT_SPAWNED: "agent.spawned",
            EventType.AGENT_UPDATED: "agent.updated",
            EventType.AGENT_STARTED: "agent.started",
            EventType.AGENT_STOPPED: "agent.stopped",
            EventType.AGENT_DESTROYED: "agent.destroyed",
            # Agent actions
            EventType.AGENT_ACTION: "agent.action",
            EventType.AGENT_OBSERVATION: "agent.observation",
            EventType.AGENT_REWARD: "agent.reward",
            # Learning events
            EventType.AGENT_LEARNED: "agent.learned",
            EventType.AGENT_GOAL_ACHIEVED: "agent.goal_achieved",
            EventType.AGENT_GOAL_FAILED: "agent.goal_failed",
            # World events
            EventType.WORLD_LOADED: "world.loaded",
            EventType.WORLD_UNLOADED: "world.unloaded",
            EventType.WORLD_UPDATED: "world.updated",
            # Knowledge events
            EventType.KNOWLEDGE_ACQUIRED: "agent.knowledge_acquired",
            EventType.KNOWLEDGE_QUERIED: "agent.knowledge_queried",
            # System events
            EventType.SIMULATION_STARTED: "simulation.started",
            EventType.SIMULATION_PAUSED: "simulation.paused",
            EventType.SIMULATION_RESUMED: "simulation.resumed",
            EventType.SIMULATION_STOPPED: "simulation.stopped",
            EventType.CHECKPOINT_SAVED: "simulation.checkpoint_saved",
            EventType.CHECKPOINT_LOADED: "simulation.checkpoint_loaded",
        }

    def event_to_message(self, event: Event) -> dict:
        """Convert Event to WebSocket message format."""
        msg_type = self._event_type_map.get(event.event_type, "unknown")

        message = {
            "type": msg_type,
            "event_id": str(event.id),
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
        }

        if event.source_agent_id:
            message["agent_id"] = str(event.source_agent_id)

        if event.source_world_id:
            message["world_id"] = event.source_world_id

        if event.metadata:
            message["metadata"] = event.metadata

        return message

    def handle_event(self, event: Event) -> None:
        """Handle an event by queueing it for broadcast.

        This is designed to be used as an EventBus handler.
        """
        message = self.event_to_message(event)
        self.ws_manager.queue_broadcast(message)

    async def handle_event_async(self, event: Event) -> None:
        """Handle an event with async broadcast.

        Use when you can await the broadcast.
        """
        message = self.event_to_message(event)
        await self.ws_manager.broadcast(message)

    def register_with_event_bus(self, event_bus) -> None:
        """Register this bridge as a handler for all event types."""
        for event_type in EventType:
            event_bus.subscribe(event_type, self.handle_event)

    def unregister_from_event_bus(self, event_bus) -> None:
        """Unregister this bridge from all event types."""
        for event_type in EventType:
            try:
                event_bus.unsubscribe(event_type, self.handle_event)
            except ValueError:
                pass  # Not subscribed


# Global bridge instance (initialized in main.py)
_bridge: EventBridge | None = None


def get_event_bridge() -> EventBridge | None:
    """Get the global event bridge instance."""
    return _bridge


def init_event_bridge(ws_manager: "ConnectionManager") -> EventBridge:
    """Initialize the global event bridge."""
    global _bridge
    _bridge = EventBridge(ws_manager)
    return _bridge
