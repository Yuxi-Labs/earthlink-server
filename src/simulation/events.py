"""Events for simulation communication."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class EventType(Enum):
    """Types of simulation events."""

    # Agent lifecycle
    AGENT_SPAWNED = auto()
    AGENT_STARTED = auto()
    AGENT_STOPPED = auto()
    AGENT_DESTROYED = auto()

    # Agent actions
    AGENT_ACTION = auto()
    AGENT_OBSERVATION = auto()
    AGENT_REWARD = auto()
    AGENT_LEARNING = auto()  # Added for knowledge exploration events

    # Learning events
    AGENT_LEARNED = auto()
    AGENT_GOAL_ACHIEVED = auto()
    AGENT_GOAL_FAILED = auto()

    # World events
    WORLD_LOADED = auto()
    WORLD_UNLOADED = auto()
    WORLD_UPDATED = auto()

    # Knowledge events
    KNOWLEDGE_ACQUIRED = auto()
    KNOWLEDGE_QUERIED = auto()

    # System events
    SIMULATION_STARTED = auto()
    SIMULATION_PAUSED = auto()
    SIMULATION_RESUMED = auto()
    SIMULATION_STOPPED = auto()
    CHECKPOINT_SAVED = auto()
    CHECKPOINT_LOADED = auto()


@dataclass
class Event:
    """Event in the simulation."""

    id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.AGENT_ACTION
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Source
    source_agent_id: UUID | None = None
    source_world_id: str | None = None

    # Payload
    data: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "source_agent_id": str(self.source_agent_id) if self.source_agent_id else None,
            "source_world_id": self.source_world_id,
            "data": self.data,
            "metadata": self.metadata,
        }


class EventBus:
    """Simple event bus for simulation events."""

    def __init__(self):
        self._handlers: dict[EventType, list[callable]] = {}
        self._history: list[Event] = []
        self._max_history = 1000

    def subscribe(self, event_type: EventType, handler: callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Call handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't stop propagation
                print(f"Event handler error: {e}")

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get recent event history."""
        if event_type:
            events = [e for e in self._history if e.event_type == event_type]
        else:
            events = self._history

        return events[-limit:]
