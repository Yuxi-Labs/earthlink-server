"""Simulation module - managing agent lifecycles and world state.

The simulation owns Earth as its state.
"""

from .config import SimulationConfig, SimulationState
from .runner import SimulationRunner
from .worlds.base.earth import Earth
from .events import Event, EventType, EventBus
from .episode import Episode, EpisodeManager
from .persistence import SimulationSnapshot, StatePersistence

__all__ = [
    "SimulationConfig",
    "SimulationState",
    "SimulationRunner",
    "Earth",
    "Event",
    "EventType",
    "EventBus",
    "Episode",
    "EpisodeManager",
    "SimulationSnapshot",
    "StatePersistence",
]
