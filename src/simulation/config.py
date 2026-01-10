"""Simulation configuration."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SimulationState(Enum):
    """State of the simulation."""

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


@dataclass
class SimulationConfig:
    """Configuration for simulation."""

    # Timing
    steps_per_second: float = 10.0
    max_steps: int | None = None
    speed_multiplier: float = 1.0  # 0.1 = slow, 1.0 = normal, 10.0 = fast
    
    # Simulation time advancement
    minutes_per_step: float = 6.0  # Each step = 6 minutes of simulation time
    # Real-world physics: walking speed ~5 km/h, so 6 minutes = 0.5 km max movement

    # Agents
    max_agents: int = 100
    target_agents: int = 10  # Desired steady-state population
    spawn_interval_seconds: float = 5.0  # How often to check spawn conditions
    agent_config: dict[str, Any] = field(default_factory=dict)

    # Learning
    train_every_n_steps: int = 4
    batch_size: int = 32

    # Checkpointing
    checkpoint_every_n_steps: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Monitoring
    log_every_n_steps: int = 100

