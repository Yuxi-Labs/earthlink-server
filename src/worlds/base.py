"""Base world interface for Earthlink environments."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class World(ABC):
    """
    Abstract base class for world environments.
    
    Worlds provide:
    - Agent spawn/despawn
    - Step dynamics (action -> observation/reward/done)
    - State persistence
    - Coordinate systems
    """
    
    def __init__(self, world_id: str, name: str):
        self.world_id = world_id
        self.name = name
    
    @abstractmethod
    async def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset world to initial state.
        
        Returns:
            Initial observations for all agents
        """
        pass
    
    @abstractmethod
    async def add_agent(self, agent_id: str, **kwargs) -> np.ndarray:
        """
        Add agent to world.
        
        Args:
            agent_id: Unique agent identifier
            **kwargs: World-specific spawn parameters
        
        Returns:
            Initial observation for agent
        """
        pass
    
    @abstractmethod
    async def remove_agent(self, agent_id: str):
        """Remove agent from world."""
        pass
    
    @abstractmethod
    async def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # dones
        Dict[str, Dict]         # infos
    ]:
        """
        Execute one step of world dynamics.
        
        Args:
            actions: Mapping of agent_id -> action index
        
        Returns:
            Tuple of (observations, rewards, dones, infos) for each agent
        """
        pass
    
    @abstractmethod
    async def get_observation(self, agent_id: str) -> np.ndarray:
        """
        Get current observation for specific agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Observation vector for agent's current state
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete world state for persistence.
        
        Returns:
            Dictionary containing all world state
        """
        pass
    
    @classmethod
    @abstractmethod
    async def from_state(cls, state: Dict[str, Any], **kwargs) -> "World":
        """
        Restore world from saved state.
        
        Args:
            state: Previously saved world state
            **kwargs: Additional context (e.g., database session)
        
        Returns:
            Restored World instance
        """
        pass
