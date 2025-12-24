"""World abstraction - data sources as worlds for agents to explore."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import torch


@dataclass
class WorldMetadata:
    """Metadata about a world."""

    id: str
    name: str
    description: str = ""
    world_type: str = "generic"  # text, spatial, temporal, relational, etc.

    # Dimensions
    observation_dim: int = 256
    action_dim: int = 64

    # Properties
    is_deterministic: bool = False
    is_episodic: bool = True
    max_steps: int | None = None

    # Data source info
    source_type: str = ""  # wikipedia, osm, csv, api, etc.
    source_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "world_type": self.world_type,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "is_deterministic": self.is_deterministic,
            "is_episodic": self.is_episodic,
            "max_steps": self.max_steps,
            "source_type": self.source_type,
        }


class World(ABC):
    """
    Abstract base class for worlds.
    
    A world is a data source that agents can explore.
    "Worlds = data" - feeding data IS how agents visit worlds.
    """

    def __init__(self, metadata: WorldMetadata):
        self.metadata = metadata
        self._current_step = 0
        self._is_loaded = False

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def name(self) -> str:
        return self.metadata.name

    @abstractmethod
    async def load(self) -> None:
        """Load world data/resources."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload world data/resources."""
        pass

    @abstractmethod
    async def reset(self) -> dict[str, Any]:
        """Reset world to initial state. Returns initial observation."""
        pass

    @abstractmethod
    async def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Take a step in the world.
        
        Args:
            action: Agent action
            
        Returns:
            (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> dict[str, Any]:
        """Return observation space specification."""
        pass

    @abstractmethod
    def get_action_space(self) -> dict[str, Any]:
        """Return action space specification."""
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Return world metadata."""
        return self.metadata.to_dict()


class TextWorld(World):
    """
    World based on text data.
    
    Agents explore by reading, querying, and navigating text.
    """

    def __init__(
        self,
        metadata: WorldMetadata,
        corpus: list[str] | None = None,
    ):
        super().__init__(metadata)
        self._corpus: list[str] = corpus or []
        self._current_position = 0
        self._embeddings: torch.Tensor | None = None

    async def load(self) -> None:
        """Load text corpus."""
        self._is_loaded = True
        # Compute embeddings if needed
        # (Would use sentence-transformers here)

    async def unload(self) -> None:
        """Unload corpus."""
        self._is_loaded = False
        self._embeddings = None

    async def reset(self) -> dict[str, Any]:
        """Reset to start of corpus."""
        self._current_position = 0
        self._current_step = 0
        return await self._get_observation()

    async def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Navigate through text corpus."""
        action_type = action.get("type", "read")

        reward = 0.0
        done = False
        info = {}

        if action_type == "read":
            # Read current text
            pass
        elif action_type == "next":
            # Move to next chunk
            self._current_position = min(
                self._current_position + 1,
                len(self._corpus) - 1,
            )
        elif action_type == "prev":
            # Move to previous chunk
            self._current_position = max(0, self._current_position - 1)
        elif action_type == "search":
            # Search for query
            query = action.get("query", "")
            found = self._search(query)
            if found >= 0:
                self._current_position = found
                reward = 0.1  # Reward for successful search

        self._current_step += 1

        if self.metadata.max_steps and self._current_step >= self.metadata.max_steps:
            done = True

        observation = await self._get_observation()
        return observation, reward, done, info

    async def _get_observation(self) -> dict[str, Any]:
        """Get current observation."""
        if not self._corpus:
            return {"text": "", "position": 0, "total": 0}

        return {
            "text": self._corpus[self._current_position],
            "position": self._current_position,
            "total": len(self._corpus),
        }

    def _search(self, query: str) -> int:
        """Search for query in corpus."""
        query_lower = query.lower()
        for i, text in enumerate(self._corpus):
            if query_lower in text.lower():
                return i
        return -1

    def get_observation_space(self) -> dict[str, Any]:
        return {
            "type": "text",
            "max_length": 4096,
            "embedding_dim": self.metadata.observation_dim,
        }

    def get_action_space(self) -> dict[str, Any]:
        return {
            "type": "discrete",
            "actions": ["read", "next", "prev", "search"],
        }

    def add_documents(self, documents: list[str]) -> None:
        """Add documents to corpus."""
        self._corpus.extend(documents)


class WorldRegistry:
    """Registry for managing available worlds."""

    def __init__(self):
        self._worlds: dict[str, World] = {}
        self._world_classes: dict[str, type[World]] = {
            "text": TextWorld,
        }

    def register_world_class(self, world_type: str, world_class: type[World]) -> None:
        """Register a new world type."""
        self._world_classes[world_type] = world_class

    def create_world(
        self,
        world_id: str,
        name: str,
        world_type: str = "text",
        **config: Any,
    ) -> World:
        """Create a new world instance."""
        if world_type not in self._world_classes:
            raise ValueError(f"Unknown world type: {world_type}")

        world_class = self._world_classes[world_type]
        metadata = WorldMetadata(
            id=world_id,
            name=name,
            world_type=world_type,
            **{k: v for k, v in config.items() if hasattr(WorldMetadata, k)},
        )

        world = world_class(metadata, **{k: v for k, v in config.items() if not hasattr(WorldMetadata, k)})
        self._worlds[world_id] = world
        return world

    def get_world(self, world_id: str) -> World | None:
        """Get world by ID."""
        return self._worlds.get(world_id)

    def list_worlds(self) -> list[dict[str, Any]]:
        """List all registered worlds."""
        return [w.get_metadata() for w in self._worlds.values()]

    def remove_world(self, world_id: str) -> bool:
        """Remove a world."""
        if world_id in self._worlds:
            del self._worlds[world_id]
            return True
        return False
