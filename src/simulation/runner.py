"""Simulation Runner - orchestrates agents and worlds."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
from uuid import UUID

import ray
from sqlalchemy import select

from src.db.database import async_session_maker
from src.db.models import Agent as AgentModel
from .events import Event, EventBus, EventType
from .world import World, WorldRegistry
from .episode import EpisodeManager
from .persistence import StatePersistence


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
    speed_multiplier: float = 1.0  # Speed control: 0.1 = slow, 1.0 = normal, 10.0 = fast

    # Agents
    max_agents: int = 100
    target_agents: int = 5  # Desired steady-state population
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


class SimulationRunner:
    """
    Main simulation orchestrator.
    
    Manages:
    - Agent lifecycle (spawn, train, deploy, destroy)
    - World loading and interactions
    - Learning loops
    - Event propagation
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
    ):
        self.config = config or SimulationConfig()
        self.state = SimulationState.IDLE

        # Components
        self.event_bus = EventBus()
        self.world_registry = WorldRegistry()
        self.episode_manager = EpisodeManager(max_episodes=1000)
        self.state_persistence = StatePersistence(storage_dir="snapshots")
        
        # Register world types
        from src.worlds.earth import EarthWorld
        self.world_registry.register_world_class("earth", EarthWorld)

        # Agent management
        self._agents: dict[UUID, ray.ObjectRef] = {}
        self._agent_worlds: dict[UUID, str] = {}  # agent_id -> world_id

        # Background tasks
        self._spawn_task: asyncio.Task | None = None

        # Statistics
        self._total_steps = 0
        self._episode_rewards: dict[UUID, list[float]] = defaultdict(list)

        # Ray initialization
        self._ray_initialized = False
        
        # Cluster management
        self._cluster_manager = None
        self._use_cluster = False

    async def initialize(self, use_cluster: bool = False, cluster_config=None) -> None:
        """Initialize simulation resources."""
        self._use_cluster = use_cluster
        
        if use_cluster:
            # Use cluster manager for distributed execution
            from ..cluster import ClusterConfig, init_cluster_manager
            
            config = cluster_config or ClusterConfig()
            self._cluster_manager = init_cluster_manager(config)
            await self._cluster_manager.initialize()
            self._ray_initialized = True
        else:
            # Use local Ray instance
            if not self._ray_initialized:
                ray.init(ignore_reinit_error=True)
                self._ray_initialized = True

        # Load persisted agents from database
        await self._load_persisted_agents()

        self.event_bus.publish(Event(
            event_type=EventType.SIMULATION_STARTED,
        ))

    async def _load_persisted_agents(self) -> None:
        """Load agents from database and recreate Ray actors."""
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentModel).where(AgentModel.state != "destroyed")
            )
            db_agents = result.scalars().all()

        for db_agent in db_agents:
            try:
                # Recreate Ray actor
                agent_ref = Agent.remote(name=db_agent.name, config=db_agent.config or {})
                
                # Verify agent ID matches
                ray_agent_id = UUID(await agent_ref.get_id.remote())
                
                # Store reference
                self._agents[db_agent.id] = agent_ref
                
                # Initialize components
                await agent_ref.initialize_components.remote(
                    memory_config=db_agent.config.get("memory", {}) if db_agent.config else {},
                    policy_config=db_agent.config.get("policy", {}) if db_agent.config else {},
                )
                
                print(f"Restored agent {db_agent.name} (ID: {db_agent.id})")
                
            except Exception as e:
                print(f"Failed to restore agent {db_agent.name}: {e}")

    async def shutdown(self) -> None:
        """Shutdown simulation and cleanup."""
        self.state = SimulationState.STOPPED

        # Destroy all agents
        agent_ids = list(self._agents.keys())
        for agent_id in agent_ids:
            await self.destroy_agent(agent_id)

        # Unload all worlds
        for world in self.world_registry._worlds.values():
            await world.unload()
        
        # Shutdown cluster if used
        if self._cluster_manager:
            from ..cluster import shutdown_cluster_manager
            await shutdown_cluster_manager()

        self.event_bus.publish(Event(
            event_type=EventType.SIMULATION_STOPPED,
        ))

    # -------------------------------------------------------------------------
    # Agent Management
    # -------------------------------------------------------------------------
    
    async def spawn_agent(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        placement_group: str | None = None,
    ) -> UUID:
        """Spawn a new agent."""
        from ..agents.core import Agent
        import random

        # Merge with default config
        agent_config = {**self.config.agent_config, **(config or {})}

        # Determine actor options
        actor_options = {}
        
        if self._cluster_manager and placement_group:
            # Use placement group for co-location/spreading
            strategy = self._cluster_manager.get_placement_strategy(placement_group)
            actor_options["scheduling_strategy"] = strategy
        
        # Create Ray actor with optional placement strategy
        if actor_options:
            agent_ref = Agent.options(**actor_options).remote(name=name, config=agent_config)
        else:
            agent_ref = Agent.remote(name=name, config=agent_config)

        # Get agent ID
        agent_id = UUID(await agent_ref.get_id.remote())

        # Store reference
        self._agents[agent_id] = agent_ref
        
        # Track placement if using cluster
        if self._cluster_manager and placement_group:
            self._cluster_manager.register_agent_placement(str(agent_id), placement_group)

        # Initialize components
        await agent_ref.initialize_components.remote(
            memory_config=agent_config.get("memory", {}),
            policy_config=agent_config.get("policy", {}),
        )
        
        # Spawn agent at random location within VW geographic bounds
        # Query database for actual extent, or use configured default
        spawn_bounds = await self._get_spawn_bounds()
        spawn_lat = random.uniform(spawn_bounds["min_lat"], spawn_bounds["max_lat"])
        spawn_lon = random.uniform(spawn_bounds["min_lon"], spawn_bounds["max_lon"])
        await agent_ref.set_earthlink_position.remote(spawn_lat, spawn_lon, 0.0)
        print(f"Spawned agent {name} at ({spawn_lat:.4f}, {spawn_lon:.4f})")

        # Persist agent to database
        try:
            async with async_session_maker() as db:
                db_agent = AgentModel(
                    id=agent_id,
                    name=name,
                    state="spawned",
                    config=agent_config,
                )
                db.add(db_agent)
                await db.commit()
                print(f"Persisted agent {name} (ID: {agent_id}) to database")
        except Exception as e:
            print(f"Failed to persist agent {name} to database: {e}")
            # Continue anyway - agent exists in Ray

        self.event_bus.publish(Event(
            event_type=EventType.AGENT_SPAWNED,
            source_agent_id=agent_id,
            data={"name": name, "config": agent_config},
        ))

        return agent_id

    async def destroy_agent(self, agent_id: UUID) -> bool:
        """Destroy an agent."""
        if agent_id not in self._agents:
            return False

        agent_ref = self._agents[agent_id]

        # Save checkpoint before destroying
        checkpoint_path = f"{self.config.checkpoint_dir}/agent_{agent_id}_final.pt"
        try:
            await agent_ref.save_checkpoint.remote(checkpoint_path)
        except Exception:
            pass

        # Update database state
        try:
            async with async_session_maker() as db:
                result = await db.execute(
                    select(AgentModel).where(AgentModel.id == agent_id)
                )
                db_agent = result.scalar_one_or_none()
                if db_agent:
                    db_agent.state = "destroyed"
                    db_agent.updated_at = datetime.utcnow()
                    await db.commit()
        except Exception as e:
            print(f"Failed to update agent state in database: {e}")

        # Remove from tracking
        del self._agents[agent_id]
        self._agent_worlds.pop(agent_id, None)

        # Remove from database
        from ..db.database import async_session_maker
        from ..db.models import Agent as AgentModel
        
        async with async_session_maker() as db:
            from sqlalchemy import select
            result = await db.execute(select(AgentModel).where(AgentModel.id == agent_id))
            db_agent = result.scalar_one_or_none()
            if db_agent:
                await db.delete(db_agent)
                await db.commit()

        self.event_bus.publish(Event(
            event_type=EventType.AGENT_DESTROYED,
            source_agent_id=agent_id,
        ))

        return True

    async def assign_agent_to_world(
        self,
        agent_id: UUID,
        world_id: str,
    ) -> bool:
        """Assign an agent to explore a world."""
        print(f"[DEBUG] Assigning agent {agent_id} to world {world_id}")
        
        if agent_id not in self._agents:
            print(f"[DEBUG] Agent {agent_id} not in self._agents")
            return False

        world = self.world_registry.get_world(world_id)
        if world is None:
            print(f"[DEBUG] World {world_id} not found in registry")
            return False

        # Load world if needed
        print(f"[DEBUG] World found, _is_loaded={getattr(world, '_is_loaded', 'MISSING')}")
        if not world._is_loaded:
            await world.load()
            self.event_bus.publish(Event(
                event_type=EventType.WORLD_LOADED,
                source_world_id=world_id,
            ))

        # Update agent's target world
        agent_ref = self._agents[agent_id]
        await agent_ref.set_target_world.remote(world_id)

        self._agent_worlds[agent_id] = world_id
        print(f"[DEBUG] Successfully assigned agent {agent_id} to world {world_id}")
        return True

    async def get_agent_state(self, agent_id: UUID) -> dict[str, Any] | None:
        """Get agent state."""
        if agent_id not in self._agents:
            return None

        agent_ref = self._agents[agent_id]
        return await agent_ref.get_state.remote()

    async def list_agents(self) -> list[dict[str, Any]]:
        """List all agents with their states."""
        states = []
        # Create a copy of items to avoid "dictionary changed size during iteration"
        agent_items = list(self._agents.items())
        for agent_id, agent_ref in agent_items:
            state = await agent_ref.get_state.remote()
            states.append(state)
        return states

    # -------------------------------------------------------------------------
    # World Management
    # -------------------------------------------------------------------------

    def create_world(
        self,
        world_id: str,
        name: str,
        world_type: str = "text",
        **config: Any,
    ) -> str:
        """Create a new world."""
        world = self.world_registry.create_world(
            world_id=world_id,
            name=name,
            world_type=world_type,
            **config,
        )
        return world.id

    def get_world(self, world_id: str) -> World | None:
        """Get world by ID."""
        return self.world_registry.get_world(world_id)

    def list_worlds(self) -> list[dict[str, Any]]:
        """List all worlds."""
        return self.world_registry.list_worlds()

    # -------------------------------------------------------------------------
    # Simulation Loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """Run the main simulation loop."""
        if self.state == SimulationState.RUNNING:
            return

        self.state = SimulationState.RUNNING

        # Start autonomous spawn manager
        if self._spawn_task is None or self._spawn_task.done():
            self._spawn_task = asyncio.create_task(self._spawn_manager())

        while self.state == SimulationState.RUNNING:
            start_time = asyncio.get_event_loop().time()

            # Calculate step delay with speed multiplier
            base_step_delay = 1.0 / self.config.steps_per_second
            step_delay = base_step_delay / self.config.speed_multiplier

            # Run one simulation step
            await self._step()

            # Check termination
            if self.config.max_steps and self._total_steps >= self.config.max_steps:
                break

            # Maintain timing
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed < step_delay:
                await asyncio.sleep(step_delay - elapsed)

        self.state = SimulationState.IDLE

        # Stop spawn manager
        if self._spawn_task:
            self._spawn_task.cancel()
            try:
                await self._spawn_task
            except asyncio.CancelledError:
                pass

    async def _step(self) -> None:
        """Execute one simulation step for all agents."""
        self._total_steps += 1

        # All agents exist in Earthlink (the VW), so just step them
        step_tasks = []
        for agent_id, agent_ref in self._agents.items():
            step_tasks.append(self._agent_autonomous_step(agent_id, agent_ref))

        # Run all steps concurrently
        if step_tasks:
            await asyncio.gather(*step_tasks, return_exceptions=True)

        # Periodic checkpointing
        if (
            self.config.checkpoint_every_n_steps > 0
            and self._total_steps % self.config.checkpoint_every_n_steps == 0
        ):
            await self._save_checkpoints()

        # Periodic logging
        if (
            self.config.log_every_n_steps > 0
            and self._total_steps % self.config.log_every_n_steps == 0
        ):
            self._log_stats()

    async def _spawn_manager(self) -> None:
        """Background loop to maintain target agent population."""
        while self.state == SimulationState.RUNNING:
            try:
                await self._maybe_spawn_agents()
            except Exception as e:
                print(f"[SPAWN] spawn manager error: {e}")
            await asyncio.sleep(self.config.spawn_interval_seconds)

    async def _maybe_spawn_agents(self) -> None:
        """Spawn agents until target population is reached (respecting max_agents)."""
        current = len(self._agents)
        target = min(self.config.target_agents, self.config.max_agents)
        if current >= target:
            return

        to_spawn = target - current

        for _ in range(to_spawn):
            # Defensive check against race conditions
            if len(self._agents) >= self.config.max_agents:
                break

            name = f"A{len(self._agents) + 1}"
            agent_id = await self.spawn_agent(name=name, config=self.config.agent_config)

            # Set agent status to idle
            agent_ref = self._agents.get(agent_id)
            if agent_ref:
                from src.agents.core import AgentStatus
                try:
                    await agent_ref.set_status.remote(AgentStatus.IDLE)
                except Exception:
                    pass

    async def _agent_autonomous_step(
        self,
        agent_id: UUID,
        agent_ref: ray.ObjectRef,
    ) -> None:
        """
        Run one autonomous step for an agent in Earthlink.
        
        Agent autonomously decides whether to:
        - Explore (move to new location in Earthlink)
        - Learn (query knowledge sources)
        - Interact (with environment/data at current location)
        """
        try:
            # Agent makes autonomous decision about what to do
            from src.agents.core import AgentStatus
            
            # Set status to exploring
            await agent_ref.set_status.remote(AgentStatus.EXPLORING)
            
            # Agent decides and executes action
            action = await agent_ref.autonomous_step.remote()
            
            # If agent moved, update its position
            if action and action.get("type") == "move":
                new_pos = action.get("position")
                if new_pos:
                    lat, lon, alt = new_pos
                    await agent_ref.set_earthlink_position.remote(lat, lon, alt)
                    
            # Publish action event
            self.event_bus.publish(Event(
                event_type=EventType.AGENT_ACTION,
                source_agent_id=agent_id,
                data={"action": action},
            ))
            
            # If knowledge was gained, publish learning event
            if action and action.get("knowledge_gained", 0) > 0:
                await agent_ref.set_status.remote(AgentStatus.LEARNING)
                self.event_bus.publish(Event(
                    event_type=EventType.AGENT_LEARNING,
                    source_agent_id=agent_id,
                    data={
                        "knowledge_gained": action.get("knowledge_gained"),
                        "topic": action.get("topic"),
                    },
                ))
            
        except Exception as e:
            print(f"[STEP] Agent {agent_id} step error: {e}")
            # Set to idle on error
            try:
                from src.agents.core import AgentStatus
                await agent_ref.set_status.remote(AgentStatus.IDLE)
            except:
                pass

    async def _agent_step(
        self,
        agent_id: UUID,
        agent_ref: ray.ObjectRef,
        world: World,
    ) -> None:
        """
        Run one step for a single agent.
        
        Agent autonomously decides whether to:
        - Explore knowledge (query Wikipedia, Reddit, etc.)
        - Execute policy action in world
        """
        try:
            # Agent autonomously decides what to do
            # This uses curiosity/prediction error to decide
            action = await agent_ref.autonomous_step.remote()

            # Publish event
            self.event_bus.publish(Event(
                event_type=EventType.AGENT_ACTION,
                source_agent_id=agent_id,
                source_world_id=world.id,
                data={"action": action},
            ))

            # If action is knowledge exploration, no world interaction needed
            if action.get("type") == "knowledge_exploration":
                # Agent explored external knowledge
                self.event_bus.publish(Event(
                    event_type=EventType.AGENT_LEARNING,
                    source_agent_id=agent_id,
                    data={
                        "topic": action.get("topic"),
                        "knowledge_gained": action.get("knowledge_gained", 0),
                    },
                ))
                return

            # Otherwise, interact with world for policy-based action
            # Get current observation from world
            observation = await world._get_observation()

            # World processes action
            next_observation, reward, done, info = await world.step(action)

            self.event_bus.publish(Event(
                event_type=EventType.AGENT_REWARD,
                source_agent_id=agent_id,
                data={"reward": reward},
            ))

            # Track episode reward
            self._episode_rewards[agent_id].append(reward)

            # Agent learns from transition
            if self._total_steps % self.config.train_every_n_steps == 0:
                import torch

                # Encode observations for transition
                obs_vector = self._encode_observation(observation)
                next_obs_vector = self._encode_observation(next_observation)

                transition = {
                    "state": torch.tensor(obs_vector, dtype=torch.float32),
                    "action": torch.tensor(action.get("action", 0), dtype=torch.float32),
                    "reward": reward,
                    "next_state": torch.tensor(next_obs_vector, dtype=torch.float32),
                    "done": done,
                    "info": info,
                }

                losses = await agent_ref.learn.remote(transition)

                self.event_bus.publish(Event(
                    event_type=EventType.AGENT_LEARNED,
                    source_agent_id=agent_id,
                    data={"losses": losses},
                ))

            # Handle episode end
            if done:
                await world.reset()
                episode_reward = sum(self._episode_rewards[agent_id])
                self._episode_rewards[agent_id] = []

                self.event_bus.publish(Event(
                    event_type=EventType.WORLD_UPDATED,
                    source_world_id=world.id,
                    data={"event": "episode_end", "total_reward": episode_reward},
                ))

        except Exception as e:
            print(f"Error in agent step {agent_id}: {e}")

    def _encode_observation(self, observation: dict[str, Any]) -> list[float]:
        """Encode observation to vector (placeholder)."""
        # TODO: Use proper encoder
        if "vector" in observation:
            return observation["vector"]

        # Default: zero vector
        return [0.0] * 256

    async def _save_checkpoints(self) -> None:
        """Save checkpoints for all agents."""
        for agent_id, agent_ref in self._agents.items():
            path = f"{self.config.checkpoint_dir}/agent_{agent_id}_step{self._total_steps}.pt"
            try:
                await agent_ref.save_checkpoint.remote(path)
            except Exception as e:
                print(f"Failed to save checkpoint for {agent_id}: {e}")

        self.event_bus.publish(Event(
            event_type=EventType.CHECKPOINT_SAVED,
            data={"step": self._total_steps},
        ))

    def _log_stats(self) -> None:
        """Log simulation statistics."""
        stats = {
            "total_steps": self._total_steps,
            "num_agents": len(self._agents),
            "num_worlds": len(self.world_registry._worlds),
        }
        print(f"[Step {self._total_steps}] Agents: {len(self._agents)}, Worlds: {len(self.world_registry._worlds)}")

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------

    def pause(self) -> None:
        """Pause simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self.event_bus.publish(Event(event_type=EventType.SIMULATION_PAUSED))

    def resume(self) -> None:
        """Resume simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.event_bus.publish(Event(event_type=EventType.SIMULATION_RESUMED))

    def stop(self) -> None:
        """Stop simulation."""
        self.state = SimulationState.STOPPED

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get simulation statistics."""
        stats = {
            "state": self.state.name,
            "total_steps": self._total_steps,
            "num_agents": len(self._agents),
            "num_worlds": len(self.world_registry._worlds),
            "event_history_size": len(self.event_bus._history),
        }
        
        # Add cluster stats if using cluster
        if self._cluster_manager:
            stats["cluster"] = self._cluster_manager.get_stats()
        
        return stats
    
    # -------------------------------------------------------------------------
    # Cluster Management
    # -------------------------------------------------------------------------
    
    async def create_agent_placement_group(
        self,
        name: str,
        num_agents: int,
        strategy: str | None = None,
    ) -> bool:
        """
        Create a placement group for agent co-location or spreading.
        
        Args:
            name: Name for the placement group
            num_agents: Number of agents to place
            strategy: SPREAD, PACK, or STRICT_SPREAD
            
        Returns:
            True if successful
        """
        if not self._cluster_manager:
            return False
        
        try:
            await self._cluster_manager.create_placement_group(name, num_agents, strategy)
            return True
        except Exception as e:
            print(f"Failed to create placement group {name}: {e}")
            return False
    
    def get_cluster_stats(self) -> dict[str, Any]:
        """Get cluster statistics."""
        if not self._cluster_manager:
            return {"enabled": False}
        
        return {
            "enabled": True,
            **self._cluster_manager.get_stats(),
        }
    
    async def _get_spawn_bounds(self) -> dict[str, float]:
        """
        Get geographic bounds for agent spawning.
        
        Queries the database for actual geographic extent of loaded OSM data.
        Falls back to configured default if database query fails.
        
        Returns:
            Dict with min_lat, max_lat, min_lon, max_lon
        """
        try:
            # Query database for actual geographic extent
            from src.db.database import async_session_maker
            from sqlalchemy import text
            
            async with async_session_maker() as db:
                # Get bounding box of all features in osm_features table
                result = await db.execute(text("""
                    SELECT 
                        ST_YMin(ST_Extent(geom)) as min_lat,
                        ST_YMax(ST_Extent(geom)) as max_lat,
                        ST_XMin(ST_Extent(geom)) as min_lon,
                        ST_XMax(ST_Extent(geom)) as max_lon
                    FROM osm_features
                    WHERE geom IS NOT NULL
                    LIMIT 1
                """))
                row = result.fetchone()
                
                if row and all(v is not None for v in row):
                    return {
                        "min_lat": float(row[0]),
                        "max_lat": float(row[1]),
                        "min_lon": float(row[2]),
                        "max_lon": float(row[3]),
                    }
        except Exception as e:
            print(f"Failed to query spawn bounds from database: {e}")
        
        # Fallback to Australia bounds (current default)
        return {
            "min_lat": -44.0,
            "max_lat": -10.0,
            "min_lon": 113.0,
            "max_lon": 154.0,
        }
