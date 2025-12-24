"""Simulation Runner - orchestrates agents and worlds."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
from uuid import UUID

import ray

from .events import Event, EventBus, EventType
from .world import World, WorldRegistry


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

    # Agents
    max_agents: int = 100
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

        # Agent management
        self._agents: dict[UUID, ray.ObjectRef] = {}
        self._agent_worlds: dict[UUID, str] = {}  # agent_id -> world_id

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

        self.event_bus.publish(Event(
            event_type=EventType.SIMULATION_STARTED,
        ))

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
    placement_group: str | None = None,
    ) -> UUID:
        """Spawn a new agent."""
        from ..agents.core import Agent

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

        self.event_bus.publish(Event(
            event_type=EventType.AGENT_SPAWNED,
            source_agent_id=agent_id,
            data={"name": name, "config": agent_config, "placement_group": placement_group

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

        # Remove from tracking
        del self._agents[agent_id]
        self._agent_worlds.pop(agent_id, None)

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
        if agent_id not in self._agents:
            return False

        world = self.world_registry.get_world(world_id)
        if world is None:
            return False

        # Load world if needed
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
        for agent_id, agent_ref in self._agents.items():
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
        step_delay = 1.0 / self.config.steps_per_second

        while self.state == SimulationState.RUNNING:
            start_time = asyncio.get_event_loop().time()

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

    async def _step(self) -> None:
        """Execute one simulation step for all agents."""
        self._total_steps += 1

        # Gather all agent-world pairs
        step_tasks = []

        for agent_id, agent_ref in self._agents.items():
            world_id = self._agent_worlds.get(agent_id)
            if world_id:
                world = self.world_registry.get_world(world_id)
                if world:
                    step_tasks.append(self._agent_step(agent_id, agent_ref, world))

        # Run all steps concurrently
        if step_tasks:
            await asyncio.gather(*step_tasks)

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
