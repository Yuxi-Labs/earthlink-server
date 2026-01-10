"""Simulation Runner - orchestrates agents and the world."""

import asyncio
import os
import random
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import ray
from sqlalchemy import select

from src.agents.core import AgentActor
from src.db.database import async_session_maker
from src.db.models import Agent as AgentModel
from .config import SimulationConfig, SimulationState
from .events import Event, EventBus, EventType
from .worlds.base.earth import Earth
from .episode import EpisodeManager
from .persistence import StatePersistence


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
        self.episode_manager = EpisodeManager(max_episodes=1000)
        self.state_persistence = StatePersistence(storage_dir="snapshots")
        
        # The world - simulation's state (created on initialize)
        self.world: Earth | None = None

        # Agent management
        self._agents: dict[UUID, ray.ObjectRef] = {}
        self._agent_counter = 0  # For sequential agent naming (A1, A2, A3...)

        # Background tasks
        self._spawn_task: asyncio.Task | None = None
        self._population_task: asyncio.Task | None = None

        # Statistics
        self._total_steps = 0
        self._episode_rewards: dict[UUID, list[float]] = defaultdict(list)

        # Ray initialization
        self._ray_initialized = False
        
        # Cluster management
        self._cluster_manager = None
        self._use_cluster = False

    async def initialize(
        self,
        use_cluster: bool = False,
        cluster_config=None,
        skip_db_restore: bool = False,
    ) -> None:
        """Initialize simulation resources including the world."""
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

        # Create and load Earth (simulation's state)
        self.world = Earth()
        await self.world.load()
        
        self.event_bus.publish(Event(
            event_type=EventType.WORLD_LOADED,
            source_world_id=self.world.id,
        ))

        # Load persisted agents from database (only if enabled)
        restore_agents = os.getenv("RESTORE_AGENTS_ON_STARTUP", "true").lower() == "true"
        if restore_agents and not skip_db_restore:
            await self._load_persisted_agents()
        elif not restore_agents:
            print("Agent restoration disabled (RESTORE_AGENTS_ON_STARTUP=false)")
        else:
            print("Agent restoration skipped because snapshot will be loaded")

        self.event_bus.publish(Event(
            event_type=EventType.SIMULATION_STARTED,
        ))
        
        # Start population maintenance task
        print(f"[INIT] Starting population maintenance task (target: {self.config.target_agents} agents)")
        self._population_task = asyncio.create_task(self._maintain_population())

    async def _load_persisted_agents(self) -> None:
        """Load agents from database and recreate Ray actors."""
        async with async_session_maker() as db:
            result = await db.execute(
                select(AgentModel).where(
                    AgentModel.lifecycle != "expires",
                    AgentModel.status != "retired"
                )
            )
            db_agents = result.scalars().all()

        print(f"[RESTORE] Found {len(db_agents)} agents in database to restore")
        
        for db_agent in db_agents:
            try:
                # Recreate Ray actor
                agent_ref = AgentActor.remote(name=db_agent.name, config=db_agent.config or {})
                
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

        
        # Cancel background tasks
        if self._population_task:
            self._population_task.cancel()
            try:
                await self._population_task
            except asyncio.CancelledError:
                pass
    async def shutdown(self) -> None:
        """Shutdown simulation and cleanup."""
        self.state = SimulationState.STOPPED

        # Destroy all agents
        agent_ids = list(self._agents.keys())
        for agent_id in agent_ids:
            await self.destroy_agent(agent_id)

        # Unload world
        if self.world:
            await self.world.unload()
            self.world = None
        
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
        """Spawn a new agent with individualized traits."""
        import random

        # Start with default config, then overlay any provided config
        agent_config = {**self.config.agent_config, **(config or {})}
        
        # =====================================================================
        # AGENT INDIVIDUALITY SYSTEM - EXPLORATION FOCUS
        # Each agent explores differently - varied exploration strategies
        # =====================================================================
        
        # CORE EXPLORATION TRAITS (what drives exploration)
        if "curiosity" not in agent_config:
            agent_config["curiosity"] = random.uniform(0.3, 0.9)  # Intrinsic drive to explore
        if "risk_tolerance" not in agent_config:
            agent_config["risk_tolerance"] = random.uniform(0.2, 0.8)  # Bold vs cautious exploration
        
        # EXPLORATION STRATEGY (how they explore)
        if "exploration_style" not in agent_config:
            # 0.0 = systematic/methodical, 1.0 = random/spontaneous
            agent_config["exploration_style"] = random.uniform(0.2, 0.8)
        if "depth_vs_breadth" not in agent_config:
            # 0.0 = breadth-first (cover wide area), 1.0 = depth-first (dig deep in one area)
            agent_config["depth_vs_breadth"] = random.uniform(0.3, 0.7)
        if "backtracking_tolerance" not in agent_config:
            # How willing to revisit areas (0.0 = never backtrack, 1.0 = happy to revisit)
            agent_config["backtracking_tolerance"] = random.uniform(0.1, 0.6)
        if "path_memory_strength" not in agent_config:
            # How well agent remembers where they've been (0.5 = weak, 1.0 = perfect)
            agent_config["path_memory_strength"] = random.uniform(0.5, 1.0)
        if "obstacle_persistence" not in agent_config:
            # Give up easily vs push through (0.2 = give up fast, 0.9 = very persistent)
            agent_config["obstacle_persistence"] = random.uniform(0.2, 0.9)
        if "collaborative_exploration" not in agent_config:
            # Explore alone vs coordinate with others (0.1 = solo, 0.8 = team player)
            agent_config["collaborative_exploration"] = random.uniform(0.1, 0.8)
        
        # LEARNING FROM EXPLORATION (how they extract value from exploration)
        if "learning_rate" not in agent_config:
            agent_config["learning_rate"] = random.uniform(0.005, 0.02)  # How fast they learn from places
        if "meta_learning_rate" not in agent_config:
            agent_config["meta_learning_rate"] = random.uniform(0.001, 0.005)  # Learning better exploration strategies
        if "memory_retention" not in agent_config:
            agent_config["memory_retention"] = random.uniform(0.7, 0.95)  # Retaining knowledge of explored areas
        
        # PERCEPTION DURING EXPLORATION (what they notice)
        if "perception_window" not in agent_config:
            agent_config["perception_window"] = random.randint(5, 20)  # How much they notice while exploring
        if "perception_confidence_floor" not in agent_config:
            agent_config["perception_confidence_floor"] = random.uniform(0.2, 0.6)  # Filter for interesting things
        if "detail_orientation" not in agent_config:
            # 0.0 = big picture only, 1.0 = notice every detail
            agent_config["detail_orientation"] = random.uniform(0.3, 0.9)
        
        # REASONING ABOUT EXPLORATION (understanding what they find)
        if "reasoning_depth" not in agent_config:
            agent_config["reasoning_depth"] = random.randint(2, 5)  # How deeply they analyze discoveries
        if "hypothesis_confidence_threshold" not in agent_config:
            agent_config["hypothesis_confidence_threshold"] = random.uniform(0.4, 0.8)  # When to trust patterns
        if "pattern_sensitivity" not in agent_config:
            # How quickly they spot patterns in explored areas (0.3 = slow, 0.9 = quick)
            agent_config["pattern_sensitivity"] = random.uniform(0.3, 0.9)
        
        # DECISION-MAKING DURING EXPLORATION (where to go next)
        if "decision_risk_bias" not in agent_config:
            agent_config["decision_risk_bias"] = random.uniform(-0.3, 0.3)  # Conservative vs aggressive routes
        if "exploration_bonus" not in agent_config:
            agent_config["exploration_bonus"] = random.uniform(0.05, 0.2)  # Bonus for going somewhere new
        if "goal_flexibility" not in agent_config:
            # How easily they abandon current exploration goal for new opportunity
            agent_config["goal_flexibility"] = random.uniform(0.2, 0.8)
        
        # ADAPTATION DURING EXPLORATION (handling the unexpected)
        if "adaptation_speed" not in agent_config:
            agent_config["adaptation_speed"] = random.uniform(0.3, 0.9)  # Adapting to new terrain/situations
        if "strategy_stickiness" not in agent_config:
            agent_config["strategy_stickiness"] = random.uniform(0.1, 0.7)  # Stick with exploration strategy vs try new approaches
        
        # SELF-MONITORING (tracking exploration effectiveness)
        if "monitoring_window" not in agent_config:
            agent_config["monitoring_window"] = random.randint(50, 200)  # How far back they assess performance
        if "error_sensitivity" not in agent_config:
            agent_config["error_sensitivity"] = random.uniform(0.5, 1.5)  # Learning from exploration mistakes
        
        # REMOVED: specialization (contradicts broad exploration)
        # REMOVED: social_preference (not exploration-focused)
        # REMOVED: message_frequency (not exploration-focused)
        # REMOVED: trust_initial (not exploration-focused)
        # REMOVED: mutation_rate (evolution not core to exploration)
        # REMOVED: fitness_selectivity (evolution not core to exploration)
        # REMOVED: transfer_willingness (not exploration-focused)
        # REMOVED: collective_weight (not exploration-focused)

        # Determine actor options
        actor_options = {}
        
        if self._cluster_manager and placement_group:
            # Use placement group for co-location/spreading
            strategy = self._cluster_manager.get_placement_strategy(placement_group)
            actor_options["scheduling_strategy"] = strategy
        
        # Create Ray actor with optional placement strategy
        if actor_options:
            agent_ref = AgentActor.options(**actor_options).remote(name=name, config=agent_config)
        else:
            agent_ref = AgentActor.remote(name=name, config=agent_config)

        # Get agent ID (get_id returns a string, convert to UUID)
        agent_id_str = await agent_ref.get_id.remote()
        agent_id = UUID(agent_id_str) if isinstance(agent_id_str, str) else agent_id_str

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
        
        # Spawn agent at random GB city from EarthConfig
        from src.simulation.worlds.base.earth.config import EarthConfig
        earth_config = EarthConfig()
        spawn_lat, spawn_lon = random.choice(earth_config.spawn_locations)
        await agent_ref.set_earthlink_position.remote(spawn_lat, spawn_lon, 0.0)
        print(f"Spawned agent {name} at ({spawn_lat:.4f}, {spawn_lon:.4f})")

        # Persist agent to database
        try:
            agent_state = await agent_ref.get_state.remote()
            async with async_session_maker() as db:
                # Check if agent with this name already exists
                from sqlalchemy import select
                result = await db.execute(
                    select(AgentModel).where(AgentModel.name == name)
                )
                existing_agent = result.scalar_one_or_none()
                
                if existing_agent:
                    # Update existing agent instead of creating duplicate
                    existing_agent.lifecycle = agent_state.get("lifecycle", "spawned")
                    existing_agent.status = agent_state.get("status", "idle")
                    existing_agent.config = agent_config
                    existing_agent.updated_at = datetime.now(UTC).replace(tzinfo=None)
                    print(f"Updated existing agent {name} (ID: {existing_agent.id}) in database")
                else:
                    # Create new agent
                    db_agent = AgentModel(
                        id=str(agent_id),  # Convert UUID to string for PostgreSQL
                        name=name,
                        lifecycle=agent_state.get("lifecycle", "spawned"),
                        status=agent_state.get("status", "idle"),
                        state=None,
                        config=agent_config,
                    )
                    db.add(db_agent)
                    print(f"Persisted new agent {name} (ID: {agent_id}) to database")
                
                await db.commit()
        except Exception as e:
            import traceback
            print(f"Failed to persist agent {name} to database: {e}")
            traceback.print_exc()
            # Continue anyway - agent exists in Ray

        self.event_bus.publish(Event(
            event_type=EventType.AGENT_SPAWNED,
            source_agent_id=agent_id,
            data={
                "name": name,
                "config": agent_config,
                "lifecycle": agent_state.get("lifecycle", "spawned"),
                "status": agent_state.get("status", "idle"),
            },
        ))

        return agent_id

    async def _maybe_spawn_agents(self):
        """Maintain population up to target_agents for tests and runtime."""
        target = self.config.target_agents
        current = len(self._agents)
        if target is None or target <= 0:
            return

        while current < target:
            await self.spawn_agent(name=f"Agent_{self._agent_counter}")
            current = len(self._agents)

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
            from src.agents.core import AgentLifecycle, AgentStatus
            async with async_session_maker() as db:
                result = await db.execute(
                    select(AgentModel).where(AgentModel.id == agent_id)
                )
                db_agent = result.scalar_one_or_none()
                if db_agent:
                    db_agent.lifecycle = AgentLifecycle.EXPIRES.value
                    db_agent.status = AgentStatus.RETIRED.value
                    db_agent.state = None
                    db_agent.updated_at = datetime.now(UTC)
                    await db.commit()
        except Exception as e:
            print(f"Failed to update agent state in database: {e}")

        # Remove from tracking
        del self._agents[agent_id]

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
    # World State
    # -------------------------------------------------------------------------

    def get_world_state(self) -> dict[str, Any]:
        """Get current world state."""
        if not self.world:
            return {}
        return self.world.get_metadata()

    def get_world_metadata(self) -> dict[str, Any]:
        """Get world metadata."""
        if not self.world:
            return {}
        return self.world.get_metadata()

    # -------------------------------------------------------------------------
    # Simulation Loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """Run the main simulation loop."""
        if self.state == SimulationState.RUNNING:
            return

        self.state = SimulationState.RUNNING

        # Start autonomous population maintainer
        if self._population_task is None or self._population_task.done():
            self._population_task = asyncio.create_task(self._maintain_population())

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

        # Stop population maintainer
        if self._population_task:
            self._population_task.cancel()
            try:
                await self._population_task
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
        world_name = self.world.name if self.world else "None"
        print(f"[Step {self._total_steps}] Agents: {len(self._agents)}, World: {world_name}")

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
            "world": self.world.name if self.world else None,
            "world_loaded": self.world._is_loaded if self.world else False,
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
    
    async def _maintain_population(self) -> None:
        """
        Maintain target agent population autonomously.
        
        Spawns initial agents to reach target_agents count.
        Future: Will implement evolutionary spawning based on world conditions.
        """
        print(f"[POPULATION] Task started - target: {self.config.target_agents} agents")
        
        # If target is 0, exit immediately
        if self.config.target_agents == 0:
            print("[POPULATION] Target is 0, task exiting")
            return
        
        # Wait for initialization to complete before checking population
        await asyncio.sleep(2)
        
        while self.state != SimulationState.STOPPED:
            try:
                current_count = len(self._agents)
                target = self.config.target_agents
                
                if current_count < target:
                    needed = target - current_count
                    print(f"[POPULATION] {current_count}/{target} agents - spawning {needed}")
                    
                    for i in range(needed):
                        # Sequential agent naming: A1, A2, A3, ...
                        self._agent_counter += 1
                        agent_name = f"A{self._agent_counter}"
                        try:
                            # spawn_agent now generates individualized traits automatically
                            agent_id = await self.spawn_agent(name=agent_name)
                            # Get the agent's actual curiosity for logging
                            state = await self.get_agent_state(agent_id)
                            curiosity = state.get("metrics", {}).get("curiosity_score", 0.5) if state else 0.5
                            print(f"[POPULATION] Auto-spawned {agent_name} (curiosity: {curiosity:.2f})")
                        except Exception as e:
                            print(f"[POPULATION] Failed to spawn {agent_name}: {e}")
                
                # Check population every spawn_interval_seconds
                await asyncio.sleep(self.config.spawn_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[POPULATION] Error in population maintenance: {e}")
                await asyncio.sleep(self.config.spawn_interval_seconds)
    
    def get_cluster_stats(self) -> dict[str, Any]:
        """Get cluster statistics."""
        if not self._cluster_manager:
            return {"enabled": False}
        
        return {
            "enabled": True,
            **self._cluster_manager.get_stats(),
        }
    
    async def _get_spawn_bounds(self) -> dict[str, float]:
        """Return spawn bounds using EarthConfig (GB digital twin)."""
        from src.simulation.worlds.base.earth.config import EarthConfig

        earth_config = EarthConfig()
        if earth_config.spawn_bounds:
            return earth_config.spawn_bounds

        # Derive bounds from spawn_locations if explicit bounds not set
        lats = [lat for lat, _ in earth_config.spawn_locations]
        lons = [lon for _, lon in earth_config.spawn_locations]
        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
        }
