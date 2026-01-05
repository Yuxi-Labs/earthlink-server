"""Simulation state persistence - save/load simulation snapshots."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import torch


class SimulationSnapshot:
    """A snapshot of simulation state."""
    
    def __init__(
        self,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
        # Simulation state
        self.total_steps = 0
        self.simulation_state = "IDLE"
        
        # Agents
        self.agents: dict[str, dict[str, Any]] = {}
        self.agent_checkpoints: dict[str, bytes] = {}
        
        # World state
        self.world: dict[str, Any] | None = None
        
        # Episodes
        self.episodes: list[dict[str, Any]] = []
        
        # Statistics
        self.stats: dict[str, Any] = {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "total_steps": self.total_steps,
            "simulation_state": self.simulation_state,
            "agents": self.agents,
            "world": self.world,
            "episodes": self.episodes,
            "stats": self.stats,
        }


class StatePersistence:
    """Handles saving and loading simulation state."""
    
    def __init__(self, storage_dir: str = "snapshots"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    async def save_snapshot(
        self,
        simulation_runner,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save current simulation state to disk."""
        
        timestamp = datetime.now()
        if name is None:
            name = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        snapshot = SimulationSnapshot(timestamp=timestamp, metadata=metadata)
        
        # Capture simulation state
        snapshot.total_steps = simulation_runner._total_steps
        snapshot.simulation_state = simulation_runner.state.name
        snapshot.stats = simulation_runner.get_stats()
        
        # Capture agent states
        for agent_id, agent_ref in simulation_runner._agents.items():
            try:
                agent_state = await agent_ref.get_state.remote()
                snapshot.agents[str(agent_id)] = agent_state
                
                # Save agent checkpoint
                checkpoint_data = await agent_ref.get_checkpoint_data.remote()
                snapshot.agent_checkpoints[str(agent_id)] = checkpoint_data
                
            except Exception as e:
                print(f"Failed to capture agent {agent_id}: {e}")
        
        # Capture world state (single world)
        if simulation_runner.world:
            snapshot.world = {
                "id": simulation_runner.world.id,
                "name": simulation_runner.world.name,
                "current_step": simulation_runner.world._current_step,
                "is_loaded": simulation_runner.world._is_loaded,
            }
        
        # Capture episode manager state (if exists)
        if hasattr(simulation_runner, 'episode_manager'):
            episode_manager = simulation_runner.episode_manager
            snapshot.episodes = [
                ep.to_dict() for ep in episode_manager.get_recent_episodes(100)
            ]
            snapshot.stats.update(episode_manager.get_stats())
        
        # Save to disk
        snapshot_dir = self.storage_dir / name
        snapshot_dir.mkdir(exist_ok=True)
        
        # Save metadata as JSON
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        
        # Save agent checkpoints
        checkpoints_dir = snapshot_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        for agent_id, checkpoint_data in snapshot.agent_checkpoints.items():
            checkpoint_path = checkpoints_dir / f"agent_{agent_id}.pt"
            with open(checkpoint_path, "wb") as f:
                f.write(checkpoint_data)
        
        # Save full snapshot as pickle (includes everything)
        snapshot_path = snapshot_dir / "snapshot.pkl"
        with open(snapshot_path, "wb") as f:
            pickle.dump(snapshot, f)
        
        print(f"✓ Snapshot saved: {snapshot_dir}")
        return snapshot_dir
    
    async def load_snapshot(
        self,
        simulation_runner,
        name: str,
    ) -> bool:
        """Load simulation state from disk."""
        
        snapshot_dir = self.storage_dir / name
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_dir}")
        
        # Load pickle snapshot
        snapshot_path = snapshot_dir / "snapshot.pkl"
        with open(snapshot_path, "rb") as f:
            snapshot: SimulationSnapshot = pickle.load(f)
        
        print(f"Loading snapshot from {snapshot.timestamp.isoformat()}...")
        
        # Restore simulation state
        simulation_runner._total_steps = snapshot.total_steps
        
        # Restore world state
        if snapshot.world and simulation_runner.world:
            simulation_runner.world._current_step = snapshot.world["current_step"]
            if snapshot.world["is_loaded"] and not simulation_runner.world._is_loaded:
                await simulation_runner.world.load()
        
        # Restore agents
        checkpoints_dir = snapshot_dir / "checkpoints"
        
        for agent_id_str, agent_state in snapshot.agents.items():
            agent_id = UUID(agent_id_str)
            
            # Create agent (agents are automatically in the world)
            agent_ref = await simulation_runner.spawn_agent(
                name=agent_state["name"],
            )
            
            # Load checkpoint
            checkpoint_path = checkpoints_dir / f"agent_{agent_id_str}.pt"
            if checkpoint_path.exists():
                try:
                    await agent_ref.load_checkpoint.remote(str(checkpoint_path))
                except Exception as e:
                    print(f"Failed to load checkpoint for {agent_id}: {e}")
        
        # Restore episode manager (if exists)
        if hasattr(simulation_runner, 'episode_manager') and snapshot.episodes:
            # Episodes are historical, just log them
            print(f"  Loaded {len(snapshot.episodes)} historical episodes")
        
        print(f"✓ Snapshot loaded: {len(snapshot.agents)} agents")
        return True
    
    def list_snapshots(self) -> list[dict[str, Any]]:
        """List available snapshots."""
        snapshots = []
        
        for snapshot_dir in self.storage_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            
            metadata_path = snapshot_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                snapshots.append({
                    "name": snapshot_dir.name,
                    "timestamp": metadata["timestamp"],
                    "total_steps": metadata["total_steps"],
                    "num_agents": len(metadata["agents"]),
                    "world": metadata.get("world", {}).get("name"),
                    "metadata": metadata.get("metadata", {}),
                })
        
        return sorted(snapshots, key=lambda x: x["timestamp"], reverse=True)
    
    def delete_snapshot(self, name: str) -> bool:
        """Delete a snapshot."""
        snapshot_dir = self.storage_dir / name
        if not snapshot_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(snapshot_dir)
        print(f"✓ Snapshot deleted: {name}")
        return True
