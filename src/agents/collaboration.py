"""Agent collaboration behaviors - joint exploration and knowledge sharing."""

from typing import Any
from uuid import UUID
import asyncio

import ray


class CollaborationProtocol:
    """Protocol for agent collaboration."""
    
    @staticmethod
    async def form_team(
        agent_refs: list[ray.ObjectRef],
        team_name: str,
        shared_goal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Form a team of agents with shared goals."""
        team_id = f"team_{len(agent_refs)}_{hash(team_name)}"
        
        # Set shared goal for all agents
        if shared_goal:
            await asyncio.gather(*[
                ref.set_goal.remote(shared_goal["type"], shared_goal.get("params", {}))
                for ref in agent_refs
            ])
        
        return {
            "team_id": team_id,
            "team_name": team_name,
            "member_count": len(agent_refs),
            "shared_goal": shared_goal,
        }
    
    @staticmethod
    async def coordinate_exploration(
        agent_refs: list[ray.ObjectRef],
        target_area: dict[str, float],  # {"lat": ..., "lon": ..., "radius_km": ...}
        strategy: str = "spread",  # "spread", "cluster", "sequential"
    ) -> list[dict[str, Any]]:
        """Coordinate multi-agent exploration of an area."""
        
        center_lat = target_area["lat"]
        center_lon = target_area["lon"]
        radius_km = target_area.get("radius_km", 10.0)
        
        results = []
        
        if strategy == "spread":
            # Spread agents evenly across area
            import math
            num_agents = len(agent_refs)
            
            for i, agent_ref in enumerate(agent_refs):
                angle = (2 * math.pi * i) / num_agents
                offset_lat = (radius_km / 111) * math.cos(angle)
                offset_lon = (radius_km / 111) * math.sin(angle) / math.cos(math.radians(center_lat))
                
                target_lat = center_lat + offset_lat
                target_lon = center_lon + offset_lon
                
                result = await agent_ref.move_to.remote(target_lat, target_lon, 10.0)
                results.append({
                    "agent_index": i,
                    "target": {"lat": target_lat, "lon": target_lon},
                    "result": result,
                })
        
        elif strategy == "cluster":
            # All agents move to same location
            tasks = [
                ref.move_to.remote(center_lat, center_lon, 10.0)
                for ref in agent_refs
            ]
            move_results = await asyncio.gather(*tasks)
            results = [
                {"agent_index": i, "result": r}
                for i, r in enumerate(move_results)
            ]
        
        elif strategy == "sequential":
            # Agents explore in sequence
            for i, agent_ref in enumerate(agent_refs):
                result = await agent_ref.explore_random_location.remote(radius_km)
                results.append({
                    "agent_index": i,
                    "result": result,
                })
        
        return results
    
    @staticmethod
    async def share_knowledge(
        source_agent: ray.ObjectRef,
        target_agents: list[ray.ObjectRef],
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Share knowledge from one agent to others."""
        
        # Get source agent's state
        source_state = await source_agent.get_state.remote()
        
        # Get knowledge to share
        knowledge_items = source_state["metrics"].get("knowledge_items", 0)
        topics_explored = source_state["metrics"].get("topics_explored", 0)
        
        # For now, broadcast via messages
        # In future, could transfer actual memory/semantic knowledge
        share_message = {
            "type": "knowledge_share",
            "from_agent": source_state["name"],
            "knowledge_items": knowledge_items,
            "topics_explored": topics_explored,
            "topic": topic,
        }
        
        # Send to all target agents
        for target_ref in target_agents:
            try:
                await target_ref.receive_broadcast.remote(share_message)
            except Exception as e:
                print(f"Failed to share knowledge: {e}")
        
        return {
            "shared_from": source_state["name"],
            "shared_to_count": len(target_agents),
            "knowledge_items": knowledge_items,
            "topics": topics_explored,
        }
    
    @staticmethod
    async def collaborative_learning(
        agent_refs: list[ray.ObjectRef],
        topic: str,
        rounds: int = 3,
    ) -> list[dict[str, Any]]:
        """Agents collaboratively learn about a topic."""
        
        results = []
        
        for round_num in range(rounds):
            # All agents explore the topic
            tasks = [ref.explore_topic.remote(topic) for ref in agent_refs]
            round_results = await asyncio.gather(*tasks)
            
            # Share findings
            for i, agent_ref in enumerate(agent_refs):
                other_agents = [a for j, a in enumerate(agent_refs) if j != i]
                if other_agents:
                    await CollaborationProtocol.share_knowledge(
                        agent_ref,
                        other_agents,
                        topic=topic
                    )
            
            results.append({
                "round": round_num + 1,
                "explorations": round_results,
            })
        
        return results
    
    @staticmethod
    async def consensus_decision(
        agent_refs: list[ray.ObjectRef],
        decision_topic: str,
    ) -> dict[str, Any]:
        """Agents reach consensus on a decision."""
        
        # Get all agent states
        states = await asyncio.gather(*[
            ref.get_state.remote() for ref in agent_refs
        ])
        
        # Simple voting based on curiosity signals
        votes = []
        for state in states:
            curiosity = state["metrics"].get("curiosity_value", 0.5)
            votes.append({
                "agent": state["name"],
                "vote": curiosity > 0.6,  # High curiosity = yes vote
                "confidence": curiosity,
            })
        
        yes_votes = sum(1 for v in votes if v["vote"])
        no_votes = len(votes) - yes_votes
        
        return {
            "decision_topic": decision_topic,
            "consensus": yes_votes > no_votes,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "votes": votes,
        }


class TeamManager:
    """Manages agent teams and collaboration."""
    
    def __init__(self):
        self.teams: dict[str, dict[str, Any]] = {}
        self.agent_teams: dict[UUID, str] = {}  # agent_id -> team_id
    
    def create_team(
        self,
        team_id: str,
        team_name: str,
        agent_ids: list[UUID],
        shared_goal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new team."""
        self.teams[team_id] = {
            "id": team_id,
            "name": team_name,
            "members": agent_ids,
            "shared_goal": shared_goal,
            "created_at": asyncio.get_event_loop().time(),
        }
        
        # Map agents to team
        for agent_id in agent_ids:
            self.agent_teams[agent_id] = team_id
        
        return self.teams[team_id]
    
    def get_team(self, team_id: str) -> dict[str, Any] | None:
        """Get team information."""
        return self.teams.get(team_id)
    
    def get_agent_team(self, agent_id: UUID) -> str | None:
        """Get team for an agent."""
        return self.agent_teams.get(agent_id)
    
    def get_team_members(self, team_id: str) -> list[UUID]:
        """Get team member IDs."""
        team = self.teams.get(team_id)
        return team["members"] if team else []
    
    def dissolve_team(self, team_id: str) -> bool:
        """Dissolve a team."""
        team = self.teams.pop(team_id, None)
        if team:
            # Remove agent mappings
            for agent_id in team["members"]:
                self.agent_teams.pop(agent_id, None)
            return True
        return False
    
    def list_teams(self) -> list[dict[str, Any]]:
        """List all active teams."""
        return list(self.teams.values())
