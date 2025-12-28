"""Collaboration API endpoints - team management and collaborative tasks."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field


router = APIRouter()


class TeamCreate(BaseModel):
    """Request to create a team."""
    team_name: str
    agent_ids: list[UUID]
    shared_goal: dict[str, Any] | None = None


class CoordinatedExploration(BaseModel):
    """Request for coordinated exploration."""
    agent_ids: list[UUID]
    target_area: dict[str, float]  # {"lat": ..., "lon": ..., "radius_km": ...}
    strategy: str = "spread"  # "spread", "cluster", "sequential"


class KnowledgeShare(BaseModel):
    """Request to share knowledge."""
    source_agent_id: UUID
    target_agent_ids: list[UUID]
    topic: str | None = None


class CollaborativeLearning(BaseModel):
    """Request for collaborative learning."""
    agent_ids: list[UUID]
    topic: str
    rounds: int = 3


def get_simulation():
    """Get simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


@router.post("/teams")
async def create_team(
    request: TeamCreate,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Create a new agent team."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    from src.agents.collaboration import TeamManager, CollaborationProtocol
    
    # Initialize team manager if not exists
    if not hasattr(simulation, 'team_manager'):
        simulation.team_manager = TeamManager()
    
    # Get agent refs
    agent_refs = []
    for agent_id in request.agent_ids:
        agent_ref = simulation._agents.get(agent_id)
        if not agent_ref:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        agent_refs.append(agent_ref)
    
    # Form team
    team_result = await CollaborationProtocol.form_team(
        agent_refs,
        request.team_name,
        request.shared_goal,
    )
    
    # Register in team manager
    team_id = team_result["team_id"]
    team = simulation.team_manager.create_team(
        team_id,
        request.team_name,
        request.agent_ids,
        request.shared_goal,
    )
    
    return {
        "team": team,
        "formation_result": team_result,
    }


@router.get("/teams")
async def list_teams(
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """List all active teams."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    if not hasattr(simulation, 'team_manager'):
        return {"teams": [], "count": 0}
    
    teams = simulation.team_manager.list_teams()
    return {"teams": teams, "count": len(teams)}


@router.get("/teams/{team_id}")
async def get_team(
    team_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get team details."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    if not hasattr(simulation, 'team_manager'):
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = simulation.team_manager.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return team


@router.delete("/teams/{team_id}")
async def dissolve_team(
    team_id: str,
    simulation=Depends(get_simulation),
) -> dict[str, str]:
    """Dissolve a team."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    if not hasattr(simulation, 'team_manager'):
        raise HTTPException(status_code=404, detail="Team not found")
    
    success = simulation.team_manager.dissolve_team(team_id)
    if not success:
        raise HTTPException(status_code=404, detail="Team not found")
    
    return {"message": f"Team {team_id} dissolved"}


@router.post("/explore/coordinated")
async def coordinate_exploration(
    request: CoordinatedExploration,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Coordinate multi-agent exploration."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    from src.agents.collaboration import CollaborationProtocol
    
    # Get agent refs
    agent_refs = []
    for agent_id in request.agent_ids:
        agent_ref = simulation._agents.get(agent_id)
        if not agent_ref:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        agent_refs.append(agent_ref)
    
    results = await CollaborationProtocol.coordinate_exploration(
        agent_refs,
        request.target_area,
        request.strategy,
    )
    
    return {
        "strategy": request.strategy,
        "target_area": request.target_area,
        "results": results,
        "agents_deployed": len(results),
    }


@router.post("/knowledge/share")
async def share_knowledge(
    request: KnowledgeShare,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Share knowledge from one agent to others."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    from src.agents.collaboration import CollaborationProtocol
    
    # Get source agent
    source_ref = simulation._agents.get(request.source_agent_id)
    if not source_ref:
        raise HTTPException(status_code=404, detail="Source agent not found")
    
    # Get target agents
    target_refs = []
    for agent_id in request.target_agent_ids:
        agent_ref = simulation._agents.get(agent_id)
        if not agent_ref:
            raise HTTPException(status_code=404, detail=f"Target agent {agent_id} not found")
        target_refs.append(agent_ref)
    
    result = await CollaborationProtocol.share_knowledge(
        source_ref,
        target_refs,
        request.topic,
    )
    
    return result


@router.post("/learn/collaborative")
async def collaborative_learning(
    request: CollaborativeLearning,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Agents collaboratively learn about a topic."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    from src.agents.collaboration import CollaborationProtocol
    
    # Get agent refs
    agent_refs = []
    for agent_id in request.agent_ids:
        agent_ref = simulation._agents.get(agent_id)
        if not agent_ref:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        agent_refs.append(agent_ref)
    
    results = await CollaborationProtocol.collaborative_learning(
        agent_refs,
        request.topic,
        request.rounds,
    )
    
    return {
        "topic": request.topic,
        "rounds": request.rounds,
        "participants": len(agent_refs),
        "results": results,
    }


@router.post("/decide/consensus")
async def consensus_decision(
    agent_ids: list[UUID],
    decision_topic: str,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Agents reach consensus on a decision."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    from src.agents.collaboration import CollaborationProtocol
    
    # Get agent refs
    agent_refs = []
    for agent_id in agent_ids:
        agent_ref = simulation._agents.get(agent_id)
        if not agent_ref:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        agent_refs.append(agent_ref)
    
    result = await CollaborationProtocol.consensus_decision(
        agent_refs,
        decision_topic,
    )
    
    return result
