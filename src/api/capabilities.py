"""Agent Capabilities API - Expose all 15 agent capabilities to frontend."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends

router = APIRouter()


def get_simulation():
    """Get the global simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


async def get_agent_or_404(agent_id: UUID, simulation):
    """Get agent reference or raise 404."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not running")
    if agent_id not in simulation._agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return simulation._agents[agent_id]


# =============================================================================
# 1. PERCEIVE - What the agent is sensing
# =============================================================================

@router.get("/{agent_id}/perception")
async def get_perception(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's current perception state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        # Get perception data from agent
        state = await agent_ref.get_state.remote()
        perception_module = getattr(agent_ref, '_perception', None)
        
        # Build perception response
        return {
            "agent_id": str(agent_id),
            "perception": {
                "current_location": state.get("location", {}),
                "nearby_features": state.get("nearby_features", []),
                "recent_observations": state.get("recent_observations", [])[-10:],
                "attention_focus": state.get("attention_focus", None),
                "sensory_inputs": {
                    "spatial": state.get("spatial_perception", {}),
                    "social": state.get("social_perception", {}),
                    "data": state.get("data_perception", {}),
                },
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "perception": {}, "error": str(e)}


# =============================================================================
# 2. EXPLORE - Movement and exploration state
# =============================================================================

@router.get("/{agent_id}/exploration")
async def get_exploration(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's exploration state and history."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        metrics = state.get("metrics", {})
        
        return {
            "agent_id": str(agent_id),
            "exploration": {
                "current_position": state.get("location", {}),
                "distance_traveled_km": metrics.get("distance_traveled_km", 0),
                "areas_visited": metrics.get("areas_visited", []),
                "exploration_frontier": metrics.get("exploration_frontier", []),
                "coverage_percent": metrics.get("coverage_percent", 0),
                "movement_history": state.get("movement_history", [])[-20:],
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "exploration": {}, "error": str(e)}


# =============================================================================
# 3. LEARN - Learning state and progress
# =============================================================================

@router.get("/{agent_id}/learning")
async def get_learning(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's learning state and progress."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        metrics = state.get("metrics", {})
        
        return {
            "agent_id": str(agent_id),
            "learning": {
                "knowledge_items_learned": metrics.get("knowledge_items_learned", 0),
                "learning_rate": metrics.get("learning_rate", 0.01),
                "recent_learnings": state.get("recent_learnings", [])[-10:],
                "skill_levels": metrics.get("skill_levels", {}),
                "curriculum_stage": metrics.get("curriculum_stage", "basic"),
                "meta_learning_enabled": metrics.get("meta_learning_enabled", False),
                "transfer_domains": metrics.get("transfer_domains", []),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "learning": {}, "error": str(e)}


# =============================================================================
# 4. REASON - Hypotheses, predictions, causal inferences
# =============================================================================

@router.get("/{agent_id}/reasoning")
async def get_reasoning(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's reasoning state - hypotheses, predictions, causal relations."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        
        # Try to get reasoning engine state
        reasoning_state = state.get("reasoning", {})
        
        return {
            "agent_id": str(agent_id),
            "reasoning": {
                "hypotheses": reasoning_state.get("hypotheses", []),
                "predictions": reasoning_state.get("predictions", []),
                "causal_relations": reasoning_state.get("causal_relations", []),
                "prediction_accuracy": reasoning_state.get("prediction_accuracy", 0),
                "hypotheses_confirmed": reasoning_state.get("hypotheses_confirmed", 0),
                "hypotheses_rejected": reasoning_state.get("hypotheses_rejected", 0),
                "current_inference": reasoning_state.get("current_inference", None),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "reasoning": {}, "error": str(e)}


# =============================================================================
# 5. DECIDE - Decision state, utility calculations
# =============================================================================

@router.get("/{agent_id}/decisions")
async def get_decisions(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's decision-making state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        decision_state = state.get("decision", {})
        
        return {
            "agent_id": str(agent_id),
            "decisions": {
                "last_decision": decision_state.get("last_decision", {}),
                "decision_history": decision_state.get("decision_history", [])[-10:],
                "available_actions": decision_state.get("available_actions", []),
                "utility_breakdown": decision_state.get("utility_breakdown", {}),
                "risk_assessment": decision_state.get("risk_assessment", {}),
                "decision_strategy": decision_state.get("strategy", "balanced"),
                "total_decisions": decision_state.get("total_decisions", 0),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "decisions": {}, "error": str(e)}


# =============================================================================
# 6. ACT - Action execution state
# =============================================================================

@router.get("/{agent_id}/actions")
async def get_actions(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's action execution state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        metrics = state.get("metrics", {})
        
        return {
            "agent_id": str(agent_id),
            "actions": {
                "last_action": state.get("last_action", {}),
                "action_history": state.get("action_history", [])[-20:],
                "actions_taken": metrics.get("actions_taken", 0),
                "action_success_rate": metrics.get("action_success_rate", 0),
                "current_action": state.get("current_action", None),
                "action_queue": state.get("action_queue", []),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "actions": {}, "error": str(e)}


# =============================================================================
# 7. COMMUNICATE - Messages, negotiations
# =============================================================================

@router.get("/{agent_id}/communication")
async def get_communication(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's communication state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        metrics = state.get("metrics", {})
        
        return {
            "agent_id": str(agent_id),
            "communication": {
                "messages_sent": metrics.get("messages_sent", 0),
                "messages_received": metrics.get("messages_received", 0),
                "recent_messages": state.get("recent_messages", [])[-10:],
                "active_negotiations": state.get("active_negotiations", []),
                "communication_partners": state.get("communication_partners", []),
                "broadcast_history": state.get("broadcast_history", [])[-5:],
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "communication": {}, "error": str(e)}


# =============================================================================
# 8. MODEL - World model, agent models
# =============================================================================

@router.get("/{agent_id}/world-model")
async def get_world_model(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's world model state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        model_state = state.get("world_model", {})
        
        return {
            "agent_id": str(agent_id),
            "world_model": {
                "regions": model_state.get("regions", {}),
                "entities": model_state.get("entities", {}),
                "relationships": model_state.get("relationships", []),
                "confidence": model_state.get("confidence", 0),
                "last_updated": model_state.get("last_updated", None),
                "model_accuracy": model_state.get("accuracy", 0),
            },
            "agent_models": state.get("agent_models", {}),
            "concept_abstractions": state.get("concept_abstractions", []),
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "world_model": {}, "error": str(e)}


# =============================================================================
# 9. ADAPT - Adaptation state, strategy changes
# =============================================================================

@router.get("/{agent_id}/adaptation")
async def get_adaptation(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's adaptation state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        adapt_state = state.get("adaptation", {})
        
        return {
            "agent_id": str(agent_id),
            "adaptation": {
                "current_strategy": adapt_state.get("current_strategy", "balanced"),
                "strategy_history": adapt_state.get("strategy_history", [])[-10:],
                "detected_shifts": adapt_state.get("detected_shifts", []),
                "behavior_modifications": adapt_state.get("behavior_modifications", []),
                "adaptation_events": adapt_state.get("events", [])[-10:],
                "performance_baseline": adapt_state.get("performance_baseline", {}),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "adaptation": {}, "error": str(e)}


# =============================================================================
# 10. SELF-MONITOR - Performance tracking, diagnostics
# =============================================================================

@router.get("/{agent_id}/monitoring")
async def get_monitoring(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's self-monitoring state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        metrics = state.get("metrics", {})
        monitor_state = state.get("monitoring", {})
        
        return {
            "agent_id": str(agent_id),
            "monitoring": {
                "performance": {
                    "avg_step_duration_ms": metrics.get("avg_step_duration_ms", 0),
                    "p95_step_duration_ms": metrics.get("p95_step_duration_ms", 0),
                    "success_rate": metrics.get("success_rate", 0),
                    "error_count": metrics.get("error_count", 0),
                },
                "uncertainty": {
                    "decision_confidence": monitor_state.get("decision_confidence", 0),
                    "prediction_uncertainty": monitor_state.get("prediction_uncertainty", 0),
                    "calibration_score": monitor_state.get("calibration_score", 0),
                },
                "diagnosis": monitor_state.get("diagnosis", {}),
                "health_status": monitor_state.get("health_status", "healthy"),
                "alerts": monitor_state.get("alerts", []),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "monitoring": {}, "error": str(e)}


# =============================================================================
# 11. MEMORY - Hierarchical memory state
# =============================================================================

@router.get("/{agent_id}/memory")
async def get_memory(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's memory state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        memory_state = state.get("memory", {})
        
        return {
            "agent_id": str(agent_id),
            "memory": {
                "episodic": {
                    "count": len(memory_state.get("episodic", [])),
                    "recent": memory_state.get("episodic", [])[-5:],
                },
                "semantic": {
                    "count": len(memory_state.get("semantic", [])),
                    "categories": memory_state.get("semantic_categories", []),
                },
                "procedural": {
                    "count": len(memory_state.get("procedural", [])),
                    "skills": memory_state.get("procedural_skills", []),
                },
                "working": {
                    "items": memory_state.get("working", []),
                    "capacity": memory_state.get("working_capacity", 7),
                },
                "consolidation_pending": memory_state.get("consolidation_pending", 0),
                "total_memories": memory_state.get("total_memories", 0),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "memory": {}, "error": str(e)}


# =============================================================================
# 12. TRANSFER KNOWLEDGE - Knowledge sharing state
# =============================================================================

@router.get("/{agent_id}/knowledge-transfer")
async def get_knowledge_transfer(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's knowledge transfer state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        transfer_state = state.get("knowledge_transfer", {})
        
        return {
            "agent_id": str(agent_id),
            "knowledge_transfer": {
                "extracted_knowledge": transfer_state.get("extracted", [])[-10:],
                "shared_with_agents": transfer_state.get("shared_with", []),
                "received_from_agents": transfer_state.get("received_from", []),
                "collective_contributions": transfer_state.get("collective_contributions", 0),
                "transfer_success_rate": transfer_state.get("success_rate", 0),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "knowledge_transfer": {}, "error": str(e)}


# =============================================================================
# 13. EVOLVE - Evolution and replication state
# =============================================================================

@router.get("/{agent_id}/evolution")
async def get_evolution(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's evolution state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        evolution_state = state.get("evolution", {})
        
        return {
            "agent_id": str(agent_id),
            "evolution": {
                "generation": evolution_state.get("generation", 0),
                "parent_id": evolution_state.get("parent_id", None),
                "offspring_ids": evolution_state.get("offspring_ids", []),
                "fitness_score": evolution_state.get("fitness_score", 0),
                "mutations": evolution_state.get("mutations", []),
                "genetic_lineage": evolution_state.get("lineage", []),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "evolution": {}, "error": str(e)}


# =============================================================================
# 14. SPECIALIZE - Domain specialization state
# =============================================================================

@router.get("/{agent_id}/specialization")
async def get_specialization(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's specialization state."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        spec_state = state.get("specialization", {})
        
        return {
            "agent_id": str(agent_id),
            "specialization": {
                "primary_domain": spec_state.get("primary_domain", None),
                "expertise_levels": spec_state.get("expertise_levels", {}),
                "niche_discovered": spec_state.get("niche_discovered", None),
                "specialization_score": spec_state.get("specialization_score", 0),
                "domain_history": spec_state.get("domain_history", []),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "specialization": {}, "error": str(e)}


# =============================================================================
# 15. GENERATE OUTPUT - Generated artifacts
# =============================================================================

@router.get("/{agent_id}/outputs")
async def get_outputs(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get agent's generated outputs."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        output_state = state.get("outputs", {})
        
        return {
            "agent_id": str(agent_id),
            "outputs": {
                "maps": output_state.get("maps", [])[-5:],
                "summaries": output_state.get("summaries", [])[-5:],
                "theories": output_state.get("theories", [])[-5:],
                "reports": output_state.get("reports", [])[-5:],
                "total_outputs": output_state.get("total_outputs", 0),
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "outputs": {}, "error": str(e)}


# =============================================================================
# UNIFIED - Get all capabilities in one call
# =============================================================================

@router.get("/{agent_id}/all")
async def get_all_capabilities(
    agent_id: UUID,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Get all agent capabilities in a single response."""
    agent_ref = await get_agent_or_404(agent_id, simulation)
    
    try:
        state = await agent_ref.get_state.remote()
        metrics = state.get("metrics", {})
        
        return {
            "agent_id": str(agent_id),
            "capabilities": {
                "perceive": {
                    "active": True,
                    "recent_observations": len(state.get("recent_observations", [])),
                },
                "explore": {
                    "active": True,
                    "distance_km": metrics.get("distance_traveled_km", 0),
                },
                "learn": {
                    "active": True,
                    "knowledge_items": metrics.get("knowledge_items_learned", 0),
                },
                "reason": {
                    "active": bool(state.get("reasoning", {})),
                    "hypotheses": len(state.get("reasoning", {}).get("hypotheses", [])),
                },
                "decide": {
                    "active": bool(state.get("decision", {})),
                    "total_decisions": state.get("decision", {}).get("total_decisions", 0),
                },
                "act": {
                    "active": True,
                    "actions_taken": metrics.get("actions_taken", 0),
                },
                "communicate": {
                    "active": True,
                    "messages_sent": metrics.get("messages_sent", 0),
                },
                "model": {
                    "active": bool(state.get("world_model", {})),
                    "model_confidence": state.get("world_model", {}).get("confidence", 0),
                },
                "adapt": {
                    "active": bool(state.get("adaptation", {})),
                    "strategy": state.get("adaptation", {}).get("current_strategy", "balanced"),
                },
                "monitor": {
                    "active": True,
                    "health": state.get("monitoring", {}).get("health_status", "healthy"),
                },
                "memory": {
                    "active": bool(state.get("memory", {})),
                    "total_memories": state.get("memory", {}).get("total_memories", 0),
                },
                "transfer": {
                    "active": bool(state.get("knowledge_transfer", {})),
                    "contributions": state.get("knowledge_transfer", {}).get("collective_contributions", 0),
                },
                "evolve": {
                    "active": bool(state.get("evolution", {})),
                    "generation": state.get("evolution", {}).get("generation", 0),
                },
                "specialize": {
                    "active": bool(state.get("specialization", {})),
                    "primary_domain": state.get("specialization", {}).get("primary_domain", None),
                },
                "generate": {
                    "active": bool(state.get("outputs", {})),
                    "total_outputs": state.get("outputs", {}).get("total_outputs", 0),
                },
            },
        }
    except Exception as e:
        return {"agent_id": str(agent_id), "capabilities": {}, "error": str(e)}

