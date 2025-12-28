"""State persistence API endpoints - save/load simulation snapshots."""

from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field


router = APIRouter()


class SnapshotRequest(BaseModel):
    """Request to save a snapshot."""
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def get_simulation():
    """Get simulation runner instance."""
    from src.main import get_simulation_runner
    return get_simulation_runner()


@router.post("/save")
async def save_snapshot(
    request: SnapshotRequest,
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """Save current simulation state to disk."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    try:
        snapshot_dir = await simulation.state_persistence.save_snapshot(
            simulation,
            name=request.name,
            metadata=request.metadata,
        )
        
        return {
            "message": "Snapshot saved successfully",
            "name": snapshot_dir.name,
            "path": str(snapshot_dir),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save snapshot: {e}")


@router.post("/load/{name}")
async def load_snapshot(
    name: str,
    simulation=Depends(get_simulation),
) -> dict[str, str]:
    """Load simulation state from disk."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    try:
        success = await simulation.state_persistence.load_snapshot(simulation, name)
        
        if success:
            return {"message": f"Snapshot '{name}' loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load snapshot")
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load snapshot: {e}")


@router.get("/list")
async def list_snapshots(
    simulation=Depends(get_simulation),
) -> dict[str, Any]:
    """List all available snapshots."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    snapshots = simulation.state_persistence.list_snapshots()
    
    return {
        "snapshots": snapshots,
        "count": len(snapshots),
    }


@router.delete("/{name}")
async def delete_snapshot(
    name: str,
    simulation=Depends(get_simulation),
) -> dict[str, str]:
    """Delete a snapshot."""
    if simulation is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    success = simulation.state_persistence.delete_snapshot(name)
    
    if success:
        return {"message": f"Snapshot '{name}' deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Snapshot '{name}' not found")
