"""Target worlds management API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class WorldCreate(BaseModel):
    """Schema for creating a target world."""

    name: str
    category: str  # physical, virtual, game, abstract
    description: str | None = None
    data_sources: list[str] = []


class WorldResponse(BaseModel):
    """Schema for world response."""

    id: UUID
    name: str
    category: str
    description: str | None
    data_sources: list[str]
    integration_status: str


@router.get("/")
async def list_worlds() -> list[dict[str, Any]]:
    """List all target worlds."""
    # TODO: Implement with database
    return [
        {"id": "mars", "name": "Mars", "category": "physical", "status": "available"},
        {"id": "moon", "name": "Moon", "category": "physical", "status": "available"},
        {"id": "second_life", "name": "Second Life", "category": "virtual", "status": "available"},
        {"id": "roblox", "name": "Roblox", "category": "virtual", "status": "available"},
    ]


@router.post("/")
async def create_world(world: WorldCreate) -> dict[str, Any]:
    """Register a new target world."""
    # TODO: Implement with database
    return {
        "id": "placeholder",
        "name": world.name,
        "category": world.category,
        "status": "pending",
    }


@router.get("/{world_id}")
async def get_world(world_id: str) -> dict[str, Any]:
    """Get target world by ID."""
    # TODO: Implement with database
    raise HTTPException(status_code=404, detail="World not found")


@router.get("/{world_id}/agents")
async def get_world_agents(world_id: str) -> list[dict[str, Any]]:
    """Get agents specialized for or deployed to this world."""
    # TODO: Implement with database
    return []


@router.get("/{world_id}/data-sources")
async def get_world_data_sources(world_id: str) -> list[dict[str, Any]]:
    """Get available data sources for a target world."""
    # TODO: Implement data source registry
    return []
