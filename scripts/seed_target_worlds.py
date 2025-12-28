"""Seed sample target worlds for Explorer Pipeline testing."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.database import async_session_maker
from sqlalchemy import text


async def seed_target_worlds():
    """Create sample target worlds for agent deployment."""
    
    worlds = [
        {
            "name": "Sydney Metropolitan Area",
            "world_type": "earth_region",
            "description": "Greater Sydney region with beaches, parks, and urban areas",
            "region_code": "AU-NSW",
            "bounding_box": {
                "min_lat": -34.0,
                "max_lat": -33.5,
                "min_lon": 150.5,
                "max_lon": 151.5
            },
            "complexity_score": 0.85,
            "feature_count": 15000,
            "required_knowledge_domains": ["geography", "culture", "urban planning"],
            "deployment_status": "active",
            "metadata": {
                "population": 5312000,
                "area_km2": 12368,
                "notable_features": ["Sydney Opera House", "Harbour Bridge", "Bondi Beach"]
            }
        },
        {
            "name": "Melbourne CBD",
            "world_type": "earth_region",
            "description": "Melbourne central business district and inner suburbs",
            "region_code": "AU-VIC",
            "bounding_box": {
                "min_lat": -37.85,
                "max_lat": -37.75,
                "min_lon": 144.90,
                "max_lon": 145.05
            },
            "complexity_score": 0.80,
            "feature_count": 12000,
            "required_knowledge_domains": ["geography", "culture", "arts"],
            "deployment_status": "active",
            "metadata": {
                "population": 178955,
                "area_km2": 37.7,
                "notable_features": ["Federation Square", "Queen Victoria Market", "Southbank"]
            }
        },
        {
            "name": "Brisbane River District",
            "world_type": "earth_region",
            "description": "Brisbane city and river precincts",
            "region_code": "AU-QLD",
            "bounding_box": {
                "min_lat": -27.50,
                "max_lat": -27.40,
                "min_lon": 152.95,
                "max_lon": 153.10
            },
            "complexity_score": 0.70,
            "feature_count": 8000,
            "required_knowledge_domains": ["geography", "climate", "tourism"],
            "deployment_status": "active",
            "metadata": {
                "population": 2582000,
                "area_km2": 15826,
                "notable_features": ["Story Bridge", "South Bank", "Brisbane River"]
            }
        },
        {
            "name": "Virtual Test Environment Alpha",
            "world_type": "virtual_world",
            "description": "Simulated 3D environment for controlled agent testing",
            "region_code": None,
            "bounding_box": None,
            "complexity_score": 0.50,
            "feature_count": 500,
            "required_knowledge_domains": ["navigation", "object recognition"],
            "deployment_status": "pending",
            "metadata": {
                "environment_type": "3D simulation",
                "physics_enabled": True,
                "max_agents": 100
            }
        },
        {
            "name": "Minecraft World - Creative Mode",
            "world_type": "game_world",
            "description": "Open-ended Minecraft creative world for exploration agents",
            "region_code": None,
            "bounding_box": None,
            "complexity_score": 0.90,
            "feature_count": None,
            "required_knowledge_domains": ["block manipulation", "spatial reasoning", "creativity"],
            "deployment_status": "pending",
            "metadata": {
                "game": "Minecraft",
                "mode": "creative",
                "seed": "earthlink_test_001"
            }
        },
    ]
    
    async with async_session_maker() as db:
        for world in worlds:
            # Check if already exists
            check_query = text("""
                SELECT id FROM target_worlds WHERE name = :name
            """)
            result = await db.execute(check_query, {"name": world["name"]})
            existing = result.fetchone()
            
            if existing:
                print(f"✓ World '{world['name']}' already exists")
                continue
            
            # Insert new world
            insert_query = text("""
                INSERT INTO target_worlds (
                    name, world_type, description, region_code,
                    bounding_box, complexity_score, feature_count,
                    required_knowledge_domains, deployment_status, metadata
                ) VALUES (
                    :name, :world_type, :description, :region_code,
                    CAST(:bounding_box AS jsonb), :complexity_score, :feature_count,
                    CAST(:required_knowledge_domains AS jsonb), :deployment_status, CAST(:metadata AS jsonb)
                )
                RETURNING id, name
            """)
            
            # Convert dicts to JSON strings for JSONB columns
            params = {
                **world,
                "bounding_box": json.dumps(world["bounding_box"]) if world["bounding_box"] else None,
                "required_knowledge_domains": json.dumps(world["required_knowledge_domains"]) if world["required_knowledge_domains"] else None,
                "metadata": json.dumps(world["metadata"]) if world["metadata"] else None,
            }
            
            result = await db.execute(insert_query, params)
            row = result.fetchone()
            
            print(f"✓ Created world '{row.name}' ({row.id})")
        
        await db.commit()
        print(f"\n✓ Seeded {len(worlds)} target worlds successfully")


if __name__ == "__main__":
    asyncio.run(seed_target_worlds())
