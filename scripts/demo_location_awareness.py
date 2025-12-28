"""
Demo: Agent Location Awareness in Earthlink

Shows agents understanding where they are and querying their surroundings.
"""

import asyncio
from uuid import uuid4

import ray

from src.agents.core import Agent


async def demo_location_awareness():
    """Demonstrate agent location awareness in Earthlink."""
    
    print("=" * 80)
    print("EARTHLINK LOCATION AWARENESS DEMO")
    print("=" * 80)
    print()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create agent
    print("Creating agent...")
    agent_id = uuid4()
    agent_ref = Agent.remote(agent_id=agent_id, name="Explorer")
    
    print(f"Agent ID: {agent_id}")
    print()
    
    # Spawn in São Paulo
    print("-" * 80)
    print("1. SPAWNING IN SÃO PAULO")
    print("-" * 80)
    
    spawn_result = await agent_ref.earthlink_spawn_random_city.remote("south_america")
    print(f"Spawned at: {spawn_result['spawned_at']}")
    print(f"Region: {spawn_result['region']}")
    print()
    
    context = spawn_result["context"]
    print(f"Position: {context['position']['latitude']:.4f}, {context['position']['longitude']:.4f}")
    print(f"Description: {context['description']}")
    print()
    
    print("Administrative Context:")
    for level, value in context["administrative"].items():
        if value:
            print(f"  {level.capitalize()}: {value}")
    print()
    
    print("Nearby Buildings:")
    for building in context["nearby_buildings"]:
        print(f"  - {building['name']} ({building['type']}) - {building['distance_meters']:.1f}m away")
    print()
    
    print("Nearby Roads:")
    for road in context["nearby_roads"]:
        print(f"  - {road['name']} ({road['type']}) - {road['distance_meters']:.1f}m away")
    print()
    
    # Query nearby features
    print("-" * 80)
    print("2. QUERYING NEARBY FEATURES (500m radius)")
    print("-" * 80)
    
    nearby = await agent_ref.earthlink_query_nearby.remote(radius_meters=500, limit=5)
    
    print(f"Total features found: {nearby['total']}")
    print(f"Categories: {', '.join(nearby['by_category'].keys())}")
    print()
    
    # Show buildings
    if "building" in nearby["by_category"]:
        buildings = nearby["by_category"]["building"]
        print(f"Buildings ({len(buildings)}):")
        for b in buildings[:3]:
            print(f"  - {b['name']}")
            print(f"    Type: {b['type']}, Height: {b.get('height_meters', 0)}m")
            print(f"    Distance: {b['distance_meters']:.1f}m")
        print()
    
    # Get comprehensive context
    print("-" * 80)
    print("3. COMPREHENSIVE LOCATION CONTEXT (5km radius)")
    print("-" * 80)
    
    full_context = await agent_ref.earthlink_get_context.remote(radius_meters=5000)
    
    print(f"Features found:")
    for category, count in full_context.get("feature_counts", {}).items():
        print(f"  {category}: {count}")
    print()
    
    print(f"Containing regions ({len(full_context.get('containing_regions', []))}):")
    for region in full_context.get("containing_regions", [])[:5]:
        print(f"  - {region.get('name', 'Unknown')} ({region.get('type', 'unknown')})")
    print()
    
    # Check where agent is now
    print("-" * 80)
    print("4. WHERE AM I?")
    print("-" * 80)
    
    where = await agent_ref.earthlink_where_am_i.remote()
    print(f"I am in: {where['description']}")
    print(f"Coordinates: {where['position']['latitude']:.6f}, {where['position']['longitude']:.6f}")
    print()
    
    # Get agent properties
    lat = await agent_ref.latitude.remote
    lon = await agent_ref.longitude.remote
    print(f"Agent properties: lat={lat:.6f}, lon={lon:.6f}")
    print()
    
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Agents can now:")
    print("  ✓ Know where they are (lat/lon)")
    print("  ✓ Query surroundings via Earthlink API")
    print("  ✓ Understand context (city, neighborhood, buildings, roads)")
    print("  ✓ Spawn in real cities")
    print("  ✓ Access 43M+ Earth features")
    

if __name__ == "__main__":
    asyncio.run(demo_location_awareness())
