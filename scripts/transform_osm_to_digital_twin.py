#!/usr/bin/env python3
"""
Transform OSM data from planet_osm_* tables to digital twin schema
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://earthlink:earthlink@db:5432/earthlink"

async def transform_osm_data():
    """Transform OSM tables into digital twin schema"""
    
    engine = create_async_engine(DATABASE_URL)
    
    async with engine.begin() as conn:
        print("=" * 80)
        print("TRANSFORMING OSM DATA TO DIGITAL TWIN SCHEMA")
        print("=" * 80)
        print()
        
        # 1. Transform buildings
        print("Transforming buildings...")
        await conn.execute(text("""
            INSERT INTO buildings (
                osm_id,
                name, 
                building_type,
                height,
                min_height,
                levels,
                footprint,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                COALESCE(LEFT(building, 50), 'yes') as building_type,
                COALESCE(
                    CASE WHEN tags->'height' ~ '^[0-9.]+$' 
                         THEN (tags->'height')::float
                         ELSE NULL
                    END,
                    CASE 
                        WHEN tags->'building:levels' ~ '^[0-9]+$' 
                        THEN (tags->'building:levels')::int * 3.5
                        ELSE NULL 
                    END
                ) as height,
                COALESCE(
                    CASE WHEN tags->'min_height' ~ '^[0-9.]+$' 
                         THEN (tags->'min_height')::float
                         ELSE NULL
                    END, 
                    0
                ) as min_height,
                CASE WHEN tags->'building:levels' ~ '^[0-9]+$' 
                     THEN (tags->'building:levels')::int
                     ELSE NULL
                END as levels,
                way::geometry(Polygon, 4326) as footprint,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_polygon
            WHERE building IS NOT NULL
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        buildings_count = await conn.scalar(text("SELECT COUNT(*) FROM buildings"))
        print(f"  ✓ Inserted {buildings_count:,} buildings")
        print()
        
        # 2. Transform roads
        print("Transforming roads...")
        await conn.execute(text("""
            INSERT INTO roads (
                osm_id,
                name,
                road_type,
                surface,
                lanes,
                max_speed_kph,
                oneway,
                geom,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                COALESCE(LEFT(highway, 50), 'unclassified') as road_type,
                LEFT(surface, 50) as surface,
                COALESCE(
                    CASE WHEN tags->'lanes' ~ '^[0-9]+$' 
                         THEN (tags->'lanes')::int
                         ELSE NULL
                    END, 
                    1
                ) as lanes,
                CASE WHEN tags->'maxspeed' ~ '^[0-9]+$' 
                     THEN (tags->'maxspeed')::int
                     ELSE NULL
                END as max_speed_kph,
                COALESCE(oneway = 'yes', false) as oneway,
                ST_Force2D(way)::geometry(LineString, 4326) as geom,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_line
            WHERE highway IS NOT NULL
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        roads_count = await conn.scalar(text("SELECT COUNT(*) FROM roads"))
        print(f"  ✓ Inserted {roads_count:,} roads")
        print()
        
        # 3. Transform places (POIs)
        print("Transforming places...")
        await conn.execute(text("""
            INSERT INTO places (
                osm_id,
                name,
                place_type,
                population,
                geom,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                COALESCE(LEFT(place, 50), 'locality') as place_type,
                CASE WHEN tags->'population' ~ '^[0-9]+$' 
                     THEN (tags->'population')::bigint
                     ELSE NULL
                END as population,
                ST_Force2D(way)::geometry(Point, 4326) as geom,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_point
            WHERE place IS NOT NULL
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        places_count = await conn.scalar(text("SELECT COUNT(*) FROM places"))
        print(f"  ✓ Inserted {places_count:,} places")
        print()
        
        # 4. Transform water features
        print("Transforming water features...")
        await conn.execute(text("""
            INSERT INTO water_features (
                osm_id,
                name,
                water_type,
                geom,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                LEFT(COALESCE(
                    CASE 
                        WHEN waterway IS NOT NULL THEN waterway
                        WHEN "natural" = 'water' THEN COALESCE(water, 'water')
                        ELSE 'water'
                    END,
                    'water'
                ), 100) as water_type,
                ST_Force2D(way)::geometry(Geometry, 4326) as geom,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_polygon
            WHERE "natural" = 'water' OR waterway IS NOT NULL
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        await conn.execute(text("""
            INSERT INTO water_features (
                osm_id,
                name,
                water_type,
                geom,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                LEFT(COALESCE(waterway, 'stream'), 100) as water_type,
                ST_Force2D(way)::geometry(Geometry, 4326) as geom,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_line
            WHERE waterway IS NOT NULL
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        water_count = await conn.scalar(text("SELECT COUNT(*) FROM water_features"))
        print(f"  ✓ Inserted {water_count:,} water features")
        print()
        
        # 5. Transform POIs
        print("Transforming POIs...")
        await conn.execute(text("""
            INSERT INTO pois (
                osm_id,
                name,
                poi_type,
                category,
                geom,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                LEFT(COALESCE(
                    amenity, shop, tourism, leisure, 'unknown'
                ), 100) as poi_type,
                CASE 
                    WHEN amenity IS NOT NULL THEN 'amenity'
                    WHEN shop IS NOT NULL THEN 'shop'
                    WHEN tourism IS NOT NULL THEN 'tourism'
                    WHEN leisure IS NOT NULL THEN 'leisure'
                    ELSE 'other'
                END as category,
                ST_Force2D(way)::geometry(Point, 4326) as geom,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_point
            WHERE amenity IS NOT NULL 
               OR shop IS NOT NULL 
               OR tourism IS NOT NULL 
               OR leisure IS NOT NULL
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        pois_count = await conn.scalar(text("SELECT COUNT(*) FROM pois"))
        print(f"  ✓ Inserted {pois_count:,} POIs")
        print()
        
        # 6. Transform land cover
        print("Transforming land cover...")
        await conn.execute(text("""
            INSERT INTO land_cover (
                osm_id,
                name,
                landuse_type,
                natural_type,
                geom,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                LEFT(landuse, 50) as landuse_type,
                LEFT("natural", 50) as natural_type,
                ST_Force2D(way)::geometry(Polygon, 4326) as geom,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_polygon
            WHERE landuse IS NOT NULL OR "natural" IN ('wood', 'forest', 'grassland', 'scrub')
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        landcover_count = await conn.scalar(text("SELECT COUNT(*) FROM land_cover"))
        print(f"  ✓ Inserted {landcover_count:,} land cover areas")
        print()
        
        # 7. Transform boundaries
        print("Transforming boundaries...")
        await conn.execute(text("""
            INSERT INTO boundaries (
                osm_id,
                name,
                boundary_type,
                admin_level,
                geom,
                properties
            )
            SELECT 
                osm_id,
                COALESCE(LEFT(name, 255), 'Unnamed') as name,
                LEFT(COALESCE(boundary, 'administrative'), 100) as boundary_type,
                CASE WHEN tags->'admin_level' ~ '^[0-9]+$' 
                     THEN (tags->'admin_level')::int
                     ELSE NULL
                END as admin_level,
                ST_Force2D(way)::geometry(Geometry, 4326) as geom,
                hstore_to_jsonb(tags) as properties
            FROM planet_osm_polygon
            WHERE boundary IS NOT NULL
            ON CONFLICT (osm_id) DO NOTHING;
        """))
        
        boundaries_count = await conn.scalar(text("SELECT COUNT(*) FROM boundaries"))
        print(f"  ✓ Inserted {boundaries_count:,} boundaries")
        print()
        
        print("=" * 80)
        print("✓ TRANSFORMATION COMPLETE")
        print("=" * 80)
        print()
        print("Digital Twin Schema Summary:")
        print(f"  - Buildings: {buildings_count:,}")
        print(f"  - Roads: {roads_count:,}")
        print(f"  - Places: {places_count:,}")
        print(f"  - Water Features: {water_count:,}")
        print(f"  - POIs: {pois_count:,}")
        print(f"  - Land Cover: {landcover_count:,}")
        print(f"  - Boundaries: {boundaries_count:,}")
        print("=" * 80)
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(transform_osm_data())
