#!/usr/bin/env python3
"""
Transform OSM data from planet_osm_* tables to digital twin schema
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://earthlink:earthlink@db:5432/earthlink"

DDL_STATEMENTS = [
    # Buildings
    """
    CREATE TABLE IF NOT EXISTS buildings (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        building_type TEXT,
        height NUMERIC,
        min_height NUMERIC,
        levels INTEGER,
        footprint geometry(Polygon, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_buildings_osm_id ON buildings(osm_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_buildings_footprint ON buildings USING GIST (footprint);
    """,

    # Roads
    """
    CREATE TABLE IF NOT EXISTS roads (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        road_type TEXT,
        surface TEXT,
        lanes INTEGER,
        max_speed_kph INTEGER,
        oneway BOOLEAN,
        geom geometry(LineString, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_roads_osm_id ON roads(osm_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_roads_geom ON roads USING GIST (geom);
    """,

    # Places
    """
    CREATE TABLE IF NOT EXISTS places (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        place_type TEXT,
        population BIGINT,
        geom geometry(Point, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_places_osm_id ON places(osm_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_places_geom ON places USING GIST (geom);
    """,

    # Water features
    """
    CREATE TABLE IF NOT EXISTS water_features (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        water_type TEXT,
        geom geometry(Geometry, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_water_osm_id ON water_features(osm_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_water_geom ON water_features USING GIST (geom);
    """,

    # POIs
    """
    CREATE TABLE IF NOT EXISTS pois (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        poi_type TEXT,
        category TEXT,
        geom geometry(Point, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pois_osm_id ON pois(osm_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST (geom);
    """,

    # Land cover
    """
    CREATE TABLE IF NOT EXISTS land_cover (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        landuse_type TEXT,
        natural_type TEXT,
        geom geometry(Polygon, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_landcover_osm_id ON land_cover(osm_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_landcover_geom ON land_cover USING GIST (geom);
    """,

    # Boundaries
    """
    CREATE TABLE IF NOT EXISTS boundaries (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        boundary_type TEXT,
        admin_level INTEGER,
        geom geometry(Geometry, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_boundaries_osm_id ON boundaries(osm_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_boundaries_geom ON boundaries USING GIST (geom);
    """,
]


async def ensure_schema(engine):
    """Create tables and unique indexes if they are missing (committed separately)."""
    async with engine.begin() as conn:
        for stmt in DDL_STATEMENTS:
            await conn.execute(text(stmt))


async def transform_osm_data():
    """Transform OSM tables into digital twin schema, idempotently."""

    engine = create_async_engine(DATABASE_URL)

    # Ensure schema in its own transaction so DDL is persisted even if inserts fail
    await ensure_schema(engine)

    async with engine.begin() as conn:
        # Make sure tables and indexes exist before inserts (already done, but safe)
        for stmt in DDL_STATEMENTS:
            await conn.execute(text(stmt))

        print("=" * 80)
        print("TRANSFORMING OSM DATA TO DIGITAL TWIN SCHEMA")
        print("=" * 80)
        print()

        # Truncate all tables first for idempotent runs
        print("Truncating existing data...")
        await conn.execute(text("""
            TRUNCATE buildings, roads, places, water_features, pois, land_cover, boundaries;
        """))
        print("  ✓ Tables truncated")
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
            WHERE building IS NOT NULL;
        """))

        buildings_count = await conn.scalar(text("SELECT COUNT(*) FROM buildings"))
        print(f"  ✓ Upserted {buildings_count:,} buildings")
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
            WHERE highway IS NOT NULL;
        """))

        roads_count = await conn.scalar(text("SELECT COUNT(*) FROM roads"))
        print(f"  ✓ Upserted {roads_count:,} roads")
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
            WHERE place IS NOT NULL;
        """))

        places_count = await conn.scalar(text("SELECT COUNT(*) FROM places"))
        print(f"  ✓ Upserted {places_count:,} places")
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
            WHERE "natural" = 'water' OR waterway IS NOT NULL;
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
            WHERE waterway IS NOT NULL;
        """))

        water_count = await conn.scalar(text("SELECT COUNT(*) FROM water_features"))
        print(f"  ✓ Upserted {water_count:,} water features")
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
               OR leisure IS NOT NULL;
        """))

        pois_count = await conn.scalar(text("SELECT COUNT(*) FROM pois"))
        print(f"  ✓ Upserted {pois_count:,} POIs")
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
            WHERE landuse IS NOT NULL OR "natural" IN ('wood', 'forest', 'grassland', 'scrub');
        """))

        landcover_count = await conn.scalar(text("SELECT COUNT(*) FROM land_cover"))
        print(f"  ✓ Upserted {landcover_count:,} land cover areas")
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
            WHERE boundary IS NOT NULL;
        """))

        boundaries_count = await conn.scalar(text("SELECT COUNT(*) FROM boundaries"))
        print(f"  ✓ Upserted {boundaries_count:,} boundaries")
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
