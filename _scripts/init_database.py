"""Database initialization script - automated setup for new deployments."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from src.config import settings
from src.db.models import Base


async def create_postgis_extensions(engine):
    """Create required PostgreSQL extensions."""
    print("Creating PostgreSQL extensions...")
    
    async with engine.begin() as conn:
        # PostGIS
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology"))
        
        # Additional extensions
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS hstore"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))  # Text search
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS btree_gist"))
        
        # TimescaleDB (if available)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            print("  ✓ TimescaleDB extension created")
        except Exception as e:
            print(f"  ⚠ TimescaleDB not available: {e}")
    
    print("  ✓ Extensions created")


async def create_tables(engine):
    """Create all database tables."""
    print("Creating database tables...")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("  ✓ Tables created")


async def create_geo_tables(engine):
    """Create OpenStreetMap geo data tables."""
    print("Creating geo data tables...")
    
    geo_tables_sql = """
    -- Buildings
    CREATE TABLE IF NOT EXISTS buildings (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        building_type TEXT,
        height NUMERIC,
        min_height NUMERIC,
        levels INTEGER,
        footprint GEOGRAPHY(POLYGON, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_buildings_footprint ON buildings USING GIST(footprint);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_buildings_osm_id_unique ON buildings(osm_id);
    
    -- Roads
    CREATE TABLE IF NOT EXISTS roads (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        road_type TEXT,
        surface TEXT,
        lanes INTEGER,
        max_speed_kph INTEGER,
        oneway BOOLEAN,
        geom GEOGRAPHY(LINESTRING, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_roads_geom ON roads USING GIST(geom);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_roads_osm_id_unique ON roads(osm_id);
    
    -- Places
    CREATE TABLE IF NOT EXISTS places (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        place_type TEXT,
        population INTEGER,
        geom GEOGRAPHY(POINT, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_places_geom ON places USING GIST(geom);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_places_osm_id_unique ON places(osm_id);
    
    -- Points of Interest
    CREATE TABLE IF NOT EXISTS pois (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        poi_type TEXT,
        category TEXT,
        geom GEOGRAPHY(POINT, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST(geom);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pois_osm_id_unique ON pois(osm_id);
    
    -- Water Features
    CREATE TABLE IF NOT EXISTS water_features (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        water_type TEXT,
        geom GEOGRAPHY(GEOMETRY, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_water_geom ON water_features USING GIST(geom);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_water_osm_id_unique ON water_features(osm_id);
    
    -- Land Cover
    CREATE TABLE IF NOT EXISTS land_cover (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        landuse_type TEXT,
        natural_type TEXT,
        geom GEOGRAPHY(GEOMETRY, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_landcover_geom ON land_cover USING GIST(geom);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_landcover_osm_id_unique ON land_cover(osm_id);
    
    -- Boundaries
    CREATE TABLE IF NOT EXISTS boundaries (
        id BIGSERIAL PRIMARY KEY,
        osm_id BIGINT,
        name TEXT,
        boundary_type TEXT,
        admin_level TEXT,
        geom GEOGRAPHY(GEOMETRY, 4326),
        properties JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_boundaries_geom ON boundaries USING GIST(geom);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_boundaries_osm_id_unique ON boundaries(osm_id);
    """
    
    async with engine.begin() as conn:
        for statement in geo_tables_sql.split(";"):
            statement = statement.strip()
            if statement:
                await conn.execute(text(statement))
    
    print("  ✓ Geo tables created")


async def run_migrations():
    """Run Alembic migrations."""
    print("Running database migrations...")
    
    import subprocess
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"  ⚠ Migration failed: {result.stderr}")
        return False
    
    print("  ✓ Migrations applied")
    return True


async def seed_target_worlds(engine):
    """Seed initial target worlds."""
    print("Seeding target worlds...")
    
    seed_sql = """
    INSERT INTO target_worlds (name, description, location_type, center_lat, center_lon, radius_km, metadata, is_active)
    VALUES
        ('Sydney', 'Sydney, Australia - Harbor city with Opera House', 'city', -33.8688, 151.2093, 25.0, 
         '{"country": "Australia", "population": 5312000, "features": ["harbor", "beaches", "urban"]}', true),
        ('Melbourne', 'Melbourne, Australia - Cultural capital', 'city', -37.8136, 144.9631, 25.0,
         '{"country": "Australia", "population": 5078000, "features": ["cultural", "urban", "sports"]}', true),
        ('Brisbane', 'Brisbane, Australia - River city', 'city', -27.4698, 153.0251, 20.0,
         '{"country": "Australia", "population": 2560000, "features": ["river", "subtropical", "urban"]}', true),
        ('Virtual Test', 'Virtual testing environment', 'virtual', 0.0, 0.0, 10.0,
         '{"type": "test", "features": ["synthetic", "controlled"]}', false),
        ('Minecraft World', 'Procedurally generated Minecraft-style world', 'procedural', 0.0, 0.0, 100.0,
         '{"type": "procedural", "seed": 12345, "biomes": ["plains", "forest", "mountains"]}', false)
    ON CONFLICT (name) DO NOTHING;
    """
    
    async with engine.begin() as conn:
        await conn.execute(text(seed_sql))
    
    print("  ✓ Target worlds seeded")


async def initialize_database():
    """Main initialization function."""
    print("=" * 60)
    print("EARTHLINK Database Initialization")
    print("=" * 60)
    
    # Create engine
    engine = create_async_engine(
        settings.async_database_url,
        echo=False,
    )
    
    try:
        # Step 1: Extensions
        await create_postgis_extensions(engine)
        
        # Step 2: Core tables via SQLAlchemy
        await create_tables(engine)
        
        # Step 3: Run Alembic migrations
        # await run_migrations()  # Commented out - run manually if needed
        
        # Step 4: Geo tables
        await create_geo_tables(engine)
        
        # Step 5: Seed data
        await seed_target_worlds(engine)
        
        print("\n" + "=" * 60)
        print("✓ Database initialization complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Ingest OSM data: python scripts/ingest_osm_pbf.py")
        print("  2. Transform to digital twin: python scripts/transform_osm_to_digital_twin.py")
        print("  3. Start the API server: uvicorn src.main:app --reload")
        print()
        
    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        raise
    
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(initialize_database())
