#!/usr/bin/env python3
"""
Full OSM reload script - idempotent, creates backup first if data exists.
Runs: ingest -> create tables -> transform -> verify -> backup
"""
import subprocess
import sys
import os
from pathlib import Path
import psycopg2

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "earthlink"),
    "password": os.getenv("DB_PASSWORD", "earthlink"),
    "dbname": os.getenv("DB_NAME", "earthlink")
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def run_sql(query: str):
    """Run SQL against the database."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()

def get_count(table: str) -> int:
    """Get row count for a table, returns 0 if table doesn't exist."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                return cur.fetchone()[0]
    except:
        return 0

def main():
    print("=" * 60)
    print("FULL OSM RELOAD")
    print("=" * 60)
    
    # Check if data already exists
    building_count = get_count("buildings")
    if building_count > 0:
        print(f"\n⚠️  Existing data found: {building_count:,} buildings")
        print("Creating backup first...")
        subprocess.run([sys.executable, "/app/scripts/backup_db.py"])
        print()
    
    # Step 1: Ingest OSM data
    print("\n[1/4] INGESTING OSM DATA (osm2pgsql)...")
    print("-" * 40)
    result = subprocess.run([sys.executable, "/app/scripts/ingest_osm_pbf.py"])
    if result.returncode != 0:
        print("❌ Ingestion failed!")
        sys.exit(1)
    
    # Step 2: Create tables
    print("\n[2/4] CREATING TABLES...")
    print("-" * 40)
    run_sql("""
        CREATE EXTENSION IF NOT EXISTS hstore;
        CREATE EXTENSION IF NOT EXISTS postgis;
        
        CREATE TABLE IF NOT EXISTS buildings (
            id BIGSERIAL PRIMARY KEY, osm_id BIGINT, name TEXT, building_type TEXT,
            height NUMERIC, min_height NUMERIC, levels INTEGER,
            footprint GEOGRAPHY(POLYGON, 4326), properties JSONB, created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_buildings_footprint ON buildings USING GIST(footprint);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_buildings_osm_id_unique ON buildings(osm_id);
        
        CREATE TABLE IF NOT EXISTS roads (
            id BIGSERIAL PRIMARY KEY, osm_id BIGINT, name TEXT, road_type TEXT,
            surface TEXT, lanes INTEGER, max_speed_kph INTEGER, oneway BOOLEAN,
            geom GEOGRAPHY(LINESTRING, 4326), properties JSONB, created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_roads_geom ON roads USING GIST(geom);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_roads_osm_id_unique ON roads(osm_id);
        
        CREATE TABLE IF NOT EXISTS places (
            id BIGSERIAL PRIMARY KEY, osm_id BIGINT, name TEXT, place_type TEXT,
            population INTEGER, geom GEOGRAPHY(POINT, 4326), properties JSONB, created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_places_geom ON places USING GIST(geom);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_places_osm_id_unique ON places(osm_id);
        
        CREATE TABLE IF NOT EXISTS pois (
            id BIGSERIAL PRIMARY KEY, osm_id BIGINT, name TEXT, poi_type TEXT,
            category TEXT, geom GEOGRAPHY(POINT, 4326), properties JSONB, created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_pois_geom ON pois USING GIST(geom);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pois_osm_id_unique ON pois(osm_id);
        
        CREATE TABLE IF NOT EXISTS water_features (
            id BIGSERIAL PRIMARY KEY, osm_id BIGINT, name TEXT, water_type TEXT,
            geom GEOGRAPHY(GEOMETRY, 4326), properties JSONB, created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_water_geom ON water_features USING GIST(geom);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_water_osm_id_unique ON water_features(osm_id);
        
        CREATE TABLE IF NOT EXISTS land_cover (
            id BIGSERIAL PRIMARY KEY, osm_id BIGINT, name TEXT, landuse_type TEXT,
            natural_type TEXT, geom GEOGRAPHY(GEOMETRY, 4326), properties JSONB, created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_landcover_geom ON land_cover USING GIST(geom);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_landcover_osm_id_unique ON land_cover(osm_id);
        
        CREATE TABLE IF NOT EXISTS boundaries (
            id BIGSERIAL PRIMARY KEY, osm_id BIGINT, name TEXT, boundary_type TEXT,
            admin_level TEXT, geom GEOGRAPHY(GEOMETRY, 4326), properties JSONB, created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_boundaries_geom ON boundaries USING GIST(geom);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_boundaries_osm_id_unique ON boundaries(osm_id);
    """)
    print("✓ Tables created")
    
    # Step 3: Transform data
    print("\n[3/4] TRANSFORMING OSM TO DIGITAL TWIN...")
    print("-" * 40)
    result = subprocess.run([sys.executable, "/app/scripts/transform_osm_to_digital_twin.py"])
    if result.returncode != 0:
        print("❌ Transform failed!")
        sys.exit(1)
    
    # Step 4: Verify
    print("\n[4/4] VERIFYING DATA...")
    print("-" * 40)
    tables = ["buildings", "roads", "places", "pois", "water_features", "land_cover", "boundaries"]
    total = 0
    for table in tables:
        count = get_count(table)
        total += count
        print(f"  {table}: {count:,}")
    print(f"\n  TOTAL: {total:,} features")
    
    if total == 0:
        print("\n❌ NO DATA LOADED!")
        sys.exit(1)
    
    # Create backup
    print("\n[BACKUP] Creating post-load backup...")
    print("-" * 40)
    subprocess.run([sys.executable, "/app/scripts/backup_db.py"])
    
    print("\n" + "=" * 60)
    print(f"✅ RELOAD COMPLETE: {total:,} features loaded and backed up")
    print("=" * 60)

if __name__ == "__main__":
    main()
