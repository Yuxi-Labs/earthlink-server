#!/usr/bin/env python3
"""
Ingest OSM PBF data using osm2pgsql
"""

import subprocess
import sys
from pathlib import Path

def run_osm2pgsql(pbf_file: Path):
    """Run osm2pgsql to import PBF data"""
    
    if not pbf_file.exists():
        print(f"❌ PBF file not found: {pbf_file}")
        sys.exit(1)
    
    print("=" * 80)
    print("INGESTING OSM DATA WITH osm2pgsql")
    print("=" * 80)
    print(f"Input: {pbf_file}")
    print(f"Size: {pbf_file.stat().st_size / (1024**3):.2f} GB")
    print("=" * 80)
    print()
    
    # Database connection
    db_url = "postgresql://earthlink:earthlink@db:5432/earthlink"
    
    # osm2pgsql command - uses standard PostgreSQL env vars
    # Note: Using pgsql output (deprecated but works). Future: migrate to flex output
    # See: https://osm2pgsql.org/doc/manual.html#the-flex-output
    cmd = [
        "osm2pgsql",
        "--create",  # Create tables (drop if exist)
        "--output", "pgsql",  # Explicit pgsql output (deprecated, but we transform later)
        "--database", "earthlink",
        "--host", "db",
        "--port", "5432",
        "--slim",  # Use slim mode for updates
        "--drop",  # Drop existing tables
        "--hstore",  # Add tags as hstore
        "--latlong",  # Use lat/long (EPSG:4326) instead of Web Mercator
        "--number-processes", "4",  # Parallel processing
        "--cache", "2000",  # 2GB cache
        str(pbf_file)
    ]
    
    print("Running osm2pgsql...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Set PostgreSQL env vars
    env = {
        "PGUSER": "earthlink",
        "PGPASSWORD": "earthlink"
    }
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            text=True,
            capture_output=False
        )
        
        print()
        print("=" * 80)
        print("✓ INGESTION COMPLETE")
        print("=" * 80)
        print()
        print("Created tables:")
        print("  - planet_osm_point   (POIs, places)")
        print("  - planet_osm_line    (roads, railways, waterways)")
        print("  - planet_osm_polygon (buildings, land use, water)")
        print("  - planet_osm_roads   (road network)")
        print()
        print("Next step: Transform to digital twin schema")
        print("=" * 80)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ osm2pgsql failed: {e}")
        sys.exit(1)

def main():
    # PBF file location - matches download location
    pbf_file = Path("/tmp/geofabrik/australia-oceania-latest.osm.pbf")
    
    run_osm2pgsql(pbf_file)

if __name__ == "__main__":
    main()
