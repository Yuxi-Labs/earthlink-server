"""Add OSM tables

Revision ID: 003_osm_tables
Revises: add_geo_tables
Create Date: 2025-12-27

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from geoalchemy2 import Geometry

revision = '003_osm_tables'
down_revision = 'add_geo_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Buildings
    op.execute("""
        CREATE TABLE IF NOT EXISTS buildings (
            id BIGSERIAL PRIMARY KEY,
            osm_id BIGINT,
            name TEXT,
            building_type TEXT,
            height NUMERIC,
            levels INTEGER,
            footprint GEOGRAPHY(POLYGON, 4326),
            properties JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_buildings_footprint ON buildings USING GIST(footprint);
        CREATE INDEX IF NOT EXISTS idx_buildings_osm_id ON buildings(osm_id);
    """)
    
    # Roads
    op.execute("""
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
        CREATE INDEX IF NOT EXISTS idx_roads_osm_id ON roads(osm_id);
    """)
    
    # Places
    op.execute("""
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
        CREATE INDEX IF NOT EXISTS idx_places_osm_id ON places(osm_id);
    """)
    
    # POIs
    op.execute("""
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
        CREATE INDEX IF NOT EXISTS idx_pois_osm_id ON pois(osm_id);
    """)
    
    # Water features
    op.execute("""
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
        CREATE INDEX IF NOT EXISTS idx_water_osm_id ON water_features(osm_id);
    """)
    
    # Land cover
    op.execute("""
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
        CREATE INDEX IF NOT EXISTS idx_landcover_osm_id ON land_cover(osm_id);
    """)
    
    # Boundaries
    op.execute("""
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
        CREATE INDEX IF NOT EXISTS idx_boundaries_osm_id ON boundaries(osm_id);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS buildings CASCADE;")
    op.execute("DROP TABLE IF EXISTS roads CASCADE;")
    op.execute("DROP TABLE IF EXISTS places CASCADE;")
    op.execute("DROP TABLE IF EXISTS pois CASCADE;")
    op.execute("DROP TABLE IF EXISTS water_features CASCADE;")
    op.execute("DROP TABLE IF EXISTS land_cover CASCADE;")
    op.execute("DROP TABLE IF EXISTS boundaries CASCADE;")
