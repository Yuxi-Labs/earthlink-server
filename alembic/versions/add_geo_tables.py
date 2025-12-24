"""Add geospatial tables

Revision ID: add_geo_tables
Revises: add_timescale_hypertables
Create Date: 2025-12-24

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from geoalchemy2 import Geometry

# revision identifiers, used by Alembic.
revision = 'add_geo_tables'
down_revision = 'add_timescale_hypertables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create geo_features table for points of interest, landmarks
    op.create_table(
        'geo_features',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('feature_type', sa.String(100), nullable=False),  # landmark, poi, natural, infrastructure
        sa.Column('geom', Geometry('POINT', srid=4326), nullable=False),
        sa.Column('properties', postgresql.JSONB, nullable=True),  # Additional metadata
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.text('NOW()'), onupdate=sa.text('NOW()')),
    )
    
    # Create spatial index for fast queries
    op.execute('CREATE INDEX idx_geo_features_geom ON geo_features USING GIST (geom);')
    op.execute('CREATE INDEX idx_geo_features_type ON geo_features (feature_type);')
    
    # Create geo_regions table for countries, cities, boundaries
    op.create_table(
        'geo_regions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('region_type', sa.String(100), nullable=False),  # country, state, city, district
        sa.Column('geom', Geometry('POLYGON', srid=4326), nullable=False),
        sa.Column('properties', postgresql.JSONB, nullable=True),  # Population, area, etc.
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.text('NOW()'), onupdate=sa.text('NOW()')),
    )
    
    # Create spatial index for regions
    op.execute('CREATE INDEX idx_geo_regions_geom ON geo_regions USING GIST (geom);')
    op.execute('CREATE INDEX idx_geo_regions_type ON geo_regions (region_type);')
    op.execute('CREATE INDEX idx_geo_regions_name ON geo_regions (name);')
    
    # Create agent_trajectories table for movement history
    op.create_table(
        'agent_trajectories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP, nullable=False, server_default=sa.text('NOW()')),
        sa.Column('geom', Geometry('POINT', srid=4326), nullable=False),
        sa.Column('action', sa.String(100), nullable=True),  # What agent was doing
        sa.Column('metadata', postgresql.JSONB, nullable=True),  # Additional context
    )
    
    # Create indices for trajectories
    op.execute('CREATE INDEX idx_agent_trajectories_agent_id ON agent_trajectories (agent_id);')
    op.execute('CREATE INDEX idx_agent_trajectories_timestamp ON agent_trajectories (timestamp);')
    op.execute('CREATE INDEX idx_agent_trajectories_geom ON agent_trajectories USING GIST (geom);')
    
    # Create knowledge_acquisition_log table
    op.create_table(
        'knowledge_acquisition_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP, nullable=False, server_default=sa.text('NOW()')),
        sa.Column('source', sa.String(100), nullable=False),  # wikipedia, ollama, reddit, etc.
        sa.Column('topic', sa.String(500), nullable=True),
        sa.Column('content_summary', sa.Text, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('knowledge_count', sa.Integer, server_default='0'),
    )
    
    # Create indices for knowledge log
    op.execute('CREATE INDEX idx_knowledge_log_agent_id ON knowledge_acquisition_log (agent_id);')
    op.execute('CREATE INDEX idx_knowledge_log_timestamp ON knowledge_acquisition_log (timestamp);')
    op.execute('CREATE INDEX idx_knowledge_log_source ON knowledge_acquisition_log (source);')


def downgrade() -> None:
    op.drop_table('knowledge_acquisition_log')
    op.drop_table('agent_trajectories')
    op.drop_table('geo_regions')
    op.drop_table('geo_features')
