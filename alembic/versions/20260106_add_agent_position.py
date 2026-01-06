"""Add agent position and world tracking

Revision ID: 20260106_add_position
Revises: 20260105_add_unique_osm_id_indexes
Create Date: 2026-01-06

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = '20260106_add_position'
down_revision = 'add_unique_osm_id_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add position and world tracking to agents table
    op.add_column('agents', sa.Column('world_id', sa.String(50), nullable=True))
    op.add_column('agents', sa.Column('latitude', sa.Float(), nullable=True))
    op.add_column('agents', sa.Column('longitude', sa.Float(), nullable=True))
    op.add_column('agents', sa.Column('last_position_update', sa.DateTime(), nullable=True))
    
    # Add index for spatial queries (find agents near a location)
    op.create_index('idx_agents_position', 'agents', ['latitude', 'longitude'])
    op.create_index('idx_agents_world', 'agents', ['world_id'])


def downgrade() -> None:
    op.drop_index('idx_agents_world', table_name='agents')
    op.drop_index('idx_agents_position', table_name='agents')
    op.drop_column('agents', 'last_position_update')
    op.drop_column('agents', 'longitude')
    op.drop_column('agents', 'latitude')
    op.drop_column('agents', 'world_id')
