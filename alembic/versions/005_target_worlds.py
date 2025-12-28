"""Add target_worlds and explorer_readiness tables

Revision ID: 005_target_worlds
Revises: 003_osm_tables
Create Date: 2025-12-28 20:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision: str = '005_target_worlds'
down_revision: Union[str, None] = '003_osm_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create target_worlds and explorer_readiness tables."""
    
    # Target worlds registry - worlds agents can explore/deploy to
    op.create_table(
        'target_worlds',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('world_type', sa.String(50), nullable=False),  # earth_region, virtual_world, game_world, simulation
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('region_code', sa.String(10), nullable=True),  # ISO country/region code
        sa.Column('bounding_box', JSONB, nullable=True),  # {"min_lat": -90, "max_lat": 90, "min_lon": -180, "max_lon": 180}
        sa.Column('complexity_score', sa.Float, nullable=True),  # 0-1 scale
        sa.Column('feature_count', sa.Integer, default=0),  # Number of POIs/features
        sa.Column('required_knowledge_domains', JSONB, nullable=True),  # ["geography", "culture", "history"]
        sa.Column('deployment_status', sa.String(50), default='available'),  # available, testing, production, deprecated
        sa.Column('metadata', JSONB, default='{}'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Indexes for target_worlds
    op.create_index('idx_target_worlds_type', 'target_worlds', ['world_type'])
    op.create_index('idx_target_worlds_status', 'target_worlds', ['deployment_status'])
    op.create_index('idx_target_worlds_region', 'target_worlds', ['region_code'])
    
    # Explorer readiness metrics - track agent fitness for deployment
    op.create_table(
        'explorer_readiness',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('agent_id', UUID(as_uuid=True), nullable=False),
        sa.Column('target_world_id', UUID(as_uuid=True), nullable=True),  # NULL = general readiness
        sa.Column('readiness_score', sa.Float, nullable=False),  # 0-1 overall fitness
        
        # Component scores (0-1 each)
        sa.Column('knowledge_coverage', sa.Float, nullable=True),  # Domain knowledge breadth
        sa.Column('exploration_depth', sa.Float, nullable=True),  # How thorough agent explores
        sa.Column('spatial_competence', sa.Float, nullable=True),  # Navigation ability
        sa.Column('curiosity_level', sa.Float, nullable=True),  # Active learning drive
        sa.Column('collaboration_score', sa.Float, nullable=True),  # Multi-agent interaction
        sa.Column('goal_achievement', sa.Float, nullable=True),  # Goal completion rate
        
        # Training metrics
        sa.Column('total_steps', sa.Integer, default=0),
        sa.Column('worlds_explored', sa.Integer, default=0),
        sa.Column('knowledge_items_acquired', sa.Integer, default=0),
        sa.Column('unique_topics_explored', sa.Integer, default=0),
        sa.Column('distance_traveled_km', sa.Float, default=0.0),
        
        # Deployment readiness
        sa.Column('is_ready', sa.Boolean, default=False),  # Meets threshold
        sa.Column('readiness_threshold', sa.Float, default=0.7),  # Minimum score required
        sa.Column('last_evaluated_at', sa.TIMESTAMP, nullable=True),
        
        sa.Column('metadata', JSONB, default='{}'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Indexes for explorer_readiness
    op.create_index('idx_readiness_agent', 'explorer_readiness', ['agent_id'])
    op.create_index('idx_readiness_world', 'explorer_readiness', ['target_world_id'])
    op.create_index('idx_readiness_score', 'explorer_readiness', ['readiness_score'])
    op.create_index('idx_readiness_ready', 'explorer_readiness', ['is_ready'])
    
    # Foreign key constraints
    op.create_foreign_key(
        'fk_readiness_world',
        'explorer_readiness', 'target_worlds',
        ['target_world_id'], ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    """Drop target_worlds and explorer_readiness tables."""
    op.drop_table('explorer_readiness')
    op.drop_table('target_worlds')
