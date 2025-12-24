"""Alembic migration: Add TimescaleDB hypertables for metrics.

Revision ID: add_timescale_hypertables
Revises: 
Create Date: 2025-12-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_timescale_hypertables'
down_revision: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    # Create new time-series tables with proper structure
    
    # Agent metrics time-series
    op.create_table(
        'agent_metrics_ts',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('agent_id', sa.UUID(), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('time', 'agent_id', 'metric_name'),
    )
    op.create_index('idx_agent_metrics_ts_agent_id', 'agent_metrics_ts', ['agent_id'])
    op.create_index('idx_agent_metrics_ts_metric_name', 'agent_metrics_ts', ['metric_name'])
    
    # Convert to hypertable
    op.execute("SELECT create_hypertable('agent_metrics_ts', 'time', if_not_exists => TRUE)")
    
    # Simulation metrics time-series
    op.create_table(
        'simulation_metrics_ts',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('simulation_id', sa.UUID(), nullable=True),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('step', sa.Integer(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('time', 'metric_name'),
    )
    op.create_index('idx_simulation_metrics_ts_sim_id', 'simulation_metrics_ts', ['simulation_id'])
    op.create_index('idx_simulation_metrics_ts_metric_name', 'simulation_metrics_ts', ['metric_name'])
    
    # Convert to hypertable
    op.execute("SELECT create_hypertable('simulation_metrics_ts', 'time', if_not_exists => TRUE)")
    
    # World metrics time-series
    op.create_table(
        'world_metrics_ts',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('world_id', sa.UUID(), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('time', 'world_id', 'metric_name'),
    )
    op.create_index('idx_world_metrics_ts_world_id', 'world_metrics_ts', ['world_id'])
    op.create_index('idx_world_metrics_ts_metric_name', 'world_metrics_ts', ['metric_name'])
    
    # Convert to hypertable
    op.execute("SELECT create_hypertable('world_metrics_ts', 'time', if_not_exists => TRUE)")
    
    # Readiness metrics time-series (from plan.md)
    op.create_table(
        'readiness_metrics_ts',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('agent_id', sa.UUID(), nullable=False),
        sa.Column('target_world_id', sa.UUID(), nullable=True),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('time', 'agent_id', 'target_world_id', 'metric_name'),
    )
    op.create_index('idx_readiness_metrics_ts_agent_id', 'readiness_metrics_ts', ['agent_id'])
    op.create_index('idx_readiness_metrics_ts_world_id', 'readiness_metrics_ts', ['target_world_id'])
    
    # Convert to hypertable
    op.execute("SELECT create_hypertable('readiness_metrics_ts', 'time', if_not_exists => TRUE)")
    
    # Set retention policy (keep 90 days)
    op.execute("SELECT add_retention_policy('agent_metrics_ts', INTERVAL '90 days', if_not_exists => TRUE)")
    op.execute("SELECT add_retention_policy('simulation_metrics_ts', INTERVAL '90 days', if_not_exists => TRUE)")
    op.execute("SELECT add_retention_policy('world_metrics_ts', INTERVAL '90 days', if_not_exists => TRUE)")
    op.execute("SELECT add_retention_policy('readiness_metrics_ts', INTERVAL '90 days', if_not_exists => TRUE)")


def downgrade() -> None:
    op.drop_table('readiness_metrics_ts')
    op.drop_table('world_metrics_ts')
    op.drop_table('simulation_metrics_ts')
    op.drop_table('agent_metrics_ts')
