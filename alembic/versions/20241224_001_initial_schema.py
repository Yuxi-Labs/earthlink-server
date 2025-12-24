"""Initial schema - agents, worlds, specializations, simulations

Revision ID: 001
Revises: 
Create Date: 2024-12-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Agents table
    op.create_table(
        'agents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.String(1000), nullable=True),
        sa.Column('state', sa.String(50), server_default='spawned'),
        sa.Column('knowledge_acquired', sa.Float, server_default='0.0'),
        sa.Column('worlds_explored', sa.Integer, server_default='0'),
        sa.Column('collaboration_score', sa.Float, server_default='0.0'),
        sa.Column('innovation_index', sa.Float, server_default='0.0'),
        sa.Column('config', postgresql.JSON, server_default='{}'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_agents_name', 'agents', ['name'])
    op.create_index('ix_agents_state', 'agents', ['state'])

    # Target worlds table
    op.create_table(
        'target_worlds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('description', sa.String(2000), nullable=True),
        sa.Column('data_sources', postgresql.JSON, server_default='[]'),
        sa.Column('integration_status', sa.String(50), server_default='pending'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('ix_target_worlds_category', 'target_worlds', ['category'])

    # Agent specializations table
    op.create_table(
        'agent_specializations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('target_world_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('target_worlds.id', ondelete='CASCADE'), nullable=False),
        sa.Column('state', sa.String(50), server_default='training'),
        sa.Column('readiness_score', sa.Float, server_default='0.0'),
        sa.Column('metrics', postgresql.JSON, server_default='{}'),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('deployed_at', sa.DateTime, nullable=True),
    )
    op.create_index('ix_agent_specializations_agent', 'agent_specializations', ['agent_id'])
    op.create_index('ix_agent_specializations_world', 'agent_specializations', ['target_world_id'])

    # Agent memories table
    op.create_table(
        'agent_memories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('memory_type', sa.String(50), nullable=False),
        sa.Column('content', postgresql.JSON, nullable=False),
        sa.Column('embedding_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('ix_agent_memories_agent', 'agent_memories', ['agent_id'])
    op.create_index('ix_agent_memories_type', 'agent_memories', ['memory_type'])

    # Interaction logs table
    op.create_table(
        'interaction_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('modality', sa.String(50), nullable=False),
        sa.Column('command_type', sa.String(50), nullable=False),
        sa.Column('payload', postgresql.JSON, server_default='{}'),
        sa.Column('response', postgresql.JSON, nullable=True),
        sa.Column('transcript', sa.String(10000), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('ix_interaction_logs_agent', 'interaction_logs', ['agent_id'])
    op.create_index('ix_interaction_logs_modality', 'interaction_logs', ['modality'])

    # Simulation runs table
    op.create_table(
        'simulation_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('config', postgresql.JSON, server_default='{}'),
        sa.Column('total_steps', sa.Integer, server_default='0'),
        sa.Column('total_agents', sa.Integer, server_default='0'),
        sa.Column('total_worlds', sa.Integer, server_default='0'),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('stopped_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_checkpoint', sa.String(500), nullable=True),
        sa.Column('checkpoint_step', sa.Integer, nullable=True),
    )
    op.create_index('ix_simulation_runs_status', 'simulation_runs', ['status'])

    # Metric snapshots table
    op.create_table(
        'metric_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('simulation_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('simulation_runs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('step', sa.Integer, nullable=True),
        sa.Column('recorded_at', sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index('ix_metric_snapshots_simulation', 'metric_snapshots', ['simulation_id'])
    op.create_index('ix_metric_snapshots_agent', 'metric_snapshots', ['agent_id'])
    op.create_index('ix_metric_snapshots_name', 'metric_snapshots', ['metric_name'])
    op.create_index('ix_metric_snapshots_recorded', 'metric_snapshots', ['recorded_at'])


def downgrade() -> None:
    op.drop_table('metric_snapshots')
    op.drop_table('simulation_runs')
    op.drop_table('interaction_logs')
    op.drop_table('agent_memories')
    op.drop_table('agent_specializations')
    op.drop_table('target_worlds')
    op.drop_table('agents')
