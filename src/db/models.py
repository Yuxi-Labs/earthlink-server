"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.database import Base


class Agent(Base):
    """Agent model - autonomous intelligent actor."""

    __tablename__ = "agents"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    
    # Lifecycle & Status (dual-dimension model per AGENT-LIFECYCLE.md)
    lifecycle: Mapped[str] = mapped_column(String(50), default="spawned")
    # Lifecycle (developmental stage): spawned, oriented, explores, learns, adapts, differentiates, acts, transforms, expires, archived
    status: Mapped[str] = mapped_column(String(50), default="idle")
    # Status (operational state): idle, exploring, learning, interacting, executing, adapting, overloaded, corrupted, retired
    
    # Legacy field - kept for backward compatibility, can be removed after migration
    state: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Metrics
    knowledge_acquired: Mapped[float] = mapped_column(Float, default=0.0)
    worlds_explored: Mapped[int] = mapped_column(default=0)
    collaboration_score: Mapped[float] = mapped_column(Float, default=0.0)
    innovation_index: Mapped[float] = mapped_column(Float, default=0.0)

    # Configuration
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    specializations: Mapped[list["AgentSpecialization"]] = relationship(
        back_populates="agent", cascade="all, delete-orphan"
    )


class TargetWorld(Base):
    """Target world model - worlds agents can explore."""

    __tablename__ = "target_worlds"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    # Categories: physical, virtual, game, abstract
    description: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    data_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    integration_status: Mapped[str] = mapped_column(String(50), default="pending")
    # Status: pending, available, active, deprecated

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    specializations: Mapped[list["AgentSpecialization"]] = relationship(
        back_populates="target_world"
    )


class AgentSpecialization(Base):
    """Agent specialization for a target world."""

    __tablename__ = "agent_specializations"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=False
    )
    target_world_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("target_worlds.id"), nullable=False
    )
    state: Mapped[str] = mapped_column(String(50), default="training")
    # States: training, specializing, testing, deployed

    # Readiness metrics
    readiness_score: Mapped[float] = mapped_column(Float, default=0.0)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    agent: Mapped["Agent"] = relationship(back_populates="specializations")
    target_world: Mapped["TargetWorld"] = relationship(back_populates="specializations")


class AgentMemory(Base):
    """Agent memory entries."""

    __tablename__ = "agent_memories"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=False
    )
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Types: episodic, semantic, procedural
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    embedding_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Reference to vector DB

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class InteractionLog(Base):
    """Log of all agent interactions (commands, voice, text)."""

    __tablename__ = "interaction_logs"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=False
    )
    modality: Mapped[str] = mapped_column(String(50), nullable=False)
    # Modalities: api, text, terminal, voice
    command_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    transcript: Mapped[str | None] = mapped_column(String(10000), nullable=True)
    # For voice interactions

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SimulationRun(Base):
    """Simulation run tracking."""

    __tablename__ = "simulation_runs"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    # Status: pending, running, paused, stopped, completed, failed

    # Configuration
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Metrics
    total_steps: Mapped[int] = mapped_column(default=0)
    total_agents: Mapped[int] = mapped_column(default=0)
    total_worlds: Mapped[int] = mapped_column(default=0)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Checkpoint info
    last_checkpoint: Mapped[str | None] = mapped_column(String(500), nullable=True)
    checkpoint_step: Mapped[int | None] = mapped_column(nullable=True)


class VoiceTranscript(Base):
    """Voice interaction transcripts."""

    __tablename__ = "voice_transcripts"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    agent_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=True
    )
    
    # Speaker
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    # Roles: user, agent, system
    
    # Content
    text: Mapped[str] = mapped_column(String(10000), nullable=False)
    
    # Audio metadata (optional)
    audio_duration_ms: Mapped[int | None] = mapped_column(nullable=True)
    audio_format: Mapped[str | None] = mapped_column(String(20), nullable=True)
    
    # Tool calls (if any)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MetricSnapshot(Base):
    """Time-series metrics snapshots."""

    __tablename__ = "metric_snapshots"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    simulation_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("simulation_runs.id"), nullable=True
    )
    agent_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("agents.id"), nullable=True
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    step: Mapped[int | None] = mapped_column(nullable=True)

    recorded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
