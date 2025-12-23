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
    state: Mapped[str] = mapped_column(String(50), default="spawned")
    # States: spawned, training, specializing, testing, deployed

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
