"""Voice transcript logging - stores voice interactions in database."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_session
from src.db.models import VoiceTranscript


class TranscriptLogger:
    """
    Logs voice transcripts to the database.
    
    Usage:
        logger = TranscriptLogger(session_id="abc123", agent_id=agent.id)
        
        # Log user speech
        await logger.log_user("What's the weather like?")
        
        # Log agent response
        await logger.log_agent("The weather is sunny today.")
        
        # Log with tool calls
        await logger.log_agent(
            "Moving to Sydney now.",
            tool_calls=[{"name": "agent_move", "arguments": {"lat": -33.8, "lon": 151.2}}]
        )
    """
    
    def __init__(
        self,
        session_id: str,
        agent_id: UUID | None = None,
    ):
        self.session_id = session_id
        self.agent_id = agent_id
    
    async def log_user(
        self,
        text: str,
        audio_duration_ms: int | None = None,
        audio_format: str | None = None,
    ) -> VoiceTranscript:
        """Log user speech transcript."""
        return await self._log(
            role="user",
            text=text,
            audio_duration_ms=audio_duration_ms,
            audio_format=audio_format,
        )
    
    async def log_agent(
        self,
        text: str,
        tool_calls: list[dict] | None = None,
        audio_duration_ms: int | None = None,
        audio_format: str | None = None,
    ) -> VoiceTranscript:
        """Log agent speech transcript."""
        return await self._log(
            role="agent",
            text=text,
            tool_calls=tool_calls,
            audio_duration_ms=audio_duration_ms,
            audio_format=audio_format,
        )
    
    async def log_system(self, text: str) -> VoiceTranscript:
        """Log system message (errors, events)."""
        return await self._log(role="system", text=text)
    
    async def _log(
        self,
        role: str,
        text: str,
        tool_calls: list[dict] | None = None,
        audio_duration_ms: int | None = None,
        audio_format: str | None = None,
    ) -> VoiceTranscript:
        """Write transcript entry to database."""
        async with get_session() as session:
            transcript = VoiceTranscript(
                id=uuid4(),
                session_id=self.session_id,
                agent_id=self.agent_id,
                role=role,
                text=text,
                tool_calls=tool_calls,
                audio_duration_ms=audio_duration_ms,
                audio_format=audio_format,
            )
            session.add(transcript)
            await session.commit()
            await session.refresh(transcript)
            return transcript
    
    @staticmethod
    async def get_session_transcripts(
        session_id: str,
        limit: int = 100,
    ) -> list[VoiceTranscript]:
        """Get all transcripts for a session."""
        async with get_session() as session:
            result = await session.execute(
                select(VoiceTranscript)
                .where(VoiceTranscript.session_id == session_id)
                .order_by(VoiceTranscript.created_at)
                .limit(limit)
            )
            return list(result.scalars().all())
    
    @staticmethod
    async def get_agent_transcripts(
        agent_id: UUID,
        limit: int = 100,
    ) -> list[VoiceTranscript]:
        """Get all transcripts for an agent."""
        async with get_session() as session:
            result = await session.execute(
                select(VoiceTranscript)
                .where(VoiceTranscript.agent_id == agent_id)
                .order_by(VoiceTranscript.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
    
    @staticmethod
    async def get_recent_transcripts(
        limit: int = 50,
        agent_id: UUID | None = None,
    ) -> list[VoiceTranscript]:
        """Get recent transcripts, optionally filtered by agent."""
        async with get_session() as session:
            query = select(VoiceTranscript).order_by(VoiceTranscript.created_at.desc())
            
            if agent_id:
                query = query.where(VoiceTranscript.agent_id == agent_id)
            
            result = await session.execute(query.limit(limit))
            return list(result.scalars().all())

