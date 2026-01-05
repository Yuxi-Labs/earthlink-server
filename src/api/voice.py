"""Voice API endpoints - WebSocket for real-time voice communication.

Connects browser/client audio to OpenAI Realtime API for speech-to-speech
conversation with agents.
"""

import asyncio
import json
import os
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel

from src.comms.voice import OpenAIVoiceSession, VoiceConfig
from src.comms.voice.openai import VoicePersonality
from src.comms.voice.transcript import TranscriptLogger


router = APIRouter()


class VoiceSessionInfo(BaseModel):
    """Voice session information."""
    available: bool
    voices: list[str]
    default_voice: str
    requires_api_key: bool


class TranscriptEntry(BaseModel):
    """Single transcript entry."""
    id: UUID
    session_id: str
    agent_id: UUID | None
    role: str
    text: str
    tool_calls: list[dict] | None
    created_at: datetime


class TranscriptResponse(BaseModel):
    """Transcript query response."""
    transcripts: list[TranscriptEntry]
    count: int


@router.get("/info")
async def voice_info() -> VoiceSessionInfo:
    """Get voice capability information."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    return VoiceSessionInfo(
        available=bool(api_key),
        voices=[v.value for v in VoicePersonality],
        default_voice=VoicePersonality.ALLOY.value,
        requires_api_key=not bool(api_key),
    )


@router.get("/transcripts")
async def get_transcripts(
    session_id: str | None = Query(None),
    agent_id: UUID | None = Query(None),
    limit: int = Query(50, le=200),
) -> TranscriptResponse:
    """
    Get voice transcripts.
    
    Filter by session_id, agent_id, or get recent transcripts.
    """
    if session_id:
        transcripts = await TranscriptLogger.get_session_transcripts(session_id, limit)
    elif agent_id:
        transcripts = await TranscriptLogger.get_agent_transcripts(agent_id, limit)
    else:
        transcripts = await TranscriptLogger.get_recent_transcripts(limit)
    
    return TranscriptResponse(
        transcripts=[
            TranscriptEntry(
                id=t.id,
                session_id=t.session_id,
                agent_id=t.agent_id,
                role=t.role,
                text=t.text,
                tool_calls=t.tool_calls,
                created_at=t.created_at,
            )
            for t in transcripts
        ],
        count=len(transcripts),
    )


@router.get("/transcripts/{session_id}")
async def get_session_transcript(session_id: str) -> TranscriptResponse:
    """Get all transcripts for a specific voice session."""
    transcripts = await TranscriptLogger.get_session_transcripts(session_id, limit=500)
    
    return TranscriptResponse(
        transcripts=[
            TranscriptEntry(
                id=t.id,
                session_id=t.session_id,
                agent_id=t.agent_id,
                role=t.role,
                text=t.text,
                tool_calls=t.tool_calls,
                created_at=t.created_at,
            )
            for t in transcripts
        ],
        count=len(transcripts),
    )


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    agent_id: UUID | None = Query(None),
    voice: str = Query("alloy"),
):
    """
    WebSocket endpoint for real-time voice communication.
    
    Protocol:
    - Client sends: Binary audio frames (PCM16, 24kHz, mono)
    - Server sends: JSON events OR binary audio frames
    
    JSON events:
    - {"type": "session_created", "session_id": "..."}
    - {"type": "transcript_input", "text": "..."}
    - {"type": "transcript_output", "text": "..."}
    - {"type": "tool_call", "name": "...", "arguments": {...}}
    - {"type": "error", "error": "..."}
    
    Binary frames are audio output (PCM16, 24kHz, mono).
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        await websocket.close(code=4001, reason="OPENAI_API_KEY not configured")
        return
    
    await websocket.accept()
    
    # Validate voice
    try:
        voice_personality = VoicePersonality(voice)
    except ValueError:
        voice_personality = VoicePersonality.ALLOY
    
    # Build instructions based on agent
    instructions = "You are a voice interface for Earthlink, an AI exploration system."
    if agent_id:
        instructions += f" You are speaking on behalf of agent {agent_id}."
    instructions += " Help the user interact with agents and explore the virtual world."
    
    config = VoiceConfig(
        api_key=api_key,
        voice=voice_personality,
        instructions=instructions,
    )
    
    # Create session ID and transcript logger
    session_id = str(uuid4())
    transcript_logger = TranscriptLogger(session_id=session_id, agent_id=agent_id)
    
    # Track tool calls for logging
    pending_tool_calls: list[dict] = []
    
    async def handle_tool_call(name: str, arguments: dict):
        """Handle tool calls from voice agent."""
        pending_tool_calls.append({"name": name, "arguments": arguments})
        
        # TODO: Wire to actual agent commands
        if name == "agent_move":
            return {"status": "moving", "to": arguments}
        elif name == "agent_explore":
            return {"status": "exploring", "topic": arguments.get("topic")}
        elif name == "agent_status":
            return {"status": "active", "agent_id": str(agent_id) if agent_id else None}
        return {"error": f"Unknown tool: {name}"}
    
    session = OpenAIVoiceSession(
        config=config,
        agent_id=agent_id,
        on_tool_call=handle_tool_call,
    )
    
    try:
        await session.connect()
        
        # Log session start
        await transcript_logger.log_system(f"Voice session started with voice={voice}")
        
        # Send session created
        await websocket.send_json({
            "type": "connected",
            "voice": voice,
            "session_id": session_id,
        })
        
        # Tasks for bidirectional streaming
        async def receive_from_client():
            """Receive audio from client WebSocket."""
            try:
                while True:
                    data = await websocket.receive()
                    
                    if "bytes" in data:
                        # Binary audio data
                        await session.send_audio(data["bytes"])
                    elif "text" in data:
                        # JSON command
                        msg = json.loads(data["text"])
                        if msg.get("type") == "commit":
                            await session.commit_audio()
                        elif msg.get("type") == "cancel":
                            await session.cancel_response()
                        elif msg.get("type") == "text":
                            text = msg.get("text", "")
                            await session.send_text(text)
                            # Log text input
                            await transcript_logger.log_user(text)
                            
            except WebSocketDisconnect:
                pass
        
        async def send_to_client():
            """Send events to client WebSocket."""
            nonlocal pending_tool_calls
            
            try:
                async for event in session.events():
                    if event.type == "audio":
                        # Send binary audio
                        await websocket.send_bytes(event.data["audio"])
                    else:
                        # Send JSON event
                        await websocket.send_json({
                            "type": event.type,
                            **event.data
                        })
                        
                        # Log transcripts
                        if event.type == "transcript_input":
                            await transcript_logger.log_user(event.data.get("text", ""))
                        elif event.type == "transcript_output":
                            # Include any pending tool calls
                            tool_calls = pending_tool_calls if pending_tool_calls else None
                            await transcript_logger.log_agent(
                                event.data.get("text", ""),
                                tool_calls=tool_calls,
                            )
                            pending_tool_calls = []
                            
            except WebSocketDisconnect:
                pass
        
        # Run both tasks
        receive_task = asyncio.create_task(receive_from_client())
        send_task = asyncio.create_task(send_to_client())
        
        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
            
    except Exception as e:
        await transcript_logger.log_system(f"Error: {str(e)}")
        await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        await transcript_logger.log_system("Voice session ended")
        await session.disconnect()

