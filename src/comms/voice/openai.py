"""OpenAI Realtime API voice integration.

Real-time speech-to-speech conversation with agents using OpenAI's Realtime API.
See: https://platform.openai.com/docs/guides/voice-agents

Key features:
- WebSocket-based real-time audio streaming
- Speech-to-speech (no transcription round-trip)
- Function/tool calling for agent actions
- Interruption handling
- Multiple voice options
"""

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID

import websockets
from websockets.client import WebSocketClientProtocol


class VoicePersonality(str, Enum):
    """Available OpenAI voice personalities."""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    PCM16 = "pcm16"  # 24kHz, 16-bit PCM, mono
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"


@dataclass
class VoiceConfig:
    """Configuration for voice session."""
    
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4o-realtime-preview"
    voice: VoicePersonality = VoicePersonality.ALLOY
    input_audio_format: AudioFormat = AudioFormat.PCM16
    output_audio_format: AudioFormat = AudioFormat.PCM16
    instructions: str = ""
    temperature: float = 0.8
    max_response_output_tokens: int | str = "inf"
    turn_detection: dict[str, Any] = field(default_factory=lambda: {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
    })
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VoiceEvent:
    """Event from voice session."""
    type: str
    data: dict[str, Any]


class OpenAIVoiceSession:
    """
    Real-time voice session with OpenAI.
    
    Usage:
        config = VoiceConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice=VoicePersonality.NOVA,
            instructions="You are a helpful exploration assistant.",
        )
        
        session = OpenAIVoiceSession(config, agent_id=agent.id)
        
        async with session:
            # Send audio chunks
            await session.send_audio(audio_bytes)
            
            # Receive events
            async for event in session.events():
                if event.type == "audio":
                    play_audio(event.data["audio"])
                elif event.type == "transcript":
                    print(event.data["text"])
    """
    
    REALTIME_URL = "wss://api.openai.com/v1/realtime"
    
    def __init__(
        self,
        config: VoiceConfig,
        agent_id: UUID | None = None,
        on_tool_call: Callable[[str, dict], Awaitable[Any]] | None = None,
    ):
        self.config = config
        self.agent_id = agent_id
        self.on_tool_call = on_tool_call
        self._ws: WebSocketClientProtocol | None = None
        self._event_queue: asyncio.Queue[VoiceEvent] = asyncio.Queue()
        self._running = False
        self._receive_task: asyncio.Task | None = None
    
    async def __aenter__(self) -> "OpenAIVoiceSession":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish WebSocket connection to OpenAI Realtime API."""
        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        url = f"{self.REALTIME_URL}?model={self.config.model}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        
        self._ws = await websockets.connect(url, extra_headers=headers)
        self._running = True
        
        # Configure session
        await self._send_event("session.update", {
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self._build_instructions(),
                "voice": self.config.voice.value,
                "input_audio_format": self.config.input_audio_format.value,
                "output_audio_format": self.config.output_audio_format.value,
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": self.config.turn_detection,
                "tools": self._build_tools(),
                "tool_choice": "auto",
                "temperature": self.config.temperature,
                "max_response_output_tokens": self.config.max_response_output_tokens,
            }
        })
        
        # Start receiving events
        self._receive_task = asyncio.create_task(self._receive_loop())
    
    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            await self._ws.close()
            self._ws = None
    
    async def send_audio(self, audio_bytes: bytes) -> None:
        """
        Send audio chunk to the session.
        
        Args:
            audio_bytes: Raw audio in configured format (default: PCM16, 24kHz, mono)
        """
        if not self._ws:
            raise RuntimeError("Not connected")
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        await self._send_event("input_audio_buffer.append", {"audio": audio_b64})
    
    async def send_text(self, text: str) -> None:
        """Send text message (will be converted to speech)."""
        if not self._ws:
            raise RuntimeError("Not connected")
        
        await self._send_event("conversation.item.create", {
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        })
        await self._send_event("response.create", {})
    
    async def commit_audio(self) -> None:
        """Commit the audio buffer and trigger response."""
        if not self._ws:
            raise RuntimeError("Not connected")
        
        await self._send_event("input_audio_buffer.commit", {})
    
    async def cancel_response(self) -> None:
        """Cancel the current response (for interruptions)."""
        if not self._ws:
            raise RuntimeError("Not connected")
        
        await self._send_event("response.cancel", {})
    
    async def events(self):
        """Async generator for voice events."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue
    
    # =========================================================================
    # Internal methods
    # =========================================================================
    
    def _build_instructions(self) -> str:
        """Build system instructions for the voice agent."""
        base = self.config.instructions or "You are a helpful AI assistant."
        
        if self.agent_id:
            base += f"\n\nYou are communicating on behalf of agent {self.agent_id}."
        
        return base
    
    def _build_tools(self) -> list[dict]:
        """Build tool definitions for function calling."""
        # Default agent tools
        default_tools = [
            {
                "type": "function",
                "name": "agent_move",
                "description": "Move the agent to a new location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "description": "Target latitude"},
                        "longitude": {"type": "number", "description": "Target longitude"},
                    },
                    "required": ["latitude", "longitude"]
                }
            },
            {
                "type": "function",
                "name": "agent_explore",
                "description": "Have the agent explore a topic or location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to explore"},
                    },
                    "required": ["topic"]
                }
            },
            {
                "type": "function",
                "name": "agent_status",
                "description": "Get the current status of the agent",
                "parameters": {"type": "object", "properties": {}}
            },
        ]
        
        return default_tools + self.config.tools
    
    async def _send_event(self, event_type: str, data: dict) -> None:
        """Send event to OpenAI."""
        if not self._ws:
            return
        
        message = {"type": event_type, **data}
        await self._ws.send(json.dumps(message))
    
    async def _receive_loop(self) -> None:
        """Background task to receive and process events."""
        if not self._ws:
            return
        
        try:
            async for message in self._ws:
                if not self._running:
                    break
                
                event = json.loads(message)
                await self._handle_event(event)
                
        except websockets.ConnectionClosed:
            self._running = False
        except Exception as e:
            await self._event_queue.put(VoiceEvent(
                type="error",
                data={"error": str(e)}
            ))
    
    async def _handle_event(self, event: dict) -> None:
        """Process incoming event from OpenAI."""
        event_type = event.get("type", "")
        
        # Audio output
        if event_type == "response.audio.delta":
            audio_b64 = event.get("delta", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await self._event_queue.put(VoiceEvent(
                    type="audio",
                    data={"audio": audio_bytes}
                ))
        
        # Transcription
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            await self._event_queue.put(VoiceEvent(
                type="transcript_input",
                data={"text": transcript}
            ))
        
        elif event_type == "response.audio_transcript.delta":
            delta = event.get("delta", "")
            await self._event_queue.put(VoiceEvent(
                type="transcript_output_delta",
                data={"text": delta}
            ))
        
        elif event_type == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            await self._event_queue.put(VoiceEvent(
                type="transcript_output",
                data={"text": transcript}
            ))
        
        # Function calls
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            arguments = json.loads(event.get("arguments", "{}"))
            
            await self._event_queue.put(VoiceEvent(
                type="tool_call",
                data={"call_id": call_id, "name": name, "arguments": arguments}
            ))
            
            # Execute tool if handler provided
            if self.on_tool_call:
                try:
                    result = await self.on_tool_call(name, arguments)
                    await self._send_tool_result(call_id, result)
                except Exception as e:
                    await self._send_tool_result(call_id, {"error": str(e)})
        
        # Session events
        elif event_type == "session.created":
            await self._event_queue.put(VoiceEvent(
                type="session_created",
                data={"session_id": event.get("session", {}).get("id")}
            ))
        
        elif event_type == "session.updated":
            await self._event_queue.put(VoiceEvent(
                type="session_updated",
                data={}
            ))
        
        # Response events
        elif event_type == "response.done":
            await self._event_queue.put(VoiceEvent(
                type="response_done",
                data={}
            ))
        
        # Error
        elif event_type == "error":
            await self._event_queue.put(VoiceEvent(
                type="error",
                data={"error": event.get("error", {})}
            ))
    
    async def _send_tool_result(self, call_id: str, result: Any) -> None:
        """Send tool call result back to OpenAI."""
        await self._send_event("conversation.item.create", {
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result)
            }
        })
        await self._send_event("response.create", {})

