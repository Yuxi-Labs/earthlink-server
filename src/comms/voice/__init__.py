"""Voice communication - real-time speech with agents.

Uses OpenAI Realtime API for speech-to-speech conversation.
Transcripts are logged to database for history and analysis.
"""

from .openai import OpenAIVoiceSession, VoiceConfig
from .transcript import TranscriptLogger

__all__ = ["OpenAIVoiceSession", "VoiceConfig", "TranscriptLogger"]

