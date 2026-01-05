"""Communication module - multi-channel human-agent interaction.

Channels:
- voice/     - Real-time speech (OpenAI Realtime API)
- text/      - Chat messaging (WebSocket)
- terminal/  - CLI commands

All channels route to agents — conversation OR commands, not "intent parsing".
"""

from .voice import OpenAIVoiceSession

__all__ = ["OpenAIVoiceSession"]

