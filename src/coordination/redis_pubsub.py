"""Redis pub/sub for distributed agent coordination."""

import asyncio
import json
from typing import Any, Callable, Coroutine
from uuid import UUID

import redis.asyncio as redis

from src.config import settings


class RedisPubSub:
    """Redis pub/sub coordinator for distributed messaging."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self.client: redis.Redis | None = None
        self.pubsub: redis.client.PubSub | None = None
        self._subscribers: dict[str, list[Callable]] = {}
        self._running = False
        self._listen_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.client.pubsub()

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self.pubsub:
            await self.pubsub.close()
        if self.client:
            await self.client.close()

    async def publish(self, channel: str, message: dict[str, Any]) -> None:
        """Publish message to channel."""
        if not self.client:
            raise RuntimeError("Not connected to Redis")

        await self.client.publish(channel, json.dumps(message))

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to channel with callback."""
        if not self.pubsub:
            raise RuntimeError("Not connected to Redis")

        if channel not in self._subscribers:
            self._subscribers[channel] = []
            await self.pubsub.subscribe(channel)

        self._subscribers[channel].append(callback)

        # Start listener if not running
        if not self._running:
            self._running = True
            self._listen_task = asyncio.create_task(self._listen())

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel."""
        if not self.pubsub:
            return

        if channel in self._subscribers:
            await self.pubsub.unsubscribe(channel)
            del self._subscribers[channel]

    async def _listen(self) -> None:
        """Listen for messages on subscribed channels."""
        if not self.pubsub:
            return

        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = json.loads(message["data"])

                    # Call all subscribers for this channel
                    if channel in self._subscribers:
                        for callback in self._subscribers[channel]:
                            asyncio.create_task(callback(data))
        except asyncio.CancelledError:
            pass

    # -------------------------------------------------------------------------
    # Agent Coordination Channels
    # -------------------------------------------------------------------------

    async def publish_agent_event(
        self,
        agent_id: UUID,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Publish agent event."""
        message = {
            "agent_id": str(agent_id),
            "event_type": event_type,
            "data": data,
        }
        await self.publish(f"agent:{agent_id}", message)
        await self.publish("agents:all", message)

    async def subscribe_to_agent(
        self,
        agent_id: UUID,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to specific agent events."""
        await self.subscribe(f"agent:{agent_id}", callback)

    async def subscribe_to_all_agents(
        self,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to all agent events."""
        await self.subscribe("agents:all", callback)

    async def publish_simulation_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Publish simulation event."""
        message = {
            "event_type": event_type,
            "data": data,
        }
        await self.publish("simulation:events", message)

    async def subscribe_to_simulation(
        self,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to simulation events."""
        await self.subscribe("simulation:events", callback)

    async def publish_message_event(
        self,
        sender_id: UUID,
        recipient_id: UUID | None,
        message_type: str,
        content: Any,
    ) -> None:
        """Publish inter-agent message event."""
        message = {
            "sender_id": str(sender_id),
            "recipient_id": str(recipient_id) if recipient_id else None,
            "message_type": message_type,
            "content": content,
        }
        await self.publish("messages:all", message)

        if recipient_id:
            await self.publish(f"messages:{recipient_id}", message)

    async def subscribe_to_messages(
        self,
        agent_id: UUID,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to messages for specific agent."""
        await self.subscribe(f"messages:{agent_id}", callback)


# Global instance
_pubsub: RedisPubSub | None = None


def get_pubsub() -> RedisPubSub | None:
    """Get the global pub/sub instance."""
    return _pubsub


async def init_pubsub(redis_url: str | None = None) -> RedisPubSub:
    """Initialize the global pub/sub instance."""
    global _pubsub
    _pubsub = RedisPubSub(redis_url)
    await _pubsub.connect()
    return _pubsub


async def shutdown_pubsub() -> None:
    """Shutdown the global pub/sub instance."""
    global _pubsub
    if _pubsub:
        await _pubsub.disconnect()
        _pubsub = None
