"""Core Agent class - autonomous intelligent actor."""

import asyncio
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID

import httpx
import numpy as np
import ray
import torch

from .messaging import Mailbox, Message, MessagePriority, MessageType
from .state import AgentLifecycle, AgentState, AgentStatus


class Agent:
    """
    Autonomous intelligent agent.
    
    NOT an LLM wrapper - agents think for themselves using true AI/ML.
    Runs as a Ray actor for concurrency.
    """

    def __init__(
        self,
        agent_id: UUID | None = None,
        name: str = "",
        config: dict[str, Any] | None = None,
    ):
        self.state = AgentState(name=name)
        if agent_id:
            self.state.id = agent_id

        self.config = config or {}

        # Initialize individual curiosity from config (each agent is unique)
        if "curiosity" in self.config:
            self.state.metrics.curiosity_score = self.config["curiosity"]

        # Components (lazy initialized)
        self._memory: Any = None
        self._policy: Any = None
        self._world_model: Any = None
        self._curiosity: Any = None
        self._goal_system: Any = None
        self._perception: Any = None  # PerceptionModule
        self._reasoning: Any = None  # ReasoningEngine
        self._decision: Any = None  # DecisionModule
        self._learning: Any = None  # LearningModule
        self._monitoring: Any = None  # SelfMonitoringModule
        self._generation: Any = None  # GenerationModule
        self._specialization: Any = None  # SpecializationModule
        self._knowledge_transfer: Any = None  # KnowledgeTransferModule
        self._modeling: Any = None  # ModelingModule
        self._evolution: Any = None  # EvolutionModule
        self._communication: Any = None  # CommunicationModule
        self._adaptation: Any = None  # AdaptationModule
        self._last_memory_consolidation_step: int = 0

        # Messaging
        self._mailbox = Mailbox(agent_id=self.state.id)
        self._outgoing_messages: list[Message] = []
        
        # Optimizers
        self._policy_optimizer: Any = None
        self._world_model_optimizer: Any = None
        self._curiosity_optimizer: Any = None

        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Earthlink location awareness (use Docker service name if available)
        import os
        earthlink_host = os.getenv("EARTHLINK_API_HOST", "localhost")
        self._earthlink_api_base = f"http://{earthlink_host}:8000/api/v1/earthlink"
        self._http_client: httpx.AsyncClient | None = None

    # -------------------------------------------------------------------------
    # Local/test helpers
    # -------------------------------------------------------------------------

    @classmethod
    def options(cls, **options):
        """
        Minimal shim mirroring Ray's ``ActorClass.options`` for local/tests.
        
        Returns an object with ``_remote`` that just instantiates the class.
        """

        class _LocalOptions:
            def __init__(self, target_cls, opts):
                self._cls = target_cls
                self._opts = opts

            def _remote(self, *args, **kwargs):
                init_kwargs = {**self._opts, **kwargs}
                # Drop Ray-only options if passed accidentally
                init_kwargs.pop("get_if_exists", None)
                return self._cls(*args, **init_kwargs)

        return _LocalOptions(cls, options)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def _run_with_retries(self, coro_or_factory, retries: int = 2, delay: float = 0.5):
        """Run coroutine (or coroutine factory) with simple retry."""
        attempt = 0
        while True:
            try:
                if callable(coro_or_factory) and not asyncio.iscoroutine(coro_or_factory):
                    return await coro_or_factory()
                # If a factory is provided, call it to get a fresh coroutine each attempt
                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                return await coro
            except Exception as e:
                attempt += 1
                if attempt > retries:
                    raise e
                await asyncio.sleep(delay)

    def get_state(self) -> dict[str, Any]:
        """Return serializable agent state including capability data."""
        base_state = self.state.to_dict()
        
        # Add capability module data
        base_state["reasoning"] = self._get_reasoning_state()
        base_state["decision"] = self._get_decision_state()
        base_state["world_model"] = self._get_world_model_state()
        base_state["memory"] = self._get_memory_state()
        base_state["adaptation"] = self._get_adaptation_state()
        base_state["monitoring"] = self._get_monitoring_state()
        base_state["specialization"] = self._get_specialization_state()
        base_state["evolution"] = self._get_evolution_state()
        base_state["knowledge_transfer"] = self._get_knowledge_transfer_state()
        base_state["outputs"] = self._get_outputs_state()
        base_state["recent_observations"] = getattr(self, '_recent_observations', [])[-10:]
        base_state["action_history"] = getattr(self, '_action_history', [])[-20:]
        base_state["movement_history"] = getattr(self, '_movement_history', [])[-20:]
        
        return base_state
    
    def _get_reasoning_state(self) -> dict[str, Any]:
        """Get reasoning engine state for API."""
        if not self._reasoning:
            return {}
        try:
            return {
                "hypotheses": [
                    {"id": h.id, "text": h.text, "confidence": h.confidence, "status": h.status}
                    for h in getattr(self._reasoning, 'hypotheses', {}).values()
                ][:10],
                "predictions": getattr(self._reasoning, '_recent_predictions', [])[-10:],
                "causal_relations": [
                    {"id": c.id, "cause": c.cause, "effect": c.effect, "strength": c.strength}
                    for c in getattr(self._reasoning, 'causal_relations', {}).values()
                ][:10],
                "prediction_accuracy": getattr(self._reasoning, 'average_prediction_error', 0),
                "hypotheses_confirmed": getattr(self._reasoning, 'hypotheses_confirmed', 0),
                "hypotheses_rejected": getattr(self._reasoning, 'hypotheses_rejected', 0),
            }
        except Exception:
            return {}
    
    def _get_decision_state(self) -> dict[str, Any]:
        """Get decision module state for API."""
        if not self._decision:
            return {}
        try:
            return {
                "last_decision": getattr(self._decision, '_last_decision', {}),
                "decision_history": getattr(self._decision, '_decision_history', [])[-10:],
                "strategy": getattr(self._decision, 'strategy', 'balanced'),
                "total_decisions": getattr(self._decision, '_total_decisions', 0),
            }
        except Exception:
            return {}
    
    def _get_world_model_state(self) -> dict[str, Any]:
        """Get world model state for API."""
        if not self._world_model:
            return {}
        try:
            return {
                "regions": getattr(self._world_model, 'regions', {}),
                "entities": getattr(self._world_model, 'entities', {}),
                "confidence": getattr(self._world_model, 'confidence', 0),
            }
        except Exception:
            return {}
    
    def _get_memory_state(self) -> dict[str, Any]:
        """Get memory state for API."""
        if not self._memory:
            return {}
        try:
            episodic = getattr(self._memory, 'episodic', [])
            semantic = getattr(self._memory, 'semantic', [])
            procedural = getattr(self._memory, 'procedural', [])
            working = getattr(self._memory, 'working', [])
            return {
                "episodic": episodic[-5:] if episodic else [],
                "semantic": semantic[-5:] if semantic else [],
                "procedural": procedural[-5:] if procedural else [],
                "working": working if working else [],
                "total_memories": len(episodic) + len(semantic) + len(procedural),
                "working_capacity": getattr(self._memory, 'working_capacity', 7),
            }
        except Exception:
            return {}
    
    def _get_adaptation_state(self) -> dict[str, Any]:
        """Get adaptation state for API."""
        if not self._adaptation:
            return {}
        try:
            return {
                "current_strategy": getattr(self._adaptation, 'current_strategy', 'balanced'),
                "strategy_history": getattr(self._adaptation, '_strategy_history', [])[-10:],
                "detected_shifts": getattr(self._adaptation, '_detected_shifts', [])[-5:],
                "behavior_modifications": getattr(self._adaptation, '_modifications', [])[-5:],
            }
        except Exception:
            return {}
    
    def _get_monitoring_state(self) -> dict[str, Any]:
        """Get self-monitoring state for API."""
        if not self._monitoring:
            return {}
        try:
            return {
                "decision_confidence": getattr(self._monitoring, 'decision_confidence', 0),
                "prediction_uncertainty": getattr(self._monitoring, 'prediction_uncertainty', 0),
                "calibration_score": getattr(self._monitoring, 'calibration_score', 0),
                "health_status": getattr(self._monitoring, 'health_status', 'healthy'),
                "diagnosis": getattr(self._monitoring, '_last_diagnosis', {}),
                "alerts": getattr(self._monitoring, '_alerts', [])[-5:],
            }
        except Exception:
            return {}
    
    def _get_specialization_state(self) -> dict[str, Any]:
        """Get specialization state for API."""
        if not self._specialization:
            return {}
        try:
            return {
                "primary_domain": getattr(self._specialization, 'primary_domain', None),
                "expertise_levels": getattr(self._specialization, 'expertise_levels', {}),
                "niche_discovered": getattr(self._specialization, 'niche_discovered', None),
                "specialization_score": getattr(self._specialization, 'specialization_score', 0),
            }
        except Exception:
            return {}
    
    def _get_evolution_state(self) -> dict[str, Any]:
        """Get evolution state for API."""
        if not self._evolution:
            return {}
        try:
            return {
                "generation": getattr(self._evolution, 'generation', 0),
                "parent_id": getattr(self._evolution, 'parent_id', None),
                "offspring_ids": getattr(self._evolution, 'offspring_ids', []),
                "fitness_score": getattr(self._evolution, 'fitness_score', 0),
                "mutations": getattr(self._evolution, 'mutations', []),
            }
        except Exception:
            return {}
    
    def _get_knowledge_transfer_state(self) -> dict[str, Any]:
        """Get knowledge transfer state for API."""
        if not self._knowledge_transfer:
            return {}
        try:
            return {
                "extracted": getattr(self._knowledge_transfer, '_extracted', [])[-10:],
                "shared_with": getattr(self._knowledge_transfer, '_shared_with', []),
                "received_from": getattr(self._knowledge_transfer, '_received_from', []),
                "collective_contributions": getattr(self._knowledge_transfer, '_contributions', 0),
            }
        except Exception:
            return {}
    
    def _get_outputs_state(self) -> dict[str, Any]:
        """Get generated outputs state for API."""
        if not self._generation:
            return {}
        try:
            return {
                "maps": getattr(self._generation, '_maps', [])[-5:],
                "summaries": getattr(self._generation, '_summaries', [])[-5:],
                "theories": getattr(self._generation, '_theories', [])[-5:],
                "total_outputs": getattr(self._generation, '_total_outputs', 0),
            }
        except Exception:
            return {}

    def get_id(self) -> UUID:
        """Return agent ID."""
        return self.state.id

    def set_lifecycle(self, lifecycle: AgentLifecycle) -> None:
        """Transition agent lifecycle state."""
        old_lifecycle = self.state.lifecycle
        self.state.lifecycle = lifecycle
        self.state.updated_at = datetime.now(UTC)
        
        # Broadcast lifecycle change if event_bus available
        if hasattr(self, 'event_bus') and self.event_bus and old_lifecycle != lifecycle:
            from src.simulation.events import Event, EventType
            event = Event(
                event_type=EventType.AGENT_UPDATED,
                source_agent_id=self.state.id,
                data={
                    "agent_id": str(self.state.id),
                    "status": self.state.status.value,
                    "lifecycle": lifecycle.value,
                    "old_lifecycle": old_lifecycle.value if old_lifecycle else None,
                }
            )
            self.event_bus.publish(event)

    def set_status(self, status: AgentStatus) -> None:
        """Update agent operational status (UI-facing)."""
        old_status = self.state.status
        self.state.status = status
        self.state.updated_at = datetime.now(UTC)
        
        # Broadcast status change if event_bus available
        if hasattr(self, 'event_bus') and self.event_bus and old_status != status:
            from src.simulation.events import Event, EventType
            event = Event(
                event_type=EventType.AGENT_UPDATED,
                source_agent_id=self.state.id,
                data={
                    "agent_id": str(self.state.id),
                    "status": status.value,
                    "lifecycle": self.state.lifecycle.value,
                    "old_status": old_status.value if old_status else None,
                }
            )
            self.event_bus.publish(event)

    def set_target_world(self, world: str) -> None:
        """Set target world for specialization/deployment."""
        self.state.target_world = world
        self.state.updated_at = datetime.now(UTC)

    # -------------------------------------------------------------------------
    # Messaging
    # -------------------------------------------------------------------------

    def receive_message(self, message: Message) -> bool:
        """Receive a message from another agent."""
        self._mailbox.receive(message)
        return True

    def get_messages(self, unread_only: bool = True, limit: int = 50) -> list[dict[str, Any]]:
        """Get messages from mailbox."""
        if unread_only:
            messages = self._mailbox.get_unread(limit)
        else:
            messages = self._mailbox.inbox[-limit:]
        return [m.to_dict() for m in messages]

    def get_messages_by_type(self, msg_type: str) -> list[dict[str, Any]]:
        """Get messages of a specific type."""
        try:
            mtype = MessageType(msg_type)
        except ValueError:
            return []
        return [m.to_dict() for m in self._mailbox.get_by_type(mtype)]

    def mark_message_read(self, message_id: str) -> bool:
        """Mark a message as read."""
        return self._mailbox.mark_read(UUID(message_id))

    def send_message(
        self,
        recipient_id: str | None,
        subject: str,
        content: Any,
        msg_type: str = "direct",
        priority: int = 1,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Queue a message to be sent to another agent.
        
        Returns the message that was queued.
        """
        try:
            mtype = MessageType(msg_type)
        except ValueError:
            mtype = MessageType.DIRECT

        try:
            mpriority = MessagePriority(priority)
        except ValueError:
            mpriority = MessagePriority.NORMAL

        msg = Message(
            type=mtype,
            priority=mpriority,
            sender_id=self.state.id,
            recipient_id=recipient_id if isinstance(recipient_id, UUID) else UUID(recipient_id) if recipient_id else None,
            subject=subject,
            content=content,
            metadata=kwargs,
        )
        self._outgoing_messages.append(msg)
        return msg.to_dict()

    def broadcast_message(
        self,
        subject: str,
        content: Any,
        priority: int = 1,
        **kwargs,
    ) -> dict[str, Any]:
        """Queue a broadcast message to all agents."""
        return self.send_message(
            recipient_id=None,
            subject=subject,
            content=content,
            msg_type="broadcast",
            priority=priority,
            **kwargs,
        )

    def get_outgoing_messages(self) -> list[dict[str, Any]]:
        """Get and clear outgoing message queue. Called by router."""
        messages = [m.to_dict() for m in self._outgoing_messages]
        self._outgoing_messages.clear()
        return messages

    def get_outgoing_raw(self) -> list[Message]:
        """Get and clear outgoing messages as Message objects."""
        messages = self._outgoing_messages.copy()
        self._outgoing_messages.clear()
        return messages

    def reply_to_message(
        self,
        original_message_id: str,
        content: Any,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Reply to a specific message."""
        # Find original message
        original = None
        for msg in self._mailbox.inbox:
            if str(msg.id) == original_message_id:
                original = msg
                break

        if not original:
            return None

        reply = original.create_reply(
            sender_id=self.state.id,
            content=content,
            msg_type=MessageType.RESPONSE,
        )
        reply.metadata.update(kwargs)
        self._outgoing_messages.append(reply)
        return reply.to_dict()

    def share_knowledge(
        self,
        recipient_id: str | None,
        knowledge_type: str,
        content: Any,
    ) -> dict[str, Any]:
        """Share knowledge with another agent or broadcast."""
        return self.send_message(
            recipient_id=recipient_id,
            subject=f"knowledge:{knowledge_type}",
            content=content,
            msg_type="share_knowledge" if recipient_id else "broadcast",
        )

    def request_collaboration(
        self,
        recipient_id: str,
        task: str,
        details: Any,
    ) -> dict[str, Any]:
        """Request collaboration from another agent."""
        return self.send_message(
            recipient_id=recipient_id,
            subject=task,
            content=details,
            msg_type="collaborate_request",
            priority=2,  # High priority
        )

    def get_mailbox_stats(self) -> dict[str, Any]:
        """Get mailbox statistics."""
        return self._mailbox.stats()

    # -------------------------------------------------------------------------
    # Knowledge Acquisition
    # -------------------------------------------------------------------------

    async def query_wikipedia(self, query: str, **kwargs) -> dict[str, Any]:
        """Query Wikipedia for knowledge."""
        from src.data.sources import WikipediaSource
        
        wiki = WikipediaSource()
        
        # Search for articles
        results = await wiki.search(query, **kwargs)
        
        # Store knowledge in semantic memory
        if results and self._memory:
            for article_data in results[:3]:  # Top 3 results
                # Fetch full article
                article = await wiki.get_article(article_data["title"])
                if article:
                    # Store in semantic memory
                    knowledge_text = f"{article.title}\n\n{article.summary}"
                    await self._memory.store_semantic(
                        content=knowledge_text,
                        metadata={"source": "wikipedia", "title": article.title, "url": article.url}
                    )
        
        return {"results": results, "query": query}

    async def query_ollama(self, prompt: str, model: str = "llama2", **kwargs) -> dict[str, Any]:
        """Query Ollama for knowledge or reasoning."""
        from src.data.sources import OllamaSource
        
        ollama = OllamaSource()
        
        # Generate response
        response = await ollama.generate(prompt=prompt, model=model, **kwargs)
        
        # Store interaction in episodic memory
        if response and self._memory:
            await self._memory.store_semantic(
                content=f"Query: {prompt}\n\nResponse: {response.get('response', '')}",
                metadata={"source": "ollama", "model": model}
            )
        
        return response

    async def query_search(self, query: str, provider: str = "duckduckgo", **kwargs) -> dict[str, Any]:
        """Lightweight alias used by tests; routes to web search."""
        return await self.search_web(query, provider=provider, **kwargs)

    async def search_web(self, query: str, provider: str = "duckduckgo", **kwargs) -> dict[str, Any]:
        """Search the web for information."""
        from src.data.sources import DuckDuckGoSearchSource
        
        search = DuckDuckGoSearchSource()
        
        # Search using DuckDuckGo
        search_results = await search.search(query, **kwargs)
        
        # Convert SearchResult objects to dicts
        results = [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in search_results]
        
        # Store results in semantic memory
        if results and self._memory:
            for result in results[:5]:  # Top 5 results
                knowledge_text = f"{result.get('title', '')}\n\n{result.get('snippet', '')}"
                await self._memory.store_semantic(
                    content=knowledge_text,
                    metadata={"source": f"search_{provider}", "url": result.get('url', '')}
                )
        
        return {"results": results, "query": query, "provider": provider}

    async def query_reddit(self, subreddit: str, **kwargs) -> dict[str, Any]:
        """Query Reddit for discussions and knowledge."""
        from src.data.sources import RedditSource
        
        reddit = RedditSource()
        
        # Get subreddit posts
        posts = await reddit.get_subreddit_posts(subreddit, **kwargs)
        
        # Store in semantic memory
        if posts and self._memory:
            for post in posts[:10]:  # Top 10 posts
                content = f"{post.get('title', '')}\n\n{post.get('selftext', '')}"
                await self._memory.store_semantic(
                    content=content,
                    metadata={"source": "reddit", "subreddit": subreddit, "url": post.get('url', '')}
                )
        
        return {"posts": posts, "subreddit": subreddit}

    async def query_x(self, query: str, **kwargs) -> dict[str, Any]:
        """Query X for recent discussions."""
        from src.data.sources import XSource
        
        x = XSource()
        
        # Search recent posts
        posts = await x.search_recent_posts(query, **kwargs)
        
        # Store in semantic memory
        if posts and self._memory:
            for post in posts[:20]:  # Top 20 posts
                await self._memory.store_semantic(
                    content=post.text if hasattr(post, 'text') else post.get('text', ''),
                    metadata={"source": "x", "author": post.author_id if hasattr(post, 'author_id') else post.get('author_id', ''), "url": f"https://x.com/i/web/status/{post.id if hasattr(post, 'id') else post.get('id', '')}"}
                )
        
        return {"posts": posts, "query": query}

    async def query_geo(
        self,
        lat: float,
        lon: float,
        radius_meters: float = 1000,
        **kwargs
    ) -> dict[str, Any]:
        """
        Query geospatial data near a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_meters: Search radius in meters
            
        Returns:
            Nearby geographic features and regions
        """
        from src.data.worlds.base.earth import get_geo_source
        
        geo = get_geo_source()
        
        # Get location context (features + regions)
        try:
            context = await geo.get_location_context(lat, lon)
        except Exception:
            # Handle missing geo_features table or other DB issues gracefully
            context = {"nearby_features": [], "containing_regions": []}
        
        # Store in semantic memory
        if context and self._memory:
            # Store nearby features
            for feature in context.get("nearby_features", [])[:10]:
                await self._memory.store_semantic(
                    content=f"Location: {feature.get('name', 'Unknown')}\nType: {feature.get('type', 'Unknown')}\nDistance: {feature.get('distance_meters', 0):.0f}m",
                    metadata={"source": "geo", "lat": lat, "lon": lon, "feature_type": feature.get('type', '')}
                )
            
            # Store containing regions
            for region in context.get("containing_regions", [])[:5]:
                await self._memory.store_semantic(
                    content=f"Region: {region.get('name', 'Unknown')}\nType: {region.get('type', 'Unknown')}",
                    metadata={"source": "geo", "region_type": region.get('type', '')}
                )
        
        # Update location
        self.state.location.x = lat
        self.state.location.y = lon
        
        return context

    async def explore_location(
        self,
        lat: float,
        lon: float
    ) -> dict[str, Any]:
        """
        Explore a geographic location - query geo data and related knowledge.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Combined geo and knowledge exploration results
        """
        results = {
            "lat": lat,
            "lon": lon,
            "geo_context": {},
            "knowledge": {},
        }
        
        # Get geo context
        geo_context = await self.query_geo(lat, lon, radius_meters=5000)
        results["geo_context"] = geo_context
        
        # Explore knowledge about top regions/features
        regions = geo_context.get("containing_regions", [])
        if regions:
            # Explore largest containing region
            region_name = regions[0].get("name", "")
            if region_name:
                knowledge = await self.explore_topic(region_name)
                results["knowledge"] = knowledge
        
        # Update metrics
        self.state.metrics.knowledge_sources_queried += 1
        
        return results

    # -------------------------------------------------------------------------
    # Earthlink Location Awareness
    # -------------------------------------------------------------------------
    
    def get_latitude(self) -> float:
        """Get current latitude in Earthlink (stored in location.x)."""
        return self.state.location.x
    
    def get_longitude(self) -> float:
        """Get current longitude in Earthlink (stored in location.y)."""
        return self.state.location.y
    
    def get_altitude(self) -> float:
        """Get current altitude in meters (stored in location.z)."""
        return self.state.location.z
    
    @property
    def latitude(self) -> float:
        """Current latitude in Earthlink (stored in location.x)."""
        return self.state.location.x
    
    @property
    def longitude(self) -> float:
        """Current longitude in Earthlink (stored in location.y)."""
        return self.state.location.y
    
    @property
    def altitude(self) -> float:
        """Current altitude in meters (stored in location.z)."""
        return self.state.location.z
    
    def set_earthlink_position(self, lat: float, lon: float, altitude: float = 0.0) -> None:
        """
        Set agent's position in Earthlink.
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            altitude: Altitude in meters (default 0)
        """
        self.state.location.x = lat
        self.state.location.y = lon
        self.state.location.z = altitude
    
    def _calculate_destination(self, lat: float, lon: float, distance_km: float, bearing_degrees: float) -> tuple[float, float]:
        """
        Calculate destination point given start point, distance, and bearing.
        Uses the haversine formula for great circle navigation.
        
        Args:
            lat: Starting latitude in degrees
            lon: Starting longitude in degrees
            distance_km: Distance to travel in kilometers
            bearing_degrees: Bearing in degrees (0-360, 0=North, 90=East)
        
        Returns:
            (new_lat, new_lon) in degrees
        """
        import math
        
        # Earth's radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing_degrees)
        
        # Angular distance in radians
        angular_distance = distance_km / R
        
        # Calculate new latitude
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        # Calculate new longitude
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        # Convert back to degrees
        new_lat = math.degrees(new_lat_rad)
        new_lon = math.degrees(new_lon_rad)
        
        # Normalize longitude to -180 to 180
        new_lon = ((new_lon + 180) % 360) - 180
        
        # Clamp latitude to valid range
        new_lat = max(-90, min(90, new_lat))
        
        return new_lat, new_lon
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for Earthlink API."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def earthlink_query_nearby(
        self,
        radius_meters: float = 500,
        limit: int = 10,
        feature_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query nearby features from Earthlink at current position.
        
        Args:
            radius_meters: Search radius (10-50000)
            limit: Max results per feature type (1-100)
            feature_types: Filter types (buildings, roads, places, pois, water, landcover)
        
        Returns:
            Dict with nearby buildings, roads, POIs, etc.
        """
        client = await self._get_http_client()
        
        params = {
            "lat": self.latitude,
            "lon": self.longitude,
            "radius_meters": radius_meters,
            "limit": limit,
        }
        
        if feature_types:
            params["types"] = ",".join(feature_types)
        
        response = await client.get(f"{self._earthlink_api_base}/nearby", params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def earthlink_get_context(
        self,
        radius_meters: float = 5000,
    ) -> dict[str, Any]:
        """
        Get comprehensive context about current location.
        
        Returns nearby features, containing regions (boundaries), and summary stats.
        
        Args:
            radius_meters: Search radius (100-50000)
        
        Returns:
            Dict with comprehensive location context
        """
        client = await self._get_http_client()
        
        params = {
            "lat": self.latitude,
            "lon": self.longitude,
            "radius_meters": radius_meters,
        }
        
        response = await client.get(f"{self._earthlink_api_base}/context", params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def earthlink_get_regions(self) -> dict[str, Any]:
        """
        Get administrative boundaries containing current position.
        
        Returns nested regions (country → state → city → neighborhood).
        
        Returns:
            Dict with containing regions
        """
        client = await self._get_http_client()
        
        params = {
            "lat": self.latitude,
            "lon": self.longitude,
        }
        
        response = await client.get(f"{self._earthlink_api_base}/regions", params=params)
        response.raise_for_status()
        
        return response.json()
    
    async def earthlink_where_am_i(self) -> dict[str, Any]:
        """
        Answer the question: "Where am I?"
        
        Returns a human-readable description of current location including:
        - Coordinates
        - Administrative regions (city, state, country)
        - Nearby landmarks/buildings
        - Local roads
        
        Returns:
            Dict with comprehensive "where am I" information
        """
        # Get both regions and nearby features
        regions_data = await self.earthlink_get_regions()
        nearby_data = await self.earthlink_query_nearby(radius_meters=200, limit=5)
        
        # Extract key information
        regions = regions_data.get("regions", [])
        
        # Parse administrative hierarchy
        admin_context = {
            "neighborhood": None,
            "city": None,
            "state": None,
            "country": None,
        }
        
        for region in regions:
            name = region.get("name", "")
            
            # Simple heuristics for region classification
            if "neighborhood" in name.lower() or region.get("admin_level") == "10":
                admin_context["neighborhood"] = name
            elif "city" in name.lower() or region.get("admin_level") in ["8", "6"]:
                admin_context["city"] = name
            elif "state" in name.lower() or region.get("admin_level") == "4":
                admin_context["state"] = name
            elif "country" in name.lower() or region.get("admin_level") == "2":
                admin_context["country"] = name
        
        # Get notable nearby features
        nearby_buildings = nearby_data.get("by_category", {}).get("building", [])[:3]
        nearby_roads = nearby_data.get("by_category", {}).get("road", [])[:3]
        
        # Build human-readable description
        location_parts = []
        if admin_context["neighborhood"]:
            location_parts.append(admin_context["neighborhood"])
        if admin_context["city"]:
            location_parts.append(admin_context["city"])
        if admin_context["state"]:
            location_parts.append(admin_context["state"])
        if admin_context["country"]:
            location_parts.append(admin_context["country"])
        
        description = ", ".join(location_parts) if location_parts else "Unknown location"
        
        return {
            "position": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude,
            },
            "description": description,
            "administrative": admin_context,
            "nearby_buildings": [
                {
                    "name": b.get("name", "Unnamed"),
                    "type": b.get("type", "unknown"),
                    "distance_meters": b.get("distance_meters", 0),
                }
                for b in nearby_buildings
            ],
            "nearby_roads": [
                {
                    "name": r.get("name", "Unnamed"),
                    "type": r.get("type", "unknown"),
                    "distance_meters": r.get("distance_meters", 0),
                }
                for r in nearby_roads
            ],
        }
    
    async def earthlink_spawn_random_city(self, region: str = "south_america") -> dict[str, Any]:
        """
        Spawn agent at a random major city.
        
        Args:
            region: Geographic region (south_america, europe, north_america)
        
        Returns:
            Dict with spawn location and context
        """
        import random
        
        # Major cities by region (with Earthlink coverage)
        cities = {
            "south_america": [
                {"name": "São Paulo", "lat": -23.55, "lon": -46.63},
                {"name": "Rio de Janeiro", "lat": -22.91, "lon": -43.17},
                {"name": "Buenos Aires", "lat": -34.60, "lon": -58.38},
                {"name": "Lima", "lat": -12.05, "lon": -77.03},
                {"name": "Bogotá", "lat": 4.71, "lon": -74.07},
                {"name": "Santiago", "lat": -33.45, "lon": -70.67},
            ],
        }
        
        city_list = cities.get(region.lower(), cities["south_america"])
        selected_city = random.choice(city_list)
        
        # Set position
        self.set_earthlink_position(selected_city["lat"], selected_city["lon"])
        
        # Get context
        context = await self.earthlink_where_am_i()
        
        return {
            "spawned_at": selected_city["name"],
            "region": region,
            "context": context,
        }

    # -------------------------------------------------------------------------
    # Knowledge Exploration
    # -------------------------------------------------------------------------

    async def explore_topic(self, topic: str) -> dict[str, Any]:
        """
        Autonomously explore a topic across multiple knowledge sources.
        
        This is the core exploration behavior - agent decides what to learn.
        """
        results = {
            "topic": topic,
            "sources": {},
            "knowledge_gained": 0,
        }
        
        # Query multiple sources in parallel with retries for flakiness
        tasks = {
            "wikipedia": self._run_with_retries(lambda: self.query_wikipedia(topic, limit=5)),
            "search": self._run_with_retries(lambda: self.query_search(topic, provider="duckduckgo")),
            "ollama": self._run_with_retries(
                lambda: self.query_ollama(
                    f"Briefly explain {topic} and list key facts.",
                    model=self.config.get("ollama_model", "llama2"),
                )
            ),
        }
        
        # Execute all queries
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Collect results (skip failed sources)
        for source_name, result in zip(tasks.keys(), task_results):
            if isinstance(result, Exception):
                # Log but don't fail - some sources may not work
                print(f"[Agent {self.state.id}] Knowledge source '{source_name}' failed: {result}")
            else:
                results["sources"][source_name] = result
                # Count knowledge items
                if "results" in result:
                    results["knowledge_gained"] += len(result["results"])
                elif "posts" in result:
                    results["knowledge_gained"] += len(result["posts"])
                elif "tweets" in result:
                    results["knowledge_gained"] += len(result["tweets"])
        
        # Get goal context for logging
        goal_context_str = None
        if self._goal_system and self.state.current_goal_id:
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal:
                goal_context_str = current_goal.description
        
        # Log to database for audit trail
        if results["knowledge_gained"] > 0:
            await self._log_knowledge_acquisition(
                topic=topic,
                sources=list(results["sources"].keys()),
                knowledge_count=results["knowledge_gained"],
                content_summary=str(results["sources"])[:500],  # First 500 chars
                goal_context=goal_context_str,
                curiosity_signal=None,  # Not available in direct explore_topic call
            )
        
        return results

    # -------------------------------------------------------------------------
    # Core Loop: Perceive → Encode → Decide → Act → Learn
    # -------------------------------------------------------------------------

    def _ensure_perception(self):
        """Lazy initialize perception module."""
        if self._perception is None:
            from .perception import PerceptionModule
            self._perception = PerceptionModule(
                agent_id=self.state.id,
                device=str(self.device),
            )
    
    def _ensure_reasoning(self):
        """Lazy initialize reasoning engine."""
        if self._reasoning is None:
            from .reasoning import ReasoningEngine
            self._reasoning = ReasoningEngine(
                agent_id=self.state.id,
                device=str(self.device),
            )
    
    def _ensure_decision(self):
        """Lazy initialize decision module."""
        if self._decision is None:
            from .decision import DecisionModule, DecisionStrategy
            self._decision = DecisionModule(
                agent_id=self.state.id,
                strategy=DecisionStrategy.BALANCED,
                risk_tolerance=0.5,
                exploration_bonus=0.1,
            )
    
    def _ensure_learning(self):
        """Lazy initialize learning module."""
        if self._learning is None:
            from .learning import LearningModule
            self._learning = LearningModule(
                agent_id=self.state.id,
                base_learning_rate=0.01,
                meta_learning_enabled=True,
            )

    def _ensure_monitoring(self):
        """Lazy initialize monitoring module."""
        if self._monitoring is None:
            from .monitoring import SelfMonitoringModule
            self._monitoring = SelfMonitoringModule(window=100)

    def _ensure_generation(self):
        """Lazy initialize output generation module."""
        if self._generation is None:
            from .generation import GenerationModule
            self._generation = GenerationModule(agent_id=self.state.id)

    def _ensure_specialization(self):
        """Lazy initialize specialization module."""
        if self._specialization is None:
            from .specialization import SpecializationModule
            self._specialization = SpecializationModule(agent_id=self.state.id)

    def _ensure_knowledge_transfer(self):
        """Lazy initialize knowledge transfer module."""
        if self._knowledge_transfer is None:
            from .knowledge_transfer import KnowledgeTransferModule
            self._knowledge_transfer = KnowledgeTransferModule(agent_id=self.state.id)

    def _ensure_modeling(self):
        """Lazy initialize modeling module."""
        if self._modeling is None:
            from .modeling import ModelingModule
            self._modeling = ModelingModule(agent_id=self.state.id)

    def _ensure_evolution(self):
        """Lazy initialize evolution module."""
        if self._evolution is None:
            from .evolution import EvolutionModule
            self._evolution = EvolutionModule(agent_id=self.state.id)

    def _ensure_communication(self):
        """Lazy initialize communication module."""
        if self._communication is None:
            from .communication import CommunicationModule
            self._communication = CommunicationModule(agent_id=self.state.id, mailbox=self._mailbox)

    def _ensure_adaptation(self):
        """Lazy initialize adaptation module."""
        if self._adaptation is None:
            from .adaptation import AdaptationModule
            self._adaptation = AdaptationModule(agent_id=self.state.id)

    async def autonomous_step(self) -> dict[str, Any]:
        """
        Autonomous agent step - driven by curiosity and information gain.
        
        Agent decides what to explore based on:
        - Prediction error (world model failures)
        - Novelty (new/unfamiliar states)
        - Expected information gain
        - Current goals
        - Spatial exploration (movement)
        
        Flow: PERCEIVE → REASON → DECIDE → ACT
        """
        import random

        step_start = perf_counter()
        
        # 0. PERCEIVE - Read environment before deciding
        self._ensure_perception()
        
        # Build environment state for perception
        environment = {
            "location": {
                "x": self.state.location.x,
                "y": self.state.location.y,
                "z": self.state.location.z,
            },
            "agent_state": self.state.to_dict(),
            "nearby_agents": [],  # TODO: Query nearby agents
            "knowledge_sources": ["wikipedia", "web"],  # Available sources
            "available_topics": [],  # TODO: Generate from current context
        }
        
        # Create attention focus based on current goal
        attention_focus = None
        if self._goal_system and self.state.current_goal_id:
            from .perception import AttentionFocus, PerceptionModality
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal:
                # Focus perception on goal-relevant information
                attention_focus = AttentionFocus(
                    modalities=[
                        PerceptionModality.SPATIAL,
                        PerceptionModality.KNOWLEDGE,
                        PerceptionModality.SELF,
                    ],
                    keywords=[current_goal.description] if current_goal.description else [],
                    priority=current_goal.priority if hasattr(current_goal, 'priority') else 0.5,
                    goal_id=self.state.current_goal_id,
                )
        
        # Perceive environment
        perceptions = await self._perception.perceive(environment, attention_focus)
        perceptions, noise_reduction = self._perception.filter_perception(
            perceptions, attention_focus=attention_focus
        )
        self.state.metrics.perception_noise_reduction = noise_reduction
        
        # Get contextual awareness
        awareness = self._perception.get_contextual_awareness()
        
        # 1. REASON - Generate hypotheses from perceptions
        self._ensure_reasoning()
        
        # Extract observations from perceptions for hypothesis generation
        observations = []
        for p in perceptions[:5]:  # Top 5 most relevant perceptions
            observations.append({
                "modality": p.modality.value,
                "data": p.data,
                "confidence": p.confidence,
                "timestamp": p.timestamp,
            })
        
        # Generate hypothesis if we have enough observations
        if len(observations) >= 2:
            hypothesis = self._reasoning.generate_hypothesis(observations)
            if hypothesis:
                print(f"[REASON] Agent {self.state.name} hypothesis: {hypothesis.description} (confidence: {hypothesis.confidence:.2f})")
        
        # 2. Check if we should generate new goals
        if self._goal_system and (not self.state.current_goal_id or np.random.random() < 0.1):
            await self._maybe_generate_goal()
        
        # 3. Compute curiosity-driven exploration signal
        exploration_signal = self._compute_exploration_signal()
        
        # 4. DECIDE on action using DecisionModule
        self._ensure_decision()
        self._ensure_monitoring()
        
        # Build available actions
        from .decision import Action, Goal
        available_actions = [
            Action(type="move", params={"exploration": True}, estimated_cost=0.1),
            Action(type="learn", params={"curiosity_driven": True}, estimated_cost=0.2),
            Action(type="policy", params={"greedy": False}, estimated_cost=0.05),
        ]
        
        # Build goals from goal system
        goals = []
        if self._goal_system and self.state.current_goal_id:
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal:
                goals.append(Goal(
                    id=self.state.current_goal_id,
                    description=current_goal.description if hasattr(current_goal, 'description') else "Active goal",
                    priority=current_goal.priority if hasattr(current_goal, 'priority') else 0.7,
                ))

        # Balance exploration and learning goals
        # Both are equally important - agent should alternate between them
        # Use exploration_signal to add slight variation, but keep them close
        base_priority = 0.5
        signal_influence = 0.15  # Reduced from 0.4 to keep goals balanced
        random_noise = random.uniform(-0.05, 0.05)  # Small randomness for variety
        
        explore_priority = base_priority + exploration_signal * signal_influence + random_noise
        learn_priority = base_priority + (1.0 - exploration_signal) * signal_influence - random_noise
        
        # Clamp to valid range
        explore_priority = max(0.3, min(0.7, explore_priority))
        learn_priority = max(0.3, min(0.7, learn_priority))
        
        goals.append(Goal(
            id="explore_world",
            description="Explore new locations",
            priority=explore_priority,
        ))

        goals.append(Goal(
            id="increase_knowledge",
            description="Learn new knowledge",
            priority=learn_priority,
        ))
        
        # World state for decision-making
        world_state = {
            "location": (self.state.location.x, self.state.location.y),
            "knowledge_count": self.state.metrics.knowledge_items_learned,
            "exploration_signal": exploration_signal,
            "nearby_agents": [],  # TODO: Query nearby agents
        }
        
        # Get predictions from reasoning (formatted for DecisionModule)
        predictions_dict = {}
        for action in available_actions:
            pred = self._reasoning.predict(
                state=world_state,
                action=action.type,
                world_model=self._world_model,
            )
            predictions_dict[action.type] = {
                "predicted_outcome": pred.predicted_outcome,
                "confidence": pred.confidence,
            }
            # Log prediction
            print(
                f"[PREDICT] Agent {self.state.name} predicted {action.type}: "
                f"{pred.predicted_outcome} (confidence: {pred.confidence:.2f})"
            )
        
        # Get reasoning context (hypotheses)
        reasoning_context = {
            "hypotheses": self._reasoning.hypotheses,
            "causal_relations": self._reasoning.causal_relations,
        }
        
        # Make decision
        decision = self._decision.decide(
            available_actions=available_actions,
            goals=goals,
            world_state=world_state,
            predictions=predictions_dict,
            reasoning_context=reasoning_context,
        )
        
        action_type = decision.action.type
        print(f"[DECIDE] Agent {self.state.name} chose {action_type}: {decision.rationale} (confidence: {decision.confidence:.2f})")
        
        # Get the prediction for the chosen action
        prediction = self._reasoning.predictions.get(f"{action_type}_{len(self._reasoning.predictions)}", 
                                                      self._reasoning.predict(action_type, world_state, self._world_model))
        
        # Track decision distribution for researchers
        if action_type not in self.state.metrics.decision_distribution:
            self.state.metrics.decision_distribution[action_type] = 0
        self.state.metrics.decision_distribution[action_type] += 1
        
        # Update time alive
        elapsed = (datetime.now(UTC) - self.state.created_at).total_seconds() / 3600
        self.state.metrics.time_alive_hours = elapsed
        
        # Update developer metrics
        self.state.metrics.total_steps_executed += 1
        
        actual_outcome = {}  # Will populate based on action results
        
        if action_type == "move":
            # Spatial exploration - move to new location
            loc = self.state.location
            current_lat = loc.x if loc and loc.x != 0 else -25.0  # Default to Australia
            current_lon = loc.y if loc and loc.y != 0 else 134.0
            
            # Move 1-50km in random direction
            distance_km = random.uniform(1, 50)
            bearing = random.uniform(0, 360)
            new_lat, new_lon = self._calculate_destination(
                current_lat, current_lon, distance_km, bearing
            )
            
            # Update position
            self.set_earthlink_position(new_lat, new_lon, 0.0)
            
            # Update metrics with actual data
            self.state.metrics.distance_traveled_km += distance_km
            self.state.metrics.current_activity = f"Moving {distance_km:.1f}km"
            
            print(f"[MOVE] Agent {self.state.name} moved {distance_km:.1f}km to ({new_lat:.4f}, {new_lon:.4f})")
            
            # Capture actual outcome
            actual_outcome = {
                "location": (new_lat, new_lon),
                "distance_moved": distance_km,
                "exploration_signal": self._compute_exploration_signal(),  # New signal after move
            }
            
            # TODO: Track unique grid cells visited (e.g., 10km x 10km)
            # grid_cell = (int(new_lat / 0.1), int(new_lon / 0.1))
            # if grid_cell not in visited_cells:
            #     self.state.metrics.unique_locations_visited += 1
            
            # Set status
            self.set_status(AgentStatus.EXPLORING)
            
            action = {
                "type": "move",
                "from": (current_lat, current_lon),
                "to": (new_lat, new_lon),
                "distance_km": distance_km,
                "bearing": bearing,
                "position": (new_lat, new_lon, 0.0),
            }
            
        elif action_type == "learn":
            # High curiosity - acquire knowledge from external world
            threshold = self.config.get("exploration_threshold", 0.3)
            self.set_status(AgentStatus.LEARNING)
            action = await self._explore_knowledge(curiosity_signal=exploration_signal)
            
            # Capture actual outcome
            actual_outcome = {
                "knowledge_count": self.state.metrics.knowledge_items_learned,
                "learning_occurred": True,
                "exploration_signal": self._compute_exploration_signal(),
            }
            
        else:
            # Low curiosity - execute policy action in environment
            self.set_status(AgentStatus.EXECUTING)
            action = {
                "type": action_type,
                "action": getattr(decision.action, "params", {}),
            }
            
            # Capture actual outcome
            actual_outcome = {
                "policy_executed": True,
                "exploration_signal": self._compute_exploration_signal(),
            }
        
        # Update prediction with actual outcome
        if prediction and actual_outcome:
            prediction.actual_outcome = actual_outcome
            # Simple error calculation
            if prediction.predicted_outcome and actual_outcome:
                # Count differing keys
                pred_keys = set(prediction.predicted_outcome.keys())
                actual_keys = set(actual_outcome.keys())
                all_keys = pred_keys | actual_keys
                if all_keys:
                    different = sum(
                        1 for k in all_keys
                        if prediction.predicted_outcome.get(k) != actual_outcome.get(k)
                    )
                    prediction.prediction_error = different / len(all_keys)
        
        # Test any active hypotheses with this outcome
        if actual_outcome:
            # Find hypotheses related to this action type
            for hyp_id, hypothesis in self._reasoning.hypotheses.items():
                if action_type in hypothesis.description.lower():
                    # Use outcome as evidence
                    supporting = actual_outcome.get("exploration_signal", 0) > 0.5
                    self._reasoning.test_hypothesis(
                        hyp_id,
                        evidence=str(actual_outcome),
                        supporting=supporting
                    )
        
        # 5. Update curiosity score
        self.state.metrics.curiosity_score = exploration_signal

        # 6. Self-monitoring: record performance and diagnostics
        step_duration_ms = (perf_counter() - step_start) * 1000
        prediction_error = getattr(prediction, "prediction_error", None) if prediction else None
        decision_conf = decision.confidence if decision else 0.0
        goal_progress = 0.0
        if self._goal_system and self.state.current_goal_id:
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal and hasattr(current_goal, "progress"):
                goal_progress = getattr(current_goal, "progress", 0.0)

        self._monitoring.record_step(
            step_duration_ms=step_duration_ms,
            decision_confidence=decision_conf,
            prediction_error=prediction_error,
            success=True,
            status=self.state.status.value,
            action_type=action_type,
            goal_progress=goal_progress,
        )

        perf_report = self._monitoring.track_performance()
        diagnosis = self._monitoring.diagnose(perf_report)

        self.state.metrics.avg_step_duration_ms = perf_report.avg_step_duration_ms
        self.state.metrics.rolling_step_duration_ms = perf_report.avg_step_duration_ms
        self.state.metrics.avg_decision_confidence = perf_report.avg_decision_confidence
        self.state.metrics.failure_rate = perf_report.failure_rate
        self.state.metrics.performance_trend = perf_report.trend
        self.state.metrics.uncertainty_score = perf_report.uncertainty_score
        self.state.metrics.prediction_accuracy = perf_report.prediction_accuracy
        if prediction_error is not None:
            self.state.metrics.decision_accuracy = max(0.0, min(1.0, 1.0 - prediction_error))
        self.state.metrics.last_diagnosis = diagnosis.probable_cause
        self.state.metrics.goal_persistence = perf_report.goal_progress
        self.state.metrics.uncertainty_score = perf_report.uncertainty_score
        self.state.metrics.last_activity_timestamp = datetime.now(UTC).isoformat()

        # Memory consolidation cadence
        self._maybe_consolidate_memory()
        
        return action

    def get_monitoring_report(self) -> dict[str, Any]:
        """Expose monitoring report for external callers (UI/analytics)."""
        self._ensure_monitoring()
        return self._monitoring.get_report()

    # -------------------------------------------------------------------------
    # Output Generation
    # -------------------------------------------------------------------------

    async def generate_outputs(
        self,
        exploration_data: list[dict[str, Any]] | None = None,
        observations: list[dict[str, Any]] | None = None,
        hypotheses: list[Any] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        map_type: str = "route",
        summary_max_length: int = 240,
    ) -> dict[str, Any]:
        """
        Generate maps, summaries, and theories from recent agent data.
        
        Returns a dictionary containing any generated outputs.
        """
        self._ensure_generation()
        self._ensure_perception()
        self._ensure_reasoning()

        outputs: dict[str, Any] = {}

        # Default to recent perceptions for summarization if none provided
        if observations is None and getattr(self._perception, "perception_history", None):
            observations = [
                {
                    "modality": p.modality.value,
                    "data": p.data,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp,
                }
                for p in self._perception.perception_history[-25:]
            ]

        # Use perception data to seed exploration map when explicit data absent
        if exploration_data is None and observations:
            exploration_data = [
                obs.get("data", {}) for obs in observations
                if isinstance(obs.get("data"), dict)
                and ("latitude" in obs["data"] or "lat" in obs["data"])
                and ("longitude" in obs["data"] or "lon" in obs["data"])
            ]

        # Default hypotheses to current reasoning context
        if hypotheses is None and self._reasoning and getattr(self._reasoning, "hypotheses", None):
            hypotheses = list(self._reasoning.hypotheses.values())

        if exploration_data is not None:
            generated_map = await self._generation.generate_map(
                exploration_data=exploration_data,
                map_type=map_type,
            )
            outputs["map"] = generated_map.to_dict()

        if observations is not None:
            summary = await self._generation.create_summary(
                observations=observations,
                max_length=summary_max_length,
            )
            outputs["summary"] = summary.to_dict()

        if hypotheses:
            theory = await self._generation.formulate_theory(
                hypotheses=hypotheses,
                evidence=evidence or [],
            )
            outputs["theory"] = theory.to_dict()

        # Track latest activity timestamp for UI
        self.state.metrics.last_activity_timestamp = datetime.now(UTC).isoformat()

        print(
            f"[GENERATE] Agent {self.state.name} produced outputs: "
            f"{', '.join(outputs.keys()) if outputs else 'none'}"
        )

        return outputs

    # -------------------------------------------------------------------------
    # Specialization & Expertise
    # -------------------------------------------------------------------------

    async def analyze_specialization(
        self,
        activity_history: list[dict[str, Any]] | None = None,
        population_profiles: list[dict[str, Any]] | None = None,
        agent_population: list[UUID] | None = None,
    ) -> dict[str, Any]:
        """
        Detect specialization, estimate expertise, and suggest a niche.
        
        Args:
            activity_history: Optional explicit activity records with domains.
            population_profiles: Optional list of other agents' domains.
            agent_population: Optional list of agent IDs for saturation scoring.
        """
        self._ensure_specialization()

        history = activity_history or self._build_activity_history_from_metrics()
        profile = await self._specialization.detect_specialization(history)

        expertise = None
        if profile.domain:
            expertise = await self._specialization.evaluate_expertise(profile.domain)

        niche = await self._specialization.discover_niche(
            agent_population=agent_population or [],
            population_profiles=population_profiles or [],
        )

        report = {
            "profile": profile.to_dict(),
            "expertise": expertise.to_dict() if expertise else None,
            "niche": niche.to_dict() if niche else None,
        }

        print(
            f"[SPECIALIZE] Agent {self.state.name} specialization: "
            f"{profile.domain or 'none'} (confidence {profile.confidence:.2f})"
        )
        return report

    def _build_activity_history_from_metrics(self) -> list[dict[str, Any]]:
        """Construct lightweight activity history from existing metrics."""
        history: list[dict[str, Any]] = []
        now = datetime.now(UTC)

        # Use decision distribution as proxy for domains
        for action, count in self.state.metrics.decision_distribution.items():
            history.append(
                {
                    "domain": action,
                    "success": True,
                    "reward": 0.0,
                    "timestamp": now,
                    "count": count,
                }
            )

        # Add knowledge and exploration signals as pseudo domains
        if self.state.metrics.knowledge_items_learned > 0:
            history.append(
                {
                    "domain": "knowledge",
                    "success": True,
                    "reward": min(1.0, self.state.metrics.knowledge_items_learned / 10),
                    "timestamp": now,
                }
            )
        if self.state.metrics.distance_traveled_km > 0:
            history.append(
                {
                    "domain": "exploration",
                    "success": True,
                    "reward": min(1.0, self.state.metrics.distance_traveled_km / 100),
                    "timestamp": now,
                }
            )

        return history

    # -------------------------------------------------------------------------
    # Knowledge Transfer
    # -------------------------------------------------------------------------

    async def transfer_knowledge(
        self,
        domain: str,
        recipient_id: UUID | None = None,
        collective_id: str | None = None,
        observations: list[dict[str, Any]] | None = None,
        hypotheses: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Extract knowledge for a domain and transfer to a recipient or collective.
        """
        self._ensure_knowledge_transfer()
        self._ensure_reasoning()

        knowledge = await self._knowledge_transfer.extract_knowledge(
            domain=domain,
            observations=observations,
            hypotheses=hypotheses or (list(self._reasoning.hypotheses.values()) if self._reasoning else []),
        )

        result: dict[str, Any] = {"knowledge": knowledge.to_dict()}

        if recipient_id:
            transfer = await self._knowledge_transfer.transfer_to_agent(recipient_id, knowledge)
            result["transfer"] = transfer.to_dict()
        if collective_id:
            contributed = await self._knowledge_transfer.contribute_to_collective(
                knowledge=knowledge,
                collective_id=collective_id,
            )
            result["collective_contribution"] = contributed

        # Track activity timestamp
        self.state.metrics.last_activity_timestamp = datetime.now(UTC).isoformat()

        print(
            f"[KNOWLEDGE] Agent {self.state.name} shared knowledge on '{domain}' "
            f"to {recipient_id or collective_id or 'self'}"
        )
        return result

    # -------------------------------------------------------------------------
    # Modeling (world, agents, concepts)
    # -------------------------------------------------------------------------

    async def update_models(
        self,
        observations: list[dict[str, Any]] | None = None,
        interaction_history: list[dict[str, Any]] | None = None,
        concept_data: list[dict[str, Any]] | None = None,
        target_agent_id: UUID | None = None,
    ) -> dict[str, Any]:
        """
        Build/refresh world model, theory of mind, and concepts.
        """
        self._ensure_modeling()

        observations = observations or []
        world_model = await self._modeling.build_world_model(observations)

        agent_model = None
        if target_agent_id and interaction_history is not None:
            agent_model = await self._modeling.model_agent(target_agent_id, interaction_history)

        concept = None
        if concept_data:
            concept = await self._modeling.extract_concept(concept_data)

        state = self._modeling.get_state()
        report = {
            "world_model": world_model.to_dict() if world_model else None,
            "agent_model": agent_model.to_dict() if agent_model else None,
            "concept": concept.to_dict() if concept else None,
            "state": state.to_dict(),
        }

        self.state.metrics.last_activity_timestamp = datetime.now(UTC).isoformat()
        print(f"[MODEL] Agent {self.state.name} updated models.")
        return report

    # -------------------------------------------------------------------------
    # Evolution (replication, fitness, crossover)
    # -------------------------------------------------------------------------

    async def evolve(
        self,
        fitness_criteria: dict[str, float] | None = None,
        metrics: dict[str, float] | None = None,
        other_agent_id: UUID | None = None,
        mutation_rate: float = 0.05,
    ) -> dict[str, Any]:
        """
        Evaluate fitness and optionally spawn variants or crossover.
        """
        self._ensure_evolution()

        fitness_report = await self._evolution.evaluate_fitness(
            fitness_criteria=fitness_criteria or {"goals_achieved": 1.0, "knowledge_items_learned": 0.5},
            metrics=metrics or {
                "goals_achieved": self.state.metrics.goals_achieved,
                "knowledge_items_learned": self.state.metrics.knowledge_items_learned,
            },
        )

        variant_id = await self._evolution.self_replicate(mutation_rate=mutation_rate)
        crossover_children = []
        if other_agent_id:
            crossover_children = await self._evolution.crossover(other_agent_id)

        self.state.metrics.last_activity_timestamp = datetime.now(UTC).isoformat()
        print(f"[EVOLVE] Agent {self.state.name} fitness {fitness_report.score:.2f}, spawned {variant_id}.")

        return {
            "fitness": fitness_report.to_dict(),
            "replica_id": str(variant_id),
            "crossover_children": [str(cid) for cid in crossover_children],
        }

    # -------------------------------------------------------------------------
    # Communication (intent-based messaging)
    # -------------------------------------------------------------------------

    async def communicate(
        self,
        recipient_id: UUID | None = None,
        content: dict | None = None,
        intent: str = "direct",
        priority: int = 1,
        negotiation_agents: list[UUID] | None = None,
        negotiation_topic: dict | None = None,
        broadcast: bool = False,
    ) -> dict[str, Any]:
        """
        Send messages, negotiation proposals, or broadcasts.
        """
        self._ensure_communication()

        result = {}
        if broadcast:
            res = await self._communication.broadcast(content=content or {})
            result["broadcast"] = res.to_dict()
        elif negotiation_agents:
            res = await self._communication.negotiate(
                agents=negotiation_agents,
                negotiation_topic=negotiation_topic or {},
            )
            result["negotiation"] = res.to_dict()
        elif recipient_id:
            from .messaging import MessageType, MessagePriority

            res = await self._communication.send_message(
                recipient=recipient_id,
                content=content or {},
                intent=MessageType(intent) if intent in MessageType._value2member_map_ else MessageType.DIRECT,
                priority=MessagePriority(priority) if priority in MessagePriority._value2member_map_ else MessagePriority.NORMAL,
                subject=content.get("subject") if isinstance(content, dict) else "",
            )
            result["direct"] = res.to_dict()

        self.state.metrics.last_activity_timestamp = datetime.now(UTC).isoformat()
        print(f"[COMMUNICATE] Agent {self.state.name} communication event: {result.keys()}")
        return result

    # -------------------------------------------------------------------------
    # Adaptation (detect shifts, strategies, behavior patches)
    # -------------------------------------------------------------------------

    async def adapt(
        self,
        recent_observations: list[dict[str, Any]] | None = None,
        baseline_distribution: dict[str, Any] | None = None,
        performance_metrics: dict[str, float] | None = None,
        failure_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Detect distribution shifts, adapt strategy, and propose behavior patches.
        """
        self._ensure_adaptation()

        shift_detected = await self._adaptation.detect_distribution_shift(
            recent_observations=recent_observations or [],
            baseline_distribution=baseline_distribution or {},
        )

        strategy_library = {
            "conservative": Strategy(
                name="conservative",
                rationale="High failure rate; reduce risk.",
                confidence=0.7,
            ),
            "balanced": Strategy(
                name="balanced",
                rationale="Default balanced strategy.",
                confidence=0.6,
            ),
        }
        strategy = await self._adaptation.adapt_strategy(
            performance_metrics=performance_metrics or {},
            strategy_library=strategy_library,
        )

        behavior_patch = await self._adaptation.modify_behavior(
            failure_context=failure_context or {},
        )

        self.state.metrics.last_activity_timestamp = datetime.now(UTC).isoformat()
        print(f"[ADAPT] Agent {self.state.name} shift={shift_detected}, strategy={strategy.name}")

        return {
            "shift_detected": shift_detected,
            "strategy": strategy.to_dict(),
            "behavior_patch": behavior_patch.to_dict(),
        }

    def step(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Execute one step of the agent loop.
        
        Returns action to take in the environment.
        """
        # 1. Perceive - process raw observation
        encoded_obs = self._perceive(observation)

        # 2. Encode - update memory and latent state
        self._encode(encoded_obs)

        # 3. Decide - select action based on state + goals
        action = self._decide()

        # 4. Update metrics
        self.state.metrics.total_steps_executed += 1
        self.state.updated_at = datetime.now(UTC)

        return action

    def learn(self, transition: dict[str, Any]) -> dict[str, float]:
        """
        Learn from a transition (s, a, r, s', done).
        
        Returns loss metrics.
        """
        self.set_status(AgentStatus.LEARNING)
        
        # Initialize advanced learning module
        self._ensure_learning()
        
        # Store in episodic memory
        if self._memory:
            self._memory.store_episode(transition)

        # Sample batch and update networks
        losses = {}

        if self._policy and self._memory:
            batch = self._memory.sample_for_learning(batch_size=32)
            if batch:
                losses = self._update_networks(batch)

        if losses:
            mean_loss = float(sum(losses.values()) / max(len(losses), 1))
            # Update training metrics
            self.state.metrics.training_updates += 1
            
            # Adapt learning rate based on performance
            if hasattr(self.state, 'recent_rewards'):
                recent_perf = self.state.recent_rewards[-10:] if len(self.state.recent_rewards) >= 10 else []
                if recent_perf:
                    adapted_lr = self._learning.adapt_learning_rate(
                        task_id="reinforcement_learning",
                        recent_performance=recent_perf,
                    )
                    print(f"[LEARN] Adapted learning rate to {adapted_lr:.4f}")

        return losses

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _perceive(self, observation: dict[str, Any]) -> torch.Tensor:
        """Process raw observation into tensor."""
        # TODO: Implement proper perception based on observation type
        # For now, just create a dummy tensor
        if "vector" in observation:
            return torch.tensor(observation["vector"], dtype=torch.float32, device=self.device)
        return torch.zeros(64, dtype=torch.float32, device=self.device)

    def _encode(self, observation: torch.Tensor) -> None:
        """Update memory and latent state from observation."""
        # Add to short-term memory
        if self._memory:
            self._memory.add_short_term(observation)

        # Update latent state (simple for now)
        # In full implementation, this uses the encoder network
        self.state.latent_state = observation

    def _maybe_consolidate_memory(self) -> None:
        """Periodically consolidate memory tiers and update metrics."""
        if not self._memory:
            return

        # Consolidate every 5 autonomous steps
        if self.state.metrics.total_steps_executed - self._last_memory_consolidation_step < 5:
            return

        result = self._memory.consolidate()
        self._last_memory_consolidation_step = self.state.metrics.total_steps_executed

        # Update memory metrics
        if hasattr(self._memory, "long_term"):
            try:
                self.state.metrics.memory_entries = self._memory.long_term.count()
            except Exception:
                pass
        if isinstance(result, dict):
            self.state.metrics.memory_entries += result.get("short_to_long", 0)
            self.state.metrics.memory_entries += result.get("episodic_to_semantic", 0)

    async def _maybe_generate_goal(self) -> None:
        """Generate new goal if needed (using GoalSystem)."""
        if not self._goal_system:
            return
        
        # Propose new goals based on current state
        state = self.state.latent_state
        if state is None:
            return
        
        # Use goal proposer to generate candidate goals
        with torch.no_grad():
            goal_embeddings = self._goal_system.goal_proposer(state.unsqueeze(0))  # [1, num_proposals, goal_dim]
            num_proposals = goal_embeddings.shape[1]
            
            # Expand state to match number of proposals
            expanded_state = state.unsqueeze(0).expand(num_proposals, -1)  # [num_proposals, state_dim]
            goal_values = self._goal_system.goal_value(
                expanded_state,
                goal_embeddings.squeeze(0)  # [num_proposals, goal_dim]
            )
        
        # Select highest value goal
        best_idx = goal_values.argmax()
        best_goal_embedding = goal_embeddings[0, best_idx]
        
        # Create goal from embedding
        from ..goals import Goal, GoalType
        new_goal = Goal(
            goal_type=GoalType.EXPLORATION,
            description=f"Explore state space (novelty: {goal_values[best_idx].item():.3f})",
            embedding=best_goal_embedding,
        )
        
        self._goal_system.add_goal(new_goal)
        self.state.current_goal_id = new_goal.id

    def _compute_exploration_signal(self) -> float:
        """
        Compute exploration signal based on curiosity metrics.
        
        Returns value in [0, 1] indicating strength of exploration drive.
        Higher values = agent wants to explore/learn.
        """
        if not self._curiosity or not self._world_model:
            return np.random.random() * 0.5  # Default to moderate exploration
        
        state = self.state.latent_state
        if state is None:
            return 0.5
        
        # Get prediction error from world model
        with torch.no_grad():
            # Simulate dummy action
            dummy_action = torch.zeros(self.config.get("action_dim", 64), device=self.device)
            
            # Compute intrinsic reward (prediction error + novelty)
            intrinsic_reward = self._curiosity.compute_intrinsic_reward(
                state.unsqueeze(0),
                dummy_action.unsqueeze(0),
                state.unsqueeze(0),  # Next state = current (no transition yet)
            )
            
            # Normalize to [0, 1]
            signal = torch.sigmoid(intrinsic_reward).item()
        
        # Update curiosity score from neural network
        self.state.metrics.curiosity_score = signal
        
        return signal

    async def _explore_knowledge(self, curiosity_signal: float | None = None) -> dict[str, Any]:
        """
        Execute knowledge exploration action.
        
        Agent decides what topic to explore based on curiosity and goals.
        """
        # Get current goal to inform exploration
        goal_topic = None
        goal_achieved = False
        goal_context = None
        
        if self._goal_system and self.state.current_goal_id:
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal:
                goal_context = current_goal.description
                # Parse goal description for topic
                desc = current_goal.description
                if "Explore" in desc or "Learn about" in desc:
                    # Extract topic from goal description
                    goal_topic = desc.split("Explore")[-1].split("Learn about")[-1].strip()
                    # Strip novelty score if present (e.g., "state space (novelty: 0.480)" -> skip it)
                    if "(" in goal_topic and "novelty" in goal_topic:
                        goal_topic = None  # Skip auto-generated goals, use _sample_exploration_topic instead
                elif desc not in ["Explore state space", "Auto-proposed goal"]:
                    goal_topic = desc
        
        # Generate exploration topic
        if goal_topic and goal_topic not in ["state space", "Auto-proposed goal"]:
            topic = goal_topic
        else:
            # Sample from knowledge frontier
            topic = await self._sample_exploration_topic()
            
            # Create new goal for this exploration
            if self._goal_system:
                from ..goals import Goal, GoalType
                new_goal = Goal(
                    goal_type=GoalType.EXPLORATION,
                    description=f"Learn about {topic}",
                    priority=0.7,
                    intrinsic_value=0.7,
                    source="curiosity",
                )
                goal = self._goal_system.add_goal(new_goal)
                self.state.current_goal_id = goal.id
        
        # Explore the topic
        results = await self.explore_topic(topic)
        
        # Update metrics with actual learning data
        knowledge_gained = results.get("knowledge_gained", 0)
        
        if knowledge_gained > 0:
            self.state.metrics.knowledge_items_learned += knowledge_gained
            self.state.metrics.current_activity = f"Learning about {topic}"
        
        # Calculate learning rate (items per hour)
        if self.state.metrics.time_alive_hours > 0:
            self.state.metrics.learning_rate = (
                self.state.metrics.knowledge_items_learned / self.state.metrics.time_alive_hours
            )
        
        # Check if goal achieved (gained knowledge)
        if results["knowledge_gained"] > 0 and self._goal_system and self.state.current_goal_id:
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal and self.state.latent_state is not None:
                # Update goal progress based on state
                self._goal_system.update_goal_progress(
                    self.state.current_goal_id,
                    self.state.latent_state,
                )
                
                # Check if goal completed
                if current_goal.progress >= 1.0:
                    from ..goals import GoalStatus
                    current_goal.status = GoalStatus.ACHIEVED
                    current_goal.completed_at = datetime.now(UTC)
                    goal_achieved = True
                    # Clear current goal to allow new goal generation
                    self.state.current_goal_id = None

        # If we gained knowledge, write it into memory and trigger learning update
        if results["knowledge_gained"] > 0:
            intrinsic_reward = self._compute_exploration_signal()
            total_reward = float(results["knowledge_gained"] + intrinsic_reward)

            if self._memory:
                # Persist semantic/long-term memory
                content_summary = str(results["sources"])[:500]
                try:
                    await self._memory.store_semantic(
                        content=f"Topic: {topic}\nSources: {', '.join(results['sources'].keys())}\nSummary: {content_summary}",
                        metadata={"source": "knowledge_exploration", "topic": topic},
                    )
                except Exception as e:
                    print(f"[Agent {self.state.id}] Failed to store semantic memory: {e}")

                # Create a simple transition for episodic/replay
                transition = {
                    "state": self.state.latent_state.detach().cpu() if hasattr(self.state.latent_state, "detach") else self.state.latent_state,
                    "action": torch.zeros(self.config.get("action_dim", 64)),
                    "reward": total_reward,
                    "next_state": self.state.latent_state.detach().cpu() if hasattr(self.state.latent_state, "detach") else self.state.latent_state,
                    "done": False,
                    "goal": topic,
                    "info": {
                        "sources": list(results["sources"].keys()),
                        "knowledge_gained": results["knowledge_gained"],
                        "intrinsic_reward": intrinsic_reward,
                    },
                }
                # Learning step (stores transition + optional network update)
                self.learn(transition)
        
        return {
            "type": "knowledge_exploration",
            "topic": topic,
            "knowledge_gained": results["knowledge_gained"],
            "sources_used": list(results["sources"].keys()),
            "goal_achieved": goal_achieved,
        }

    async def _sample_exploration_topic(self) -> str:
        """
        Sample a topic from the knowledge frontier.
        
        Agent decides what it's curious about based on:
        1. Current location context (POIs, landmarks, features nearby)
        2. Recent topics explored (memory)
        3. Generic Earth-related domains (fallback)
        """
        # FIRST: Use actual location to drive curiosity
        # Query nearby features to find what's interesting around the agent
        # Note: Earthlink position stored separately, not in state.location
        lat = self.get_latitude()
        lon = self.get_longitude()
        
        if lat is not None and lon is not None:
            
            try:
                # Import here to avoid circular dependency
                from src.data.worlds.base.earth import GeoSource
                
                # Get location context (nearby POIs, regions, features)
                geo = GeoSource()
                context = await geo.get_location_context(lat, lon, radius_meters=5000)
                
                # Extract interesting topics from location
                topics = []
                
                # Named features (landmarks, POIs)
                for feature in context.get("nearby_features", []):
                    name = feature.get("name")
                    if name and len(name) > 3:  # Skip abbreviations
                        topics.append(name)
                
                # Containing regions (cities, countries)
                for region in context.get("containing_regions", []):
                    name = region.get("name")
                    if name and len(name) > 3:
                        topics.append(name)
                
                # Nearby place names
                for place in context.get("nearby_places", []):
                    name = place.get("name")
                    if name and len(name) > 3:
                        topics.append(name)
                
                # If we found location-based topics, use them!
                if topics:
                    print(f"[Agent {self.state.id}] Location-based topics: {topics[:5]}")
                    return np.random.choice(topics)
            except Exception as e:
                print(f"[Agent {self.state.id}] Failed to get location context for topic generation: {e}")
        
        # SECOND: Try to query semantic memory for topics with low coverage
        if self._memory and hasattr(self._memory, 'query_semantic'):
            try:
                recent_topics = await self._memory.query_semantic(
                    query="recent topics explored",
                    k=10
                )
                
                # If we have memory, explore related but novel topics
                if recent_topics:
                    # Use Ollama to generate related topic
                    last_topic = recent_topics[0].get("metadata", {}).get("topic", "science")
                    prompt = f"Given the topic '{last_topic}', suggest one related but novel topic to explore. Reply with just the topic name, no explanation."
                    
                    try:
                        response = await self.query_ollama(prompt, model="llama2")
                        return response.get("response", "").strip()
                    except:
                        pass
            except Exception:
                pass  # Fall through to default domains
        
        # THIRD: Fallback to generic Earth-related exploration domains
        # (Only if location and memory queries both failed)
        domains = [
            "ocean ecosystems", "rainforest biodiversity", "mountain formation",
            "ancient civilizations", "climate patterns", "tectonic plates",
            "coastal erosion", "urban development", "agricultural practices",
            "water cycles", "wildlife migration", "desert ecosystems",
            "polar ice caps", "river systems", "volcanic activity",
            "human geography", "cultural diversity", "natural resources",
        ]
        print(f"[Agent {self.state.id}] Using fallback generic topic (location query failed)")
        return np.random.choice(domains)

    def _get_current_observation(self) -> dict[str, Any]:
        """Get current observation from environment/world."""
        # Return current state as observation
        state = self.state.latent_state
        if state is not None:
            return {"vector": state.cpu().numpy().tolist()}
        return {"vector": [0.0] * 256}

    def _decide(self) -> dict[str, Any]:
        """Select action based on current state and goals."""
        # Get current goal embedding
        goal_embedding = self._get_current_goal()

        # Prepare state for policy
        state = self.state.latent_state
        if state is None:
            return {"type": "idle", "action": None}

        # Concatenate state and goal for goal-conditioned policy
        if goal_embedding is not None:
            policy_input = torch.cat([state, goal_embedding], dim=-1)
        else:
            goal_dim = self.config.get("goal_dim", 32)
            policy_input = torch.cat([state, torch.zeros(goal_dim, device=self.device)], dim=-1)

        # Select action from policy
        if self._policy:
            with torch.no_grad():
                deterministic = self.state.lifecycle == AgentLifecycle.ACTS
                action, log_prob, entropy, value = self._policy.get_action_and_value(
                    policy_input.unsqueeze(0),
                    deterministic=deterministic,
                )
                action = action.squeeze(0)

            return {
                "type": "action",
                "action": action.cpu().numpy().tolist() if action.dim() > 0 else action.item(),
                "log_prob": log_prob.item(),
                "value": value.item(),
            }
        else:
            # Default: no-op action
            return {"type": "idle", "action": None}

    def _get_current_goal(self) -> torch.Tensor | None:
        """Get current goal embedding."""
        if self._goal_system and self.state.latent_state is not None:
            return self._goal_system.get_current_goal_embedding(self.state.latent_state)
        return None

    def _update_networks(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Update neural networks from experience batch (PPO-style)."""
        losses = {}

        # Unpack batch
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        weights = batch.get("weights", torch.ones_like(rewards))

        # -------------------------------------------------------------------------
        # 1. World Model Update
        # -------------------------------------------------------------------------
        if self._world_model:
            world_model_losses = self._world_model.compute_loss(
                states, actions, next_states, rewards, dones
            )
            
            self._world_model_optimizer.zero_grad()
            world_model_losses["total_loss"].backward()
            self._world_model_optimizer.step()
            
            losses["world_model_loss"] = world_model_losses["total_loss"].item()

        # -------------------------------------------------------------------------
        # 2. Curiosity Module Update
        # -------------------------------------------------------------------------
        if self._curiosity:
            curiosity_losses = self._curiosity.update(states, actions, next_states)
            
            self._curiosity_optimizer.zero_grad()
            curiosity_losses["total_loss"].backward()
            self._curiosity_optimizer.step()
            
            losses["curiosity_loss"] = curiosity_losses["total_loss"].item()
            
            # Compute intrinsic reward
            with torch.no_grad():
                intrinsic_reward = self._curiosity.compute_intrinsic_reward(
                    states, actions, next_states
                )
                # Combine with extrinsic reward
                combined_rewards = rewards + self.config.get("curiosity_coef", 0.3) * intrinsic_reward
        else:
            combined_rewards = rewards

        # -------------------------------------------------------------------------
        # 3. Policy Update (PPO-style)
        # -------------------------------------------------------------------------
        if self._policy:
            # Get goal embeddings for conditioning
            goal_dim = self.config.get("goal_dim", 32)
            goal_embeddings = torch.zeros(states.shape[0], goal_dim, device=self.device)
            
            # Prepare policy input
            policy_states = torch.cat([states, goal_embeddings], dim=-1)
            
            # Get current values and old log probs
            with torch.no_grad():
                _, old_log_probs = self._policy.policy.get_action(policy_states) if hasattr(self._policy, 'policy') else (None, None)
                values = self._policy.get_value(policy_states)
            
            # Compute advantages (simple version)
            next_policy_states = torch.cat([next_states, goal_embeddings], dim=-1)
            with torch.no_grad():
                next_values = self._policy.get_value(next_policy_states)
            
            advantages = combined_rewards + 0.99 * next_values * (1 - dones) - values
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            _, log_probs, entropy, new_values = self._policy.evaluate_actions(policy_states, actions)
            
            # Value loss
            value_loss = 0.5 * ((new_values - returns) ** 2 * weights).mean()
            
            # Policy loss (simplified, no clipping for now)
            policy_loss = -(log_probs * advantages.detach() * weights).mean()
            entropy_loss = -entropy.mean()
            
            total_policy_loss = (
                policy_loss
                + self.config.get("value_coef", 0.5) * value_loss
                + self.config.get("entropy_coef", 0.01) * entropy_loss
            )
            
            self._policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
            self._policy_optimizer.step()
            
            losses["policy_loss"] = policy_loss.item()
            losses["value_loss"] = value_loss.item()
            losses["entropy"] = entropy.mean().item()

        # Update replay priorities if using prioritized replay
        if "indices" in batch and hasattr(self._memory, "update_replay_priorities"):
            td_errors = (advantages ** 2).detach().cpu().numpy()
            self._memory.update_replay_priorities(batch["indices"], td_errors)

        return losses

    # -------------------------------------------------------------------------
    # Component Initialization
    # -------------------------------------------------------------------------

    def initialize_components(
        self,
        memory_config: dict[str, Any] | None = None,
        policy_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize agent components (memory, policy, etc.)."""
        from ..memory import MemorySystem
        from ..networks import ActorCritic, WorldModel, ICM
        from ..goals import GoalSystem

        # Configuration
        state_dim = self.config.get("state_dim", 256)
        action_dim = self.config.get("action_dim", 64)
        goal_dim = self.config.get("goal_dim", 32)
        hidden_dim = self.config.get("hidden_dim", 256)
        continuous_actions = self.config.get("continuous_actions", False)

        # Initialize memory system
        self._memory = MemorySystem(config=memory_config or {})

        # Initialize policy (Actor-Critic)
        self._policy = ActorCritic(
            state_dim=state_dim + goal_dim,  # State + goal conditioning
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            continuous=continuous_actions,
        ).to(self.device)

        # Initialize world model
        self._world_model = WorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Initialize curiosity module
        self._curiosity = ICM(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Initialize goal system
        self._goal_system = GoalSystem(
            state_dim=state_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )

        # Optimizers
        self._policy_optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=self.config.get("policy_lr", 3e-4),
        )
        self._world_model_optimizer = torch.optim.Adam(
            self._world_model.parameters(),
            lr=self.config.get("world_model_lr", 1e-3),
        )
        self._curiosity_optimizer = torch.optim.Adam(
            self._curiosity.parameters(),
            lr=self.config.get("curiosity_lr", 1e-3),
        )

        # Transition to learning stage
        self.set_lifecycle(AgentLifecycle.LEARNS)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save agent state and networks to checkpoint."""
        checkpoint = {
            "state": self.state.to_dict(),
            "config": self.config,
        }

        if self._policy:
            checkpoint["policy_state_dict"] = self._policy.state_dict()
        if self._world_model:
            checkpoint["world_model_state_dict"] = self._world_model.state_dict()
        if self._curiosity:
            checkpoint["curiosity_state_dict"] = self._curiosity.state_dict()
        if self._goal_system:
            checkpoint["goal_system_state"] = self._goal_system.save_state()
            checkpoint["goal_encoder_state_dict"] = self._goal_system.goal_encoder.state_dict()
            checkpoint["goal_proposer_state_dict"] = self._goal_system.goal_proposer.state_dict()
            checkpoint["goal_value_state_dict"] = self._goal_system.goal_value.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent state and networks from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Restore config
        self.config = checkpoint.get("config", {})

        # Restore networks if they exist
        if self._policy and "policy_state_dict" in checkpoint:
            self._policy.load_state_dict(checkpoint["policy_state_dict"])
        if self._world_model and "world_model_state_dict" in checkpoint:
            self._world_model.load_state_dict(checkpoint["world_model_state_dict"])
        if self._curiosity and "curiosity_state_dict" in checkpoint:
            self._curiosity.load_state_dict(checkpoint["curiosity_state_dict"])
        if self._goal_system and "goal_system_state" in checkpoint:
            self._goal_system.load_state(checkpoint["goal_system_state"])

    async def _log_knowledge_acquisition(
        self,
        topic: str,
        sources: list[str],
        knowledge_count: int,
        content_summary: str,
        goal_context: str | None = None,
        curiosity_signal: float | None = None,
    ) -> None:
        """Log knowledge acquisition to database with full context (WHAT, WHEN, WHY, WHERE, QUALITY)."""
        try:
            from db.database import async_session_maker
        except ModuleNotFoundError:
            # In lightweight/test environments without DB, skip logging
            return
        from sqlalchemy import text
        import traceback
        
        try:
            async with async_session_maker() as db:
                query = text("""
                    INSERT INTO knowledge_acquisition_log 
                    (agent_id, source, topic, content_summary, knowledge_count, 
                     goal_context, curiosity_signal, lat, lon, agent_state, metadata)
                    VALUES (:agent_id, :source, :topic, :content_summary, :knowledge_count,
                            :goal_context, :curiosity_signal, :lat, :lon, :agent_state, :metadata)
                """)
                
                # Get current position
                lat = self.latitude
                lon = self.longitude
                
                await db.execute(query, {
                    "agent_id": str(self.state.id),
                    "source": ", ".join(sources),
                    "topic": topic,
                    "content_summary": content_summary,
                    "knowledge_count": knowledge_count,
                    "goal_context": goal_context,
                    "curiosity_signal": curiosity_signal,
                    "lat": lat,
                    "lon": lon,
                    "agent_state": self.state.lifecycle.value,
                    "metadata": "{}"
                })
                await db.commit()
        except Exception as e:
            # Make error visible - raise it so runner can catch and log
            error_msg = f"Failed to log knowledge acquisition for agent {self.state.id}: {e}\n{traceback.format_exc()}"
            raise RuntimeError(error_msg)

    # -------------------------------------------------------------------------
    # Spatial Movement & Navigation (15.6M OSM Features)
    # -------------------------------------------------------------------------

    async def move_to(self, lat: float, lon: float, altitude: float = 0.0) -> dict[str, Any]:
        """
        Move agent to a specific location in Earthlink.
        
        Args:
            lat: Target latitude (-90 to 90)
            lon: Target longitude (-180 to 180)
            altitude: Target altitude in meters (default 0)
            
        Returns:
            Movement result with new location and nearby context
        """
        # Validate coordinates
        lat = max(-90.0, min(90.0, lat))
        lon = max(-180.0, min(180.0, lon))
        
        # Store old position
        old_lat, old_lon = self.latitude, self.longitude
        
        # Update position
        self.set_earthlink_position(lat, lon, altitude)
        
        # Calculate distance moved
        distance_km = self._haversine_distance(old_lat, old_lon, lat, lon)
        
        # Query geo context at new location (skip if geo module not available)
        context = {}
        try:
            context = await self.query_geo(lat, lon, radius_meters=1000)
        except (ImportError, ModuleNotFoundError):
            # Geo module not available, skip context query
            pass
        
        # Update metrics
        self.state.metrics.total_steps_executed += 1
        
        return {
            "type": "movement",
            "old_position": {"lat": old_lat, "lon": old_lon},
            "new_position": {"lat": lat, "lon": lon, "altitude": altitude},
            "distance_km": distance_km,
            "nearby_features": len(context.get("nearby_features", [])),
            "containing_regions": [r.get("name") for r in context.get("containing_regions", [])[:3]],
        }

    async def explore_random_location(self, region_bounds: dict[str, float] | None = None) -> dict[str, Any]:
        """
        Move to a random location and explore it.
        
        Args:
            region_bounds: Optional bounds {"lat_min", "lat_max", "lon_min", "lon_max"}
                          Default: Australia bounds (OSM data coverage)
                          
        Returns:
            Exploration result with movement + knowledge gained
        """
        # Default to Australia/Oceania bounds (where we have OSM data)
        bounds = region_bounds or {
            "lat_min": -47.0,  # Southern Australia
            "lat_max": -10.0,  # Northern Australia
            "lon_min": 110.0,  # Western Australia
            "lon_max": 180.0,  # Eastern edge of Oceania
        }
        
        # Sample random location
        target_lat = np.random.uniform(bounds["lat_min"], bounds["lat_max"])
        target_lon = np.random.uniform(bounds["lon_min"], bounds["lon_max"])
        
        # Move to location
        movement = await self.move_to(target_lat, target_lon)
        
        # Explore location (geo + knowledge)
        exploration = await self.explore_location(target_lat, target_lon)
        
        return {
            "type": "random_exploration",
            "movement": movement,
            "exploration": exploration,
        }

    async def navigate_to_poi(self, poi_name: str, poi_type: str | None = None) -> dict[str, Any]:
        """
        Navigate to a specific point of interest by name.
        
        Args:
            poi_name: Name of the POI (e.g., "Sydney Opera House")
            poi_type: Optional filter by type (e.g., "building", "place")
            
        Returns:
            Navigation result with path and final location
        """
        from src.data.worlds.base.earth import get_geo_source
        from db.database import async_session_maker
        from sqlalchemy import text
        
        # Search for POI in database
        async with async_session_maker() as db:
            query_str = """
                SELECT osm_id, name, poi_type, 
                       ST_Y(geom::geometry) as lat, 
                       ST_X(geom::geometry) as lon
                FROM pois 
                WHERE name ILIKE :name
            """
            if poi_type:
                query_str += " AND poi_type = :poi_type"
            query_str += " LIMIT 1"
            
            params = {"name": f"%{poi_name}%"}
            if poi_type:
                params["poi_type"] = poi_type
                
            result = await db.execute(text(query_str), params)
            poi = result.fetchone()
        
        if not poi:
            return {
                "type": "navigation",
                "success": False,
                "error": f"POI '{poi_name}' not found",
            }
        
        # Move to POI
        target_lat, target_lon = poi.lat, poi.lon
        movement = await self.move_to(target_lat, target_lon)
        
        # Explore the POI location
        exploration = await self.explore_location(target_lat, target_lon)
        
        return {
            "type": "navigation",
            "success": True,
            "poi": {
                "name": poi.name,
                "type": poi.poi_type,
                "lat": target_lat,
                "lon": target_lon,
            },
            "movement": movement,
            "exploration": exploration,
        }

    async def find_nearest(
        self,
        feature_type: str,
        max_distance_km: float = 10.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find nearest features of a specific type.
        
        Args:
            feature_type: Type to search for (buildings, roads, pois, water, landcover, places)
            max_distance_km: Maximum search distance
            limit: Max results to return
            
        Returns:
            List of nearby features with distance and details
        """
        from db.database import async_session_maker
        from sqlalchemy import text
        
        # Map feature type to table
        table_map = {
            "building": "buildings",
            "buildings": "buildings",
            "road": "roads",
            "roads": "roads",
            "poi": "pois",
            "pois": "pois",
            "place": "places",
            "places": "places",
            "water": "water_features",
            "landcover": "land_cover",
            "land": "land_cover",
        }
        
        table = table_map.get(feature_type.lower())
        if not table:
            return []
        
        # Determine geometry column name
        geom_col = "footprint" if table == "buildings" else "geom"
        
        # Query nearest features
        async with async_session_maker() as db:
            query = text(f"""
                SELECT 
                    name,
                    ST_Y(ST_Centroid({geom_col}::geometry)) as lat,
                    ST_X(ST_Centroid({geom_col}::geometry)) as lon,
                    ST_Distance({geom_col}::geography, ST_Point(:lon, :lat)::geography) as distance_m
                FROM {table}
                WHERE ST_DWithin(
                    {geom_col}::geography,
                    ST_Point(:lon, :lat)::geography,
                    :max_distance_m
                )
                ORDER BY distance_m
                LIMIT :limit
            """)
            
            result = await db.execute(query, {
                "lat": self.latitude,
                "lon": self.longitude,
                "max_distance_m": max_distance_km * 1000,
                "limit": limit,
            })
            features = result.fetchall()
        
        return [
            {
                "name": f.name or "Unnamed",
                "lat": float(f.lat),
                "lon": float(f.lon),
                "distance_m": float(f.distance_m),
                "distance_km": float(f.distance_m) / 1000,
            }
            for f in features
        ]

    async def plan_exploration_route(
        self,
        num_waypoints: int = 5,
        region_bounds: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Plan a multi-waypoint exploration route.
        
        Args:
            num_waypoints: Number of waypoints to visit
            region_bounds: Optional bounds for route planning
            
        Returns:
            List of waypoints with planned route
        """
        # Default to Australia/Oceania bounds
        bounds = region_bounds or {
            "lat_min": -47.0,
            "lat_max": -10.0,
            "lon_min": 110.0,
            "lon_max": 180.0,
        }
        
        waypoints = []
        current_lat, current_lon = self.latitude, self.longitude
        
        for i in range(num_waypoints):
            # Sample next waypoint (bias towards exploration frontier)
            if i == 0:
                # First waypoint: random in bounds
                next_lat = np.random.uniform(bounds["lat_min"], bounds["lat_max"])
                next_lon = np.random.uniform(bounds["lon_min"], bounds["lon_max"])
            else:
                # Subsequent waypoints: random walk from current
                step_size_km = 50.0  # 50km steps
                bearing = np.random.uniform(0, 360)
                next_lat, next_lon = self._destination_point(
                    current_lat, current_lon, step_size_km, bearing
                )
                
                # Clip to bounds
                next_lat = np.clip(next_lat, bounds["lat_min"], bounds["lat_max"])
                next_lon = np.clip(next_lon, bounds["lon_min"], bounds["lon_max"])
            
            distance = self._haversine_distance(current_lat, current_lon, next_lat, next_lon)
            
            waypoints.append({
                "index": i,
                "lat": next_lat,
                "lon": next_lon,
                "distance_from_previous_km": distance,
            })
            
            current_lat, current_lon = next_lat, next_lon
        
        return waypoints

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points (Haversine formula).
        
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c

    def _destination_point(
        self,
        lat: float,
        lon: float,
        distance_km: float,
        bearing_deg: float,
    ) -> tuple[float, float]:
        """
        Calculate destination point given distance and bearing.
        
        Args:
            lat: Starting latitude
            lon: Starting longitude
            distance_km: Distance to travel (km)
            bearing_deg: Bearing in degrees (0-360, 0=North)
            
        Returns:
            (dest_lat, dest_lon) tuple
        """
        from math import radians, sin, cos, asin, atan2, degrees
        
        R = 6371  # Earth radius in km
        
        lat1 = radians(lat)
        lon1 = radians(lon)
        bearing = radians(bearing_deg)
        
        lat2 = asin(
            sin(lat1) * cos(distance_km / R) +
            cos(lat1) * sin(distance_km / R) * cos(bearing)
        )
        
        lon2 = lon1 + atan2(
            sin(bearing) * sin(distance_km / R) * cos(lat1),
            cos(distance_km / R) - sin(lat1) * sin(lat2)
        )
        
        return degrees(lat2), degrees(lon2)
    # -------------------------------------------------------------------------
    # Autonomous Behavior
    # -------------------------------------------------------------------------


# Ray actor wrapper for concurrent execution.
AgentActor = ray.remote(Agent)

__all__ = ["Agent", "AgentActor"]
