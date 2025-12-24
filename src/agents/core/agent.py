"""Core Agent class - autonomous intelligent actor."""

from datetime import datetime
from typing import Any
from uuid import UUID

import ray
import torch

from .messaging import Mailbox, Message, MessagePriority, MessageType
from .state import AgentLifecycle, AgentState


@ray.remote
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

        # Components (lazy initialized)
        self._memory: Any = None
        self._policy: Any = None
        self._world_model: Any = None
        self._curiosity: Any = None
        self._goal_system: Any = None

        # Messaging
        self._mailbox = Mailbox(agent_id=self.state.id)
        self._outgoing_messages: list[Message] = []
        
        # Optimizers
        self._policy_optimizer: Any = None
        self._world_model_optimizer: Any = None
        self._curiosity_optimizer: Any = None

        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return serializable agent state."""
        return self.state.to_dict()

    def get_id(self) -> str:
        """Return agent ID as string."""
        return str(self.state.id)

    def set_lifecycle(self, lifecycle: AgentLifecycle) -> None:
        """Transition agent lifecycle state."""
        self.state.lifecycle = lifecycle
        self.state.updated_at = datetime.utcnow()

    def set_target_world(self, world: str) -> None:
        """Set target world for specialization/deployment."""
        self.state.target_world = world
        self.state.updated_at = datetime.utcnow()

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
            recipient_id=UUID(recipient_id) if recipient_id else None,
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
        from ...knowledge import WikipediaSource
        
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
        from ...knowledge import OllamaSource
        
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

    async def search_web(self, query: str, provider: str = "duckduckgo", **kwargs) -> dict[str, Any]:
        """Search the web for information."""
        from ...knowledge import SearchSource
        
        search = SearchSource()
        
        # Search using specified provider
        if provider == "serper" and search.serper:
            results = await search.serper_search(query, **kwargs)
        elif provider == "tavily" and search.tavily:
            results = await search.tavily_search(query, **kwargs)
        else:
            results = await search.duckduckgo_search(query, **kwargs)
        
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
        from ...knowledge import RedditSource
        
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

    async def query_twitter(self, query: str, **kwargs) -> dict[str, Any]:
        """Query Twitter/X for recent discussions."""
        from ...knowledge import TwitterSource
        
        twitter = TwitterSource()
        
        # Search recent tweets
        tweets = await twitter.search_recent_tweets(query, **kwargs)
        
        # Store in semantic memory
        if tweets and self._memory:
            for tweet in tweets[:20]:  # Top 20 tweets
                await self._memory.store_semantic(
                    content=tweet.get('text', ''),
                    metadata={"source": "twitter", "author": tweet.get('author_id', ''), "url": f"https://twitter.com/i/web/status/{tweet.get('id', '')}"}
                )
        
        return {"tweets": tweets, "query": query}

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
        from ...knowledge import get_geo_source
        
        geo = get_geo_source()
        
        # Get location context (features + regions)
        context = await geo.get_location_context(lat, lon)
        
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
        
        # Query multiple sources in parallel
        import asyncio
        
        tasks = {
            "wikipedia": self.query_wikipedia(topic, limit=5),
            "web": self.search_web(topic, provider="duckduckgo"),
        }
        
        # Add optional sources if available
        try:
            tasks["reddit"] = self.query_reddit(topic.replace(" ", ""))  # Attempt subreddit name
        except:
            pass
        
        try:
            tasks["twitter"] = self.query_twitter(topic, max_results=10)
        except:
            pass
        
        # Execute all queries
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Collect results
        for source_name, result in zip(tasks.keys(), task_results):
            if not isinstance(result, Exception):
                results["sources"][source_name] = result
                # Count knowledge items
                if "results" in result:
                    results["knowledge_gained"] += len(result["results"])
                elif "posts" in result:
                    results["knowledge_gained"] += len(result["posts"])
                elif "tweets" in result:
                    results["knowledge_gained"] += len(result["tweets"])
        
        # Update exploration metrics
        self.state.metrics.topics_explored += 1
        
        return results

    # -------------------------------------------------------------------------
    # Core Loop: Perceive → Encode → Decide → Act → Learn
    # -------------------------------------------------------------------------

    async def autonomous_step(self) -> dict[str, Any]:
        """
        Autonomous agent step - driven by curiosity and information gain.
        
        Agent decides what to explore based on:
        - Prediction error (world model failures)
        - Novelty (new/unfamiliar states)
        - Expected information gain
        - Current goals
        """
        # 1. Check if we should generate new goals
        if self._goal_system and (not self.state.current_goal_id or np.random.random() < 0.1):
            await self._maybe_generate_goal()
        
        # 2. Compute curiosity-driven exploration signal
        exploration_signal = self._compute_exploration_signal()
        
        # 3. Decide on action: knowledge acquisition or policy action
        if exploration_signal > self.config.get("exploration_threshold", 0.5):
            # High curiosity - acquire knowledge from external world
            action = await self._explore_knowledge()
        else:
            # Low curiosity - execute policy action in environment
            observation = self._get_current_observation()
            action = self.step(observation)
        
        # 4. Update curiosity score
        self.state.metrics.curiosity_score = exploration_signal
        
        return action

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
        self.state.metrics.total_steps += 1
        self.state.updated_at = datetime.utcnow()

        return action

    def learn(self, transition: dict[str, Any]) -> dict[str, float]:
        """
        Learn from a transition (s, a, r, s', done).
        
        Returns loss metrics.
        """
        # Store in episodic memory
        if self._memory:
            self._memory.store_episode(transition)

        # Sample batch and update networks
        losses = {}

        if self._policy and self._memory:
            batch = self._memory.sample_for_learning(batch_size=32)
            if batch:
                losses = self._update_networks(batch)

        # Update reward metrics
        reward = transition.get("reward", 0.0)
        self.state.metrics.total_rewards += reward

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
            goal_embeddings = self._goal_system.goal_proposer(state.unsqueeze(0))
            goal_values = self._goal_system.goal_value(
                state.unsqueeze(0).expand(goal_embeddings.shape[0], -1),
                goal_embeddings.squeeze(0)
            )
        
        # Select highest value goal
        best_idx = goal_values.argmax()
        best_goal_embedding = goal_embeddings[0, best_idx]
        
        # Create goal from embedding
        from ..goals import Goal, GoalType
        new_goal = Goal(
            type=GoalType.KNOWLEDGE,
            description=f"Explore state space (novelty: {goal_values[best_idx].item():.3f})",
            target_embedding=best_goal_embedding,
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
        
        # Update metrics
        self.state.metrics.prediction_errors = signal
        
        return signal

    async def _explore_knowledge(self) -> dict[str, Any]:
        """
        Execute knowledge exploration action.
        
        Agent decides what topic to explore based on curiosity and goals.
        """
        # Get current goal to inform exploration
        goal_topic = None
        goal_achieved = False
        
        if self._goal_system and self.state.current_goal_id:
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal:
                # Parse goal description for topic
                desc = current_goal.description
                if "Explore" in desc or "Learn about" in desc:
                    # Extract topic from goal description
                    goal_topic = desc.split("Explore")[-1].split("Learn about")[-1].strip()
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
                    type=GoalType.KNOWLEDGE,
                    description=f"Learn about {topic}",
                    priority=0.7,
                    intrinsic_value=0.7,
                    source="curiosity",
                )
                goal_id = self._goal_system.add_goal(new_goal)
                self.state.current_goal_id = goal_id
        
        # Explore the topic
        results = await self.explore_topic(topic)
        
        # Update metrics
        self.state.metrics.knowledge_sources_queried += len(results["sources"])
        self.state.metrics.knowledge_acquired += results["knowledge_gained"]
        
        # Check if goal achieved (gained knowledge)
        if results["knowledge_gained"] > 0 and self._goal_system and self.state.current_goal_id:
            current_goal = self._goal_system.get_goal(self.state.current_goal_id)
            if current_goal:
                # Update goal progress based on knowledge gained
                progress = min(1.0, results["knowledge_gained"] / 10.0)  # Scale to [0, 1]
                self._goal_system.update_goal_progress(
                    self.state.current_goal_id,
                    self.state.latent_state,
                    progress
                )
                
                # Check if goal completed
                if current_goal.progress >= 1.0:
                    from ..goals import GoalStatus
                    current_goal.status = GoalStatus.ACHIEVED
                    current_goal.completed_at = datetime.utcnow()
                    goal_achieved = True
                    # Clear current goal to allow new goal generation
                    self.state.current_goal_id = None
        
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
        
        Agent decides what it's curious about based on memory gaps.
        """
        # Query semantic memory for topics with low coverage
        if self._memory:
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
        
        # Fallback: sample from predefined exploration domains
        domains = [
            "Mars geology", "ocean ecosystems", "quantum mechanics",
            "ancient civilizations", "machine learning", "neuroscience",
            "climate science", "space exploration", "renewable energy",
            "artificial intelligence", "human cognition", "evolutionary biology",
        ]
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
                deterministic = self.state.lifecycle == AgentLifecycle.DEPLOYED
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
            log_probs, entropy, new_values = self._policy.evaluate_actions(policy_states, actions)
            
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

        # Transition to training
        self.set_lifecycle(AgentLifecycle.TRAINING)

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
            if "goal_encoder_state_dict" in checkpoint:
                self._goal_system.goal_encoder.load_state_dict(checkpoint["goal_encoder_state_dict"])
            if "goal_proposer_state_dict" in checkpoint:
                self._goal_system.goal_proposer.load_state_dict(checkpoint["goal_proposer_state_dict"])
            if "goal_value_state_dict" in checkpoint:
                self._goal_system.goal_value.load_state_dict(checkpoint["goal_value_state_dict"])
