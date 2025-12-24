"""Memory System - unified 4-tier memory architecture."""

from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .episodic import EpisodicMemory, Transition
from .semantic import SemanticMemory


class MemorySystem:
    """
    Unified 4-tier memory system.
    
    - Short-term: Recent observations, working memory buffer
    - Long-term: Vector store / embeddings for persistent knowledge
    - Episodic: Experience replay buffer for learning from past actions
    - Semantic: Knowledge graph for structured relationships
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        config = config or {}

        # Initialize memory tiers
        self.short_term = ShortTermMemory(
            capacity=config.get("short_term_capacity", 100),
        )

        self.long_term = LongTermMemory(
            collection_name=config.get("collection_name", "agent_memory"),
            chroma_url=config.get("chroma_url"),
            persist_directory=config.get("persist_directory"),
        )

        self.episodic = EpisodicMemory(
            capacity=config.get("episodic_capacity", 100_000),
            prioritized=config.get("prioritized_replay", True),
        )

        self.semantic = SemanticMemory()

        # Embedding model for long-term memory
        self._encoder: SentenceTransformer | None = None
        self._encoder_name = config.get("encoder_model", "all-MiniLM-L6-v2")

    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._encoder_name)
        return self._encoder

    # -------------------------------------------------------------------------
    # Short-term memory operations
    # -------------------------------------------------------------------------

    def add_short_term(
        self,
        observation: torch.Tensor,
        source: str = "environment",
        **metadata: Any,
    ) -> None:
        """Add observation to short-term memory."""
        self.short_term.add(observation, source=source, **metadata)

    def get_recent_context(self, n: int = 10) -> torch.Tensor:
        """Get recent context as stacked tensor."""
        return self.short_term.get_context_tensor(n)

    # -------------------------------------------------------------------------
    # Long-term memory operations
    # -------------------------------------------------------------------------

    def encode_to_long_term(
        self,
        content: str,
        source: str = "observation",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Encode content and store in long-term memory."""
        embedding = self.encoder.encode(content).tolist()
        return self.long_term.add(
            embedding=embedding,
            content=content,
            source=source,
            metadata=metadata,
        )

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories based on semantic similarity."""
        query_embedding = self.encoder.encode(query).tolist()
        return self.long_term.search(query_embedding, top_k=top_k)

    # -------------------------------------------------------------------------
    # Episodic memory operations
    # -------------------------------------------------------------------------

    def store_episode(self, transition: dict[str, Any]) -> None:
        """Store experience transition for replay learning."""
        t = Transition(
            state=transition["state"],
            action=transition["action"],
            reward=transition["reward"],
            next_state=transition["next_state"],
            done=transition.get("done", False),
            goal=transition.get("goal"),
            info=transition.get("info", {}),
        )
        self.episodic.push(t)

    def sample_for_learning(
        self,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, torch.Tensor] | None:
        """Sample batch of transitions for learning."""
        if len(self.episodic) < batch_size:
            return None

        transitions, indices, weights = self.episodic.sample(batch_size)
        batch = self.episodic.to_batch_tensors(transitions, device)
        batch["indices"] = indices
        batch["weights"] = torch.tensor(weights, dtype=torch.float32, device=device)

        return batch

    def update_replay_priorities(
        self,
        indices: Any,
        td_errors: Any,
    ) -> None:
        """Update priorities for prioritized experience replay."""
        self.episodic.update_priorities(indices, td_errors)

    # -------------------------------------------------------------------------
    # Semantic memory operations
    # -------------------------------------------------------------------------

    def add_knowledge(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> bool:
        """Add structured knowledge triple."""
        return self.semantic.add_triple(subject, predicate, obj)

    def query_knowledge(
        self,
        concept: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Query knowledge graph around a concept."""
        return self.semantic.query_subgraph(concept, depth)

    def find_relation_path(
        self,
        source: str,
        target: str,
    ) -> list[str] | None:
        """Find path between two concepts in knowledge graph."""
        return self.semantic.find_path(source, target)

    # -------------------------------------------------------------------------
    # Memory consolidation
    # -------------------------------------------------------------------------

    def consolidate(self, batch: list[dict[str, Any]] | None = None) -> None:
        """
        Consolidate memories across tiers.
        
        - Move important short-term to long-term
        - Extract patterns from episodic to semantic
        """
        # TODO: Implement memory consolidation logic
        # - Identify important short-term observations
        # - Extract knowledge from episodic experiences
        # - Update semantic graph with learned patterns
        pass

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": self.long_term.count(),
            "episodic_count": len(self.episodic),
            "semantic_nodes": self.semantic.count_nodes(),
            "semantic_edges": self.semantic.count_edges(),
        }

    def clear_all(self) -> None:
        """Clear all memory tiers."""
        self.short_term.clear()
        self.long_term.clear()
        self.episodic.clear()
        self.semantic.clear()
