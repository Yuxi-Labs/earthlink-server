"""
Memory system for agents - hierarchical memory with consolidation and retrieval.

Store Memory capability: Multi-level memory hierarchy (episodic, semantic, procedural),
automatic consolidation, relevance-based retrieval, and adaptive forgetting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MemoryType(str, Enum):
    """Types of memory in the hierarchical system."""
    
    EPISODIC = "episodic"  # Specific events/experiences
    SEMANTIC = "semantic"  # General knowledge/facts
    PROCEDURAL = "procedural"  # Skills/how-to knowledge
    WORKING = "working"  # Temporary short-term memory


class MemoryImportance(str, Enum):
    """Importance levels for memory retention."""
    
    CRITICAL = "critical"  # Never forget
    HIGH = "high"  # Long retention
    MEDIUM = "medium"  # Standard retention
    LOW = "low"  # Quick decay


@dataclass
class Memory:
    """A single memory unit in the agent's memory system."""
    
    id: UUID = field(default_factory=uuid4)
    agent_id: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    content: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5  # 0-1, affects retention
    access_count: int = 0  # How often accessed
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None  # Semantic embedding for retrieval
    tags: list[str] = field(default_factory=list)
    related_memories: list[UUID] = field(default_factory=list)
    consolidated: bool = False  # Whether moved from working to long-term
    decay_rate: float = 0.01  # How fast it fades (0-1)
    
    def age_hours(self) -> float:
        """Get age of memory in hours."""
        return (datetime.utcnow() - self.timestamp).total_seconds() / 3600
    
    def recency_score(self) -> float:
        """Calculate recency score (more recent = higher score)."""
        hours_since_access = (datetime.utcnow() - self.last_accessed).total_seconds() / 3600
        # Exponential decay: score = e^(-decay_rate * hours)
        return np.exp(-self.decay_rate * hours_since_access)
    
    def access(self):
        """Record memory access (strengthens memory)."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        # Accessing memory reduces decay rate (strengthens retention)
        self.decay_rate *= 0.9  # 10% reduction in decay each access
    
    def current_strength(self) -> float:
        """Calculate current memory strength (combines importance, recency, access)."""
        recency = self.recency_score()
        access_factor = min(1.0, self.access_count / 10)  # Cap at 1.0
        
        # Weighted combination
        strength = (
            self.importance * 0.4 +
            recency * 0.3 +
            access_factor * 0.3
        )
        
        return min(1.0, max(0.0, strength))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self.id),
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "tags": self.tags,
            "consolidated": self.consolidated,
            "strength": self.current_strength(),
        }


class AdvancedMemory:
    """
    Enhanced memory system with hierarchical organization and smart retrieval.
    
    Features:
    - Multi-level memory (episodic, semantic, procedural, working)
    - Automatic consolidation from working to long-term
    - Relevance-based retrieval using embeddings
    - Adaptive forgetting based on importance
    - Memory strengthening through access
    """
    
    def __init__(
        self,
        agent_id: str,
        max_working_memory: int = 50,
        max_long_term_memory: int = 1000,
        consolidation_interval_hours: float = 1.0,
        forgetting_threshold: float = 0.2,
    ):
        self.agent_id = agent_id
        self.max_working_memory = max_working_memory
        self.max_long_term_memory = max_long_term_memory
        self.consolidation_interval_hours = consolidation_interval_hours
        self.forgetting_threshold = forgetting_threshold
        
        # Memory stores by type
        self.working_memory: list[Memory] = []
        self.episodic_memory: list[Memory] = []
        self.semantic_memory: list[Memory] = []
        self.procedural_memory: list[Memory] = []
        
        # Consolidation tracking
        self.last_consolidation = datetime.utcnow()
        
        # Statistics
        self.stats = {
            "memories_created": 0,
            "memories_consolidated": 0,
            "memories_forgotten": 0,
            "memories_retrieved": 0,
            "consolidations_performed": 0,
        }
    
    def store(
        self,
        content: dict[str, Any],
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        tags: list[str] = None,
    ) -> Memory:
        """
        Store new memory.
        
        Args:
            content: Memory content (dict with any structure)
            memory_type: Type of memory
            importance: Importance score (0-1)
            tags: Optional tags for categorization
        
        Returns:
            Created Memory object
        """
        memory = Memory(
            agent_id=self.agent_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags or [],
            decay_rate=self._calculate_decay_rate(importance),
        )
        
        # Generate semantic embedding for retrieval
        memory.embedding = self._generate_embedding(content)
        
        # Store in working memory first
        self.working_memory.append(memory)
        self.stats["memories_created"] += 1
        
        # Auto-consolidate if working memory full
        if len(self.working_memory) >= self.max_working_memory:
            self.consolidate()
        
        return memory
    
    def retrieve(
        self,
        query: dict[str, Any],
        top_k: int = 5,
        memory_types: list[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """
        Retrieve memories by semantic relevance.
        
        Args:
            query: Query dict (similar structure to memory content)
            top_k: Number of memories to retrieve
            memory_types: Filter by memory types (None = all types)
            min_importance: Minimum importance threshold
        
        Returns:
            List of most relevant memories
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Collect all memories to search
        all_memories = []
        if memory_types is None:
            memory_types = list(MemoryType)
        
        for mem_type in memory_types:
            all_memories.extend(self._get_memories_by_type(mem_type))
        
        # Filter by importance
        candidates = [m for m in all_memories if m.importance >= min_importance]
        
        if not candidates:
            return []
        
        # Calculate relevance scores
        scored_memories = []
        for memory in candidates:
            if memory.embedding is None:
                continue
            
            # Semantic similarity
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            
            # Combine with memory strength
            strength = memory.current_strength()
            relevance_score = similarity * 0.7 + strength * 0.3
            
            scored_memories.append((memory, relevance_score))
        
        # Sort by relevance and take top_k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m for m, score in scored_memories[:top_k]]
        
        # Mark as accessed (strengthens memories)
        for memory in top_memories:
            memory.access()
        
        self.stats["memories_retrieved"] += len(top_memories)
        
        return top_memories
    
    def consolidate(self) -> int:
        """
        Consolidate working memory to long-term memory.
        
        Moves important memories from working memory to appropriate
        long-term stores (episodic, semantic, procedural).
        
        Returns:
            Number of memories consolidated
        """
        if not self.working_memory:
            return 0
        
        # Sort by importance
        self.working_memory.sort(key=lambda m: m.importance, reverse=True)
        
        consolidated_count = 0
        memories_to_keep = []
        
        for memory in self.working_memory:
            # Decide whether to consolidate based on age and importance
            age_hours = memory.age_hours()
            should_consolidate = (
                age_hours >= self.consolidation_interval_hours or
                memory.importance > 0.7
            )
            
            if should_consolidate:
                # Move to appropriate long-term store
                memory.consolidated = True
                
                if memory.memory_type == MemoryType.EPISODIC:
                    self.episodic_memory.append(memory)
                elif memory.memory_type == MemoryType.SEMANTIC:
                    self.semantic_memory.append(memory)
                elif memory.memory_type == MemoryType.PROCEDURAL:
                    self.procedural_memory.append(memory)
                
                consolidated_count += 1
            else:
                # Keep in working memory
                memories_to_keep.append(memory)
        
        self.working_memory = memories_to_keep
        self.last_consolidation = datetime.utcnow()
        self.stats["memories_consolidated"] += consolidated_count
        self.stats["consolidations_performed"] += 1
        
        # After consolidation, check if we need to forget old memories
        self.forget()
        
        return consolidated_count
    
    def forget(self, importance_threshold: float = None) -> int:
        """
        Remove low-strength memories (adaptive forgetting).
        
        Args:
            importance_threshold: Custom threshold (uses default if None)
        
        Returns:
            Number of memories forgotten
        """
        threshold = importance_threshold or self.forgetting_threshold
        forgotten_count = 0
        
        # Forget from each memory store
        for memory_store in [self.episodic_memory, self.semantic_memory, self.procedural_memory]:
            before_count = len(memory_store)
            
            # Keep memories above strength threshold
            memory_store[:] = [
                m for m in memory_store
                if m.current_strength() >= threshold
            ]
            
            forgotten_count += (before_count - len(memory_store))
        
        # Apply capacity limits
        forgotten_count += self._enforce_capacity_limits()
        
        self.stats["memories_forgotten"] += forgotten_count
        
        return forgotten_count
    
    def retrieve_by_tag(self, tag: str, top_k: int = 10) -> list[Memory]:
        """Retrieve memories by tag."""
        all_memories = (
            self.working_memory +
            self.episodic_memory +
            self.semantic_memory +
            self.procedural_memory
        )
        
        tagged_memories = [m for m in all_memories if tag in m.tags]
        
        # Sort by strength
        tagged_memories.sort(key=lambda m: m.current_strength(), reverse=True)
        
        return tagged_memories[:top_k]
    
    def retrieve_recent(
        self,
        hours: float = 24.0,
        memory_type: MemoryType = None,
    ) -> list[Memory]:
        """Retrieve recent memories within time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        memories = self._get_memories_by_type(memory_type) if memory_type else (
            self.working_memory +
            self.episodic_memory +
            self.semantic_memory +
            self.procedural_memory
        )
        
        recent = [m for m in memories if m.timestamp >= cutoff_time]
        recent.sort(key=lambda m: m.timestamp, reverse=True)
        
        return recent
    
    def get_statistics(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self.stats,
            "working_memory_count": len(self.working_memory),
            "episodic_memory_count": len(self.episodic_memory),
            "semantic_memory_count": len(self.semantic_memory),
            "procedural_memory_count": len(self.procedural_memory),
            "total_memories": (
                len(self.working_memory) +
                len(self.episodic_memory) +
                len(self.semantic_memory) +
                len(self.procedural_memory)
            ),
        }
    
    # Private helper methods
    
    def _get_memories_by_type(self, memory_type: MemoryType) -> list[Memory]:
        """Get all memories of a specific type."""
        if memory_type == MemoryType.WORKING:
            return self.working_memory
        elif memory_type == MemoryType.EPISODIC:
            return self.episodic_memory
        elif memory_type == MemoryType.SEMANTIC:
            return self.semantic_memory
        elif memory_type == MemoryType.PROCEDURAL:
            return self.procedural_memory
        return []
    
    def _generate_embedding(self, content: dict[str, Any]) -> list[float]:
        """
        Generate semantic embedding for memory content.
        
        Simplified version using content features.
        In production, use a proper embedding model.
        """
        # Extract text features
        text_features = []
        for key, value in content.items():
            text_features.append(str(key))
            text_features.append(str(value))
        
        combined_text = " ".join(text_features).lower()
        
        # Simple hash-based embedding (128-dim)
        # In production, use sentence transformers or similar
        embedding_dim = 128
        embedding = np.zeros(embedding_dim)
        
        for i, char in enumerate(combined_text):
            idx = hash(char + str(i)) % embedding_dim
            embedding[idx] += ord(char) / 1000.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def _cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        emb1_arr = np.array(emb1)
        emb2_arr = np.array(emb2)
        
        dot_product = np.dot(emb1_arr, emb2_arr)
        norm1 = np.linalg.norm(emb1_arr)
        norm2 = np.linalg.norm(emb2_arr)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _calculate_decay_rate(self, importance: float) -> float:
        """Calculate decay rate based on importance (higher importance = slower decay)."""
        # importance 1.0 -> decay 0.001 (very slow)
        # importance 0.0 -> decay 0.1 (very fast)
        return 0.1 - (importance * 0.099)
    
    def _enforce_capacity_limits(self) -> int:
        """Enforce memory capacity limits, remove lowest strength memories."""
        forgotten = 0
        
        # Check episodic memory limit
        if len(self.episodic_memory) > self.max_long_term_memory:
            self.episodic_memory.sort(key=lambda m: m.current_strength(), reverse=True)
            excess = len(self.episodic_memory) - self.max_long_term_memory
            self.episodic_memory = self.episodic_memory[:-excess]
            forgotten += excess
        
        # Similar for semantic and procedural
        for memory_store in [self.semantic_memory, self.procedural_memory]:
            if len(memory_store) > self.max_long_term_memory // 2:
                memory_store.sort(key=lambda m: m.current_strength(), reverse=True)
                excess = len(memory_store) - (self.max_long_term_memory // 2)
                memory_store[:] = memory_store[:-excess]
                forgotten += excess
        
        return forgotten
