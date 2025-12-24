"""Long-term memory - vector store / embeddings for persistent knowledge."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import chromadb
from chromadb.config import Settings


@dataclass
class LongTermMemoryEntry:
    """Entry in long-term memory."""

    id: UUID = field(default_factory=uuid4)
    embedding: list[float] = field(default_factory=list)
    content: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class LongTermMemory:
    """
    Long-term memory using ChromaDB.
    
    Stores embeddings for persistent knowledge retrieval.
    """

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_directory: str | None = None,
        chroma_url: str | None = None,
    ):
        self.collection_name = collection_name

        # Initialize ChromaDB client
        if chroma_url:
            # Remote ChromaDB
            self.client = chromadb.HttpClient(host=chroma_url)
        elif persist_directory:
            # Persistent local
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            # In-memory
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        embedding: list[float],
        content: str,
        source: str = "observation",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add embedding to long-term memory."""
        entry_id = str(uuid4())
        
        self.collection.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "source": source,
                **(metadata or {}),
                "created_at": datetime.utcnow().isoformat(),
            }],
        )
        
        return entry_id

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar memories."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                memories.append({
                    "id": id_,
                    "content": results["documents"][0][i] if results["documents"] else None,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return memories

    def get(self, entry_id: str) -> dict[str, Any] | None:
        """Get specific memory by ID."""
        result = self.collection.get(ids=[entry_id])
        
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "content": result["documents"][0] if result["documents"] else None,
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
            }
        return None

    def delete(self, entry_id: str) -> None:
        """Delete memory by ID."""
        self.collection.delete(ids=[entry_id])

    def count(self) -> int:
        """Get total number of memories."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all memories in collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
