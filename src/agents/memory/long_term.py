"""Long-term memory - vector store / embeddings for persistent knowledge."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

# ChromaDB is optional; fallback to in-memory collection if unavailable.
try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
except Exception:  # pragma: no cover - fallback when chromadb not installed
    chromadb = None
    Settings = None


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
        self._init_collection(chroma_url, persist_directory)

    def _init_collection(self, chroma_url: str | None, persist_directory: str | None) -> None:
        """Initialize vector collection with ChromaDB or in-memory fallback."""
        if chromadb is None or Settings is None:
            self.collection = _InMemoryCollection()
            return

        # Initialize ChromaDB client
        if chroma_url:
            self.client = chromadb.HttpClient(host=chroma_url)
        elif persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
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

    def count(self) -> int:
        """Return total number of stored long-term memories."""
        try:
            return self.collection.count()
        except Exception:
            return 0


class _InMemoryCollection:
    """Minimal in-memory vector store fallback for development/tests."""

    def __init__(self):
        self._records: list[dict[str, Any]] = []

    def add(self, ids, embeddings, documents, metadatas):
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._records.append(
                {"id": id_, "embedding": emb, "document": doc, "metadata": meta}
            )

    def query(self, query_embeddings, n_results=5, where=None):
        # Very naive similarity: return first n items
        records = self._records[:n_results]
        return {
            "ids": [[r["id"] for r in records]],
            "documents": [[r["document"] for r in records]],
            "metadatas": [[r["metadata"] for r in records]],
            "distances": [[0.0 for _ in records]],
        }

    def get(self, ids):
        found = [r for r in self._records if r["id"] in ids]
        if not found:
            return {"ids": [], "documents": [], "metadatas": []}
        first = found[0]
        return {
            "ids": [first["id"]],
            "documents": [first["document"]],
            "metadatas": [first["metadata"]],
        }

    def count(self):
        return len(self._records)

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
