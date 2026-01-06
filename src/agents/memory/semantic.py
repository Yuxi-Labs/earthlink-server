"""Semantic memory - knowledge graph for structured relationships."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterator
from uuid import UUID, uuid4

import networkx as nx


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph."""

    id: UUID = field(default_factory=uuid4)
    label: str = ""
    node_type: str = "concept"  # concept, entity, fact, relation
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class KnowledgeEdge:
    """Edge (relationship) in the knowledge graph."""

    source_id: UUID
    target_id: UUID
    relation: str  # e.g., "is_a", "has_part", "causes", "related_to"
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SemanticMemory:
    """
    Semantic memory using a knowledge graph.
    
    Stores structured relationships between concepts.
    Uses NetworkX for in-memory graph operations.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_index: dict[str, UUID] = {}  # label -> id mapping

    def add_node(
        self,
        label: str,
        node_type: str = "concept",
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> UUID:
        """Add node to knowledge graph."""
        # Check if node with label already exists
        if label in self._node_index:
            return self._node_index[label]

        node = KnowledgeNode(
            label=label,
            node_type=node_type,
            properties=properties or {},
            embedding=embedding,
        )

        self.graph.add_node(
            str(node.id),
            label=label,
            node_type=node_type,
            properties=node.properties,
            embedding=embedding,
            created_at=node.created_at.isoformat(),
        )

        self._node_index[label] = node.id
        return node.id

    def add_edge(
        self,
        source_label: str,
        target_label: str,
        relation: str,
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Add edge between nodes (creates nodes if they don't exist)."""
        # Ensure nodes exist
        source_id = self.add_node(source_label)
        target_id = self.add_node(target_label)

        # Add edge
        self.graph.add_edge(
            str(source_id),
            str(target_id),
            relation=relation,
            weight=weight,
            properties=properties or {},
            created_at=datetime.now(UTC).isoformat(),
        )

        return True

    def add_triple(self, subject: str, predicate: str, obj: str) -> bool:
        """Add a triple (subject, predicate, object) to the graph."""
        return self.add_edge(subject, obj, relation=predicate)

    def get_node(self, label: str) -> dict[str, Any] | None:
        """Get node by label."""
        if label not in self._node_index:
            return None

        node_id = str(self._node_index[label])
        if node_id in self.graph:
            return {"id": node_id, **self.graph.nodes[node_id]}
        return None

    def get_neighbors(
        self,
        label: str,
        relation: str | None = None,
        direction: str = "out",  # "out", "in", "both"
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes."""
        if label not in self._node_index:
            return []

        node_id = str(self._node_index[label])
        neighbors = []

        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if relation is None or data.get("relation") == relation:
                    neighbors.append({
                        "id": target,
                        "direction": "out",
                        "relation": data.get("relation"),
                        **self.graph.nodes[target],
                    })

        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(node_id, data=True):
                if relation is None or data.get("relation") == relation:
                    neighbors.append({
                        "id": source,
                        "direction": "in",
                        "relation": data.get("relation"),
                        **self.graph.nodes[source],
                    })

        return neighbors

    def find_path(
        self,
        source_label: str,
        target_label: str,
        max_length: int = 5,
    ) -> list[str] | None:
        """Find shortest path between two nodes."""
        if source_label not in self._node_index or target_label not in self._node_index:
            return None

        source_id = str(self._node_index[source_label])
        target_id = str(self._node_index[target_label])

        try:
            path_ids = nx.shortest_path(
                self.graph,
                source=source_id,
                target=target_id,
            )

            if len(path_ids) > max_length:
                return None

            # Convert IDs to labels
            return [self.graph.nodes[nid]["label"] for nid in path_ids]

        except nx.NetworkXNoPath:
            return None

    def query_subgraph(
        self,
        center_label: str,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get subgraph around a central node."""
        if center_label not in self._node_index:
            return {"nodes": [], "edges": []}

        center_id = str(self._node_index[center_label])

        # BFS to get nodes within depth
        visited = {center_id}
        frontier = [center_id]

        for _ in range(depth):
            next_frontier = []
            for node_id in frontier:
                for neighbor in self.graph.successors(node_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                for neighbor in self.graph.predecessors(node_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        # Extract subgraph
        subgraph = self.graph.subgraph(visited)

        nodes = [{"id": nid, **subgraph.nodes[nid]} for nid in subgraph.nodes]
        edges = [
            {"source": u, "target": v, **data}
            for u, v, data in subgraph.edges(data=True)
        ]

        return {"nodes": nodes, "edges": edges}

    def get_all_relations(self) -> list[str]:
        """Get all unique relation types in the graph."""
        relations = set()
        for _, _, data in self.graph.edges(data=True):
            if "relation" in data:
                relations.add(data["relation"])
        return list(relations)

    def count_nodes(self) -> int:
        """Get total number of nodes."""
        return self.graph.number_of_nodes()

    def count_edges(self) -> int:
        """Get total number of edges."""
        return self.graph.number_of_edges()

    def clear(self) -> None:
        """Clear the knowledge graph."""
        self.graph.clear()
        self._node_index.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export graph to dictionary."""
        return {
            "nodes": [{"id": nid, **self.graph.nodes[nid]} for nid in self.graph.nodes],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Import graph from dictionary."""
        self.clear()

        for node in data.get("nodes", []):
            node_id = node.pop("id")
            label = node.get("label", node_id)
            self.graph.add_node(node_id, **node)
            self._node_index[label] = UUID(node_id)

        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            self.graph.add_edge(source, target, **edge)
