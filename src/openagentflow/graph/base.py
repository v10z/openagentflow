"""Abstract graph backend protocol for execution tracing.

Defines the interface that all graph backends must implement.
Graph backends store execution DAGs as vertices and edges,
enabling full lineage tracing of agent, chain, and swarm runs.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph-based execution tracing backends.

    All methods are async and non-blocking. Implementations must be
    safe to call concurrently from multiple coroutines.

    Vertices represent execution units (agents, tools, chains, swarms).
    Edges represent relationships (CALLED, STEP_N, PARALLEL, etc.).

    Example:
        backend = SQLiteGraphBackend(":memory:")
        await backend.add_vertex("v1", "agent", {"name": "planner"})
        await backend.add_vertex("v2", "tool", {"name": "search"})
        await backend.add_edge("v1", "v2", "CALLED", {"iteration": 1})
        trace = await backend.get_full_trace(run_id="abc123")
    """

    async def add_vertex(
        self,
        vertex_id: str,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        """Add a vertex to the graph.

        Args:
            vertex_id: Unique identifier for the vertex.
            label: Vertex type label (e.g., "agent", "tool", "chain", "swarm").
            properties: Key-value properties to store on the vertex.
                Must include "run_id" for trace grouping.
        """
        ...

    async def update_vertex(
        self,
        vertex_id: str,
        properties: dict[str, Any],
    ) -> None:
        """Update properties on an existing vertex.

        Args:
            vertex_id: The vertex to update.
            properties: Properties to merge into existing properties.
        """
        ...

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        """Add a directed edge between two vertices.

        Args:
            source_id: Source vertex ID.
            target_id: Target vertex ID.
            label: Edge type label (e.g., "CALLED", "STEP_0", "PARALLEL").
            properties: Key-value properties to store on the edge.
        """
        ...

    async def query_vertex(self, vertex_id: str) -> dict[str, Any] | None:
        """Query a single vertex by ID.

        Args:
            vertex_id: The vertex ID to look up.

        Returns:
            Vertex data dict with keys: id, label, properties, run_id,
            created_at. Returns None if not found.
        """
        ...

    async def query_children(self, vertex_id: str) -> list[dict[str, Any]]:
        """Query all vertices that are direct children (outgoing edges) of the given vertex.

        Args:
            vertex_id: The parent vertex ID.

        Returns:
            List of child vertex dicts, each with: id, label, properties,
            edge_label, edge_properties.
        """
        ...

    async def query_parents(self, vertex_id: str) -> list[dict[str, Any]]:
        """Query all vertices that are direct parents (incoming edges) of the given vertex.

        Args:
            vertex_id: The child vertex ID.

        Returns:
            List of parent vertex dicts, each with: id, label, properties,
            edge_label, edge_properties.
        """
        ...

    async def get_full_trace(self, run_id: str) -> dict[str, Any]:
        """Get the full execution DAG for a given run.

        Args:
            run_id: The run identifier to retrieve the trace for.

        Returns:
            Dict with keys:
                "vertices": list of vertex dicts
                "edges": list of edge dicts (source_id, target_id, label, properties)
        """
        ...

    async def clear(self) -> None:
        """Remove all data from the graph backend.

        Use with caution -- this deletes all vertices and edges.
        """
        ...
