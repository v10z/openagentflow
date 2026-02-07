"""Optional Gremlin/TinkerGraph backend for production graph tracing.

Requires the ``gremlinpython`` package (optional dependency).
Install via:
    pip install openagentflow[gremlin]
    # or: pip install gremlinpython

Connects to a Gremlin-compatible graph server (TinkerPop, AWS Neptune,
JanusGraph, etc.) over WebSocket.

If ``gremlinpython`` is not installed, importing this module will not
crash -- the class will raise a clear error at instantiation time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Try to import gremlinpython; set a flag if not available
_GREMLIN_AVAILABLE = False
try:
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.process.graph_traversal import GraphTraversalSource, __
    from gremlin_python.process.traversal import T, Cardinality

    _GREMLIN_AVAILABLE = True
except ImportError:
    DriverRemoteConnection = None  # type: ignore[assignment, misc]
    traversal = None  # type: ignore[assignment]
    GraphTraversalSource = None  # type: ignore[assignment, misc]
    __ = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    Cardinality = None  # type: ignore[assignment]


_DEFAULT_ENDPOINT = "ws://localhost:8182/gremlin"


class GremlinGraphBackend:
    """Gremlin-based graph backend for production execution tracing.

    Requires ``gremlinpython`` to be installed. Connects to a Gremlin
    server over WebSocket and uses Gremlin traversal queries.

    Example:
        backend = GremlinGraphBackend("ws://localhost:8182/gremlin")
        await backend.add_vertex("agent-1", "agent", {"name": "planner", "run_id": "run-1"})
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        traversal_source: str = "g",
    ) -> None:
        """Initialize the Gremlin graph backend.

        Args:
            endpoint: WebSocket URL for the Gremlin server.
                Defaults to ws://localhost:8182/gremlin.
            traversal_source: The traversal source name on the server.
                Defaults to "g".

        Raises:
            ImportError: If gremlinpython is not installed.
        """
        if not _GREMLIN_AVAILABLE:
            raise ImportError(
                "gremlinpython is required for GremlinGraphBackend. "
                "Install it with: pip install gremlinpython"
            )

        self._endpoint = endpoint or _DEFAULT_ENDPOINT
        self._traversal_source = traversal_source
        self._connection: Any = None
        self._g: Any = None

    @property
    def endpoint(self) -> str:
        """Return the Gremlin server endpoint."""
        return self._endpoint

    def _ensure_connected(self) -> Any:
        """Ensure a connection to the Gremlin server exists.

        Returns the graph traversal source.
        """
        if self._connection is None or self._g is None:
            self._connection = DriverRemoteConnection(
                self._endpoint,
                self._traversal_source,
            )
            self._g = traversal().withRemote(self._connection)
        return self._g

    async def add_vertex(
        self,
        vertex_id: str,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        """Add a vertex to the Gremlin graph.

        Uses g.addV(label).property(T.id, vertex_id).property(k, v)...
        """
        try:
            g = self._ensure_connected()
            t = g.addV(label).property(T.id, vertex_id)

            # Add run_id as a top-level property for indexing
            run_id = properties.get("run_id")
            if run_id:
                t = t.property("run_id", run_id)

            # Store properties as JSON string for complex nested values
            t = t.property("properties_json", json.dumps(properties, default=str))
            t = t.property("created_at", datetime.now(timezone.utc).isoformat())

            # Also add simple string/numeric properties directly for queryability
            for key, value in properties.items():
                if isinstance(value, (str, int, float, bool)):
                    t = t.property(key, value)

            t.next()
        except Exception:
            logger.exception("Failed to add vertex %s to Gremlin graph", vertex_id)
            raise

    async def update_vertex(
        self,
        vertex_id: str,
        properties: dict[str, Any],
    ) -> None:
        """Update properties on an existing vertex."""
        try:
            g = self._ensure_connected()
            t = g.V(vertex_id)

            # Get existing properties_json and merge
            existing_json = t.values("properties_json").next()
            existing = json.loads(existing_json) if existing_json else {}
            existing.update(properties)

            # Re-traverse and update
            t = g.V(vertex_id)
            t = t.property("properties_json", json.dumps(existing, default=str))
            for key, value in properties.items():
                if isinstance(value, (str, int, float, bool)):
                    t = t.property(key, value)

            t.next()
        except Exception:
            logger.exception("Failed to update vertex %s in Gremlin graph", vertex_id)
            raise

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        """Add an edge between two vertices in the Gremlin graph.

        Uses g.V(source_id).addE(label).to(g.V(target_id)).property(k, v)...
        """
        try:
            g = self._ensure_connected()
            t = g.V(source_id).addE(label).to(__.V(target_id))

            t = t.property("properties_json", json.dumps(properties, default=str))
            t = t.property("created_at", datetime.now(timezone.utc).isoformat())

            for key, value in properties.items():
                if isinstance(value, (str, int, float, bool)):
                    t = t.property(key, value)

            t.next()
        except Exception:
            logger.exception(
                "Failed to add edge %s -> %s (%s) to Gremlin graph",
                source_id,
                target_id,
                label,
            )
            raise

    async def query_vertex(self, vertex_id: str) -> dict[str, Any] | None:
        """Query a single vertex by ID."""
        try:
            g = self._ensure_connected()
            results = g.V(vertex_id).elementMap().toList()
            if not results:
                return None

            vertex_map = results[0]
            properties_json = vertex_map.get("properties_json", "{}")
            return {
                "id": vertex_id,
                "label": vertex_map.get(T.label, ""),
                "properties": json.loads(properties_json),
                "run_id": vertex_map.get("run_id"),
                "created_at": vertex_map.get("created_at"),
            }
        except Exception:
            logger.exception("Failed to query vertex %s from Gremlin graph", vertex_id)
            return None

    async def query_children(self, vertex_id: str) -> list[dict[str, Any]]:
        """Query all vertices connected by outgoing edges."""
        try:
            g = self._ensure_connected()
            # Get outgoing edges and their target vertices
            results = (
                g.V(vertex_id)
                .outE()
                .project("edge_label", "edge_props", "target")
                .by(T.label)
                .by(__.values("properties_json"))
                .by(__.inV().elementMap())
                .toList()
            )

            children = []
            for item in results:
                target = item["target"]
                target_props_json = target.get("properties_json", "{}")
                children.append({
                    "id": str(target.get(T.id, "")),
                    "label": target.get(T.label, ""),
                    "properties": json.loads(target_props_json),
                    "run_id": target.get("run_id"),
                    "created_at": target.get("created_at"),
                    "edge_label": item["edge_label"],
                    "edge_properties": json.loads(item.get("edge_props", "{}")),
                })
            return children
        except Exception:
            logger.exception(
                "Failed to query children of %s from Gremlin graph", vertex_id
            )
            return []

    async def query_parents(self, vertex_id: str) -> list[dict[str, Any]]:
        """Query all vertices connected by incoming edges."""
        try:
            g = self._ensure_connected()
            results = (
                g.V(vertex_id)
                .inE()
                .project("edge_label", "edge_props", "source")
                .by(T.label)
                .by(__.values("properties_json"))
                .by(__.outV().elementMap())
                .toList()
            )

            parents = []
            for item in results:
                source = item["source"]
                source_props_json = source.get("properties_json", "{}")
                parents.append({
                    "id": str(source.get(T.id, "")),
                    "label": source.get(T.label, ""),
                    "properties": json.loads(source_props_json),
                    "run_id": source.get("run_id"),
                    "created_at": source.get("created_at"),
                    "edge_label": item["edge_label"],
                    "edge_properties": json.loads(item.get("edge_props", "{}")),
                })
            return parents
        except Exception:
            logger.exception(
                "Failed to query parents of %s from Gremlin graph", vertex_id
            )
            return []

    async def get_full_trace(self, run_id: str) -> dict[str, Any]:
        """Get the full execution DAG for a run.

        Returns all vertices with the given run_id and all edges
        connecting them.
        """
        try:
            g = self._ensure_connected()

            # Get all vertices for this run
            v_results = g.V().has("run_id", run_id).elementMap().toList()
            vertices = []
            vertex_ids = set()
            for vm in v_results:
                vid = str(vm.get(T.id, ""))
                vertex_ids.add(vid)
                props_json = vm.get("properties_json", "{}")
                vertices.append({
                    "id": vid,
                    "label": vm.get(T.label, ""),
                    "properties": json.loads(props_json),
                    "run_id": vm.get("run_id"),
                    "created_at": vm.get("created_at"),
                })

            if not vertex_ids:
                return {"vertices": [], "edges": []}

            # Get all edges between these vertices
            edges = []
            for vid in vertex_ids:
                edge_results = (
                    g.V(vid)
                    .outE()
                    .project("source", "target", "label", "props", "created")
                    .by(__.outV().id_())
                    .by(__.inV().id_())
                    .by(T.label)
                    .by(__.values("properties_json"))
                    .by(__.values("created_at"))
                    .toList()
                )
                for e in edge_results:
                    target_id = str(e["target"])
                    if target_id in vertex_ids:
                        edges.append({
                            "source_id": str(e["source"]),
                            "target_id": target_id,
                            "label": e["label"],
                            "properties": json.loads(e.get("props", "{}")),
                            "created_at": e.get("created"),
                        })

            return {"vertices": vertices, "edges": edges}
        except Exception:
            logger.exception("Failed to get full trace for run %s from Gremlin graph", run_id)
            return {"vertices": [], "edges": []}

    async def clear(self) -> None:
        """Drop all vertices and edges from the graph."""
        try:
            g = self._ensure_connected()
            g.V().drop().iterate()
        except Exception:
            logger.exception("Failed to clear Gremlin graph")
            raise

    async def close(self) -> None:
        """Close the Gremlin connection."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                logger.exception("Error closing Gremlin connection")
            finally:
                self._connection = None
                self._g = None

    def __repr__(self) -> str:
        return f"GremlinGraphBackend(endpoint={self._endpoint!r})"
