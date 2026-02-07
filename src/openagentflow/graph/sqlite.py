"""SQLite-based graph backend for execution tracing (zero dependencies).

Uses Python's built-in sqlite3 module with async wrappers via
asyncio.loop.run_in_executor to keep all operations non-blocking.

Default storage: ~/.openagentflow/traces.db
In-memory mode: pass ":memory:" as the path.

This is the default backend -- it requires no external packages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

# Default database path
_DEFAULT_DB_DIR = os.path.join(os.path.expanduser("~"), ".openagentflow")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "traces.db")

# Schema version for future migrations
_SCHEMA_VERSION = 1

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS vertices (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}',
    run_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    label TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES vertices(id),
    FOREIGN KEY (target_id) REFERENCES vertices(id)
);

CREATE INDEX IF NOT EXISTS idx_vertices_run_id ON vertices(run_id);
CREATE INDEX IF NOT EXISTS idx_edges_source_id ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target_id ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_vertices_label ON vertices(label);
CREATE INDEX IF NOT EXISTS idx_edges_label ON edges(label);
"""


class SQLiteGraphBackend:
    """SQLite-backed graph storage for execution tracing.

    Thread-safe and async-friendly. All database operations are dispatched
    to a thread pool executor so they never block the event loop.

    Example:
        # File-based (persistent)
        backend = SQLiteGraphBackend()

        # In-memory (ephemeral, great for tests)
        backend = SQLiteGraphBackend(":memory:")

        await backend.add_vertex("agent-1", "agent", {
            "name": "planner",
            "run_id": "run-abc",
            "model": "claude-sonnet-4-20250514",
        })
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the SQLite graph backend.

        Args:
            db_path: Path to the SQLite database file.
                Defaults to ~/.openagentflow/traces.db.
                Use ":memory:" for an in-memory database.
        """
        if db_path is None:
            db_path = _DEFAULT_DB_PATH

        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    @property
    def db_path(self) -> str:
        """Return the database path."""
        return self._db_path

    def _ensure_directory(self) -> None:
        """Create the database directory if it does not exist."""
        if self._db_path != ":memory:":
            db_dir = os.path.dirname(self._db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the SQLite connection (called from executor thread)."""
        if self._conn is None:
            self._ensure_directory()
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit for reads, explicit tx for writes
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent read performance
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_schema(self) -> None:
        """Initialize the database schema (called from executor thread)."""
        conn = self._get_connection()
        conn.executescript(_CREATE_TABLES_SQL)
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Ensure the schema has been created."""
        if not self._initialized:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._init_schema)

    async def _run_in_executor(self, func: Any, *args: Any) -> Any:
        """Run a synchronous function in the default thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(func, *args))

    # -------------------------------------------------------------------------
    # Synchronous helpers (run inside executor)
    # -------------------------------------------------------------------------

    def _add_vertex_sync(
        self,
        vertex_id: str,
        label: str,
        properties_json: str,
        run_id: str | None,
        created_at: str,
    ) -> None:
        conn = self._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO vertices (id, label, properties_json, run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (vertex_id, label, properties_json, run_id, created_at),
        )

    def _update_vertex_sync(
        self,
        vertex_id: str,
        properties_json: str,
    ) -> None:
        conn = self._get_connection()
        # Merge new properties into existing
        cursor = conn.execute(
            "SELECT properties_json FROM vertices WHERE id = ?",
            (vertex_id,),
        )
        row = cursor.fetchone()
        if row is None:
            logger.warning("Cannot update non-existent vertex: %s", vertex_id)
            return

        existing = json.loads(row["properties_json"])
        new_props = json.loads(properties_json)
        existing.update(new_props)

        conn.execute(
            "UPDATE vertices SET properties_json = ? WHERE id = ?",
            (json.dumps(existing, default=str), vertex_id),
        )

    def _add_edge_sync(
        self,
        source_id: str,
        target_id: str,
        label: str,
        properties_json: str,
        created_at: str,
    ) -> None:
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO edges (source_id, target_id, label, properties_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, label, properties_json, created_at),
        )

    def _query_vertex_sync(self, vertex_id: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT id, label, properties_json, run_id, created_at "
            "FROM vertices WHERE id = ?",
            (vertex_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "label": row["label"],
            "properties": json.loads(row["properties_json"]),
            "run_id": row["run_id"],
            "created_at": row["created_at"],
        }

    def _query_children_sync(self, vertex_id: str) -> list[dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT v.id, v.label, v.properties_json, v.run_id, v.created_at,
                   e.label AS edge_label, e.properties_json AS edge_properties_json
            FROM edges e
            JOIN vertices v ON e.target_id = v.id
            WHERE e.source_id = ?
            ORDER BY e.id
            """,
            (vertex_id,),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "label": row["label"],
                "properties": json.loads(row["properties_json"]),
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "edge_label": row["edge_label"],
                "edge_properties": json.loads(row["edge_properties_json"]),
            })
        return results

    def _query_parents_sync(self, vertex_id: str) -> list[dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT v.id, v.label, v.properties_json, v.run_id, v.created_at,
                   e.label AS edge_label, e.properties_json AS edge_properties_json
            FROM edges e
            JOIN vertices v ON e.source_id = v.id
            WHERE e.target_id = ?
            ORDER BY e.id
            """,
            (vertex_id,),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "label": row["label"],
                "properties": json.loads(row["properties_json"]),
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "edge_label": row["edge_label"],
                "edge_properties": json.loads(row["edge_properties_json"]),
            })
        return results

    def _get_full_trace_sync(self, run_id: str) -> dict[str, Any]:
        conn = self._get_connection()

        # Get all vertices for this run
        v_cursor = conn.execute(
            "SELECT id, label, properties_json, run_id, created_at "
            "FROM vertices WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        )
        vertices = []
        vertex_ids = set()
        for row in v_cursor.fetchall():
            vertex_ids.add(row["id"])
            vertices.append({
                "id": row["id"],
                "label": row["label"],
                "properties": json.loads(row["properties_json"]),
                "run_id": row["run_id"],
                "created_at": row["created_at"],
            })

        # Get all edges between these vertices
        if not vertex_ids:
            return {"vertices": [], "edges": []}

        placeholders = ",".join("?" for _ in vertex_ids)
        ids_list = list(vertex_ids)
        e_cursor = conn.execute(
            f"SELECT source_id, target_id, label, properties_json, created_at "
            f"FROM edges WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders}) "
            f"ORDER BY id",
            ids_list + ids_list,
        )
        edges = []
        for row in e_cursor.fetchall():
            edges.append({
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "label": row["label"],
                "properties": json.loads(row["properties_json"]),
                "created_at": row["created_at"],
            })

        return {"vertices": vertices, "edges": edges}

    def _clear_sync(self) -> None:
        conn = self._get_connection()
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM vertices")

    def _close_sync(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    # -------------------------------------------------------------------------
    # Async public API
    # -------------------------------------------------------------------------

    async def add_vertex(
        self,
        vertex_id: str,
        label: str,
        properties: dict[str, Any],
    ) -> None:
        """Add a vertex to the graph.

        Args:
            vertex_id: Unique vertex identifier.
            label: Vertex type label.
            properties: Vertex properties. Should include "run_id".
        """
        await self._ensure_initialized()
        run_id = properties.get("run_id")
        created_at = datetime.now(timezone.utc).isoformat()
        properties_json = json.dumps(properties, default=str)
        await self._run_in_executor(
            self._add_vertex_sync,
            vertex_id,
            label,
            properties_json,
            run_id,
            created_at,
        )

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
        await self._ensure_initialized()
        properties_json = json.dumps(properties, default=str)
        await self._run_in_executor(
            self._update_vertex_sync,
            vertex_id,
            properties_json,
        )

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
            label: Edge type label.
            properties: Edge properties.
        """
        await self._ensure_initialized()
        created_at = datetime.now(timezone.utc).isoformat()
        properties_json = json.dumps(properties, default=str)
        await self._run_in_executor(
            self._add_edge_sync,
            source_id,
            target_id,
            label,
            properties_json,
            created_at,
        )

    async def query_vertex(self, vertex_id: str) -> dict[str, Any] | None:
        """Query a single vertex by ID."""
        await self._ensure_initialized()
        return await self._run_in_executor(self._query_vertex_sync, vertex_id)

    async def query_children(self, vertex_id: str) -> list[dict[str, Any]]:
        """Query all child vertices (outgoing edges)."""
        await self._ensure_initialized()
        return await self._run_in_executor(self._query_children_sync, vertex_id)

    async def query_parents(self, vertex_id: str) -> list[dict[str, Any]]:
        """Query all parent vertices (incoming edges)."""
        await self._ensure_initialized()
        return await self._run_in_executor(self._query_parents_sync, vertex_id)

    async def get_full_trace(self, run_id: str) -> dict[str, Any]:
        """Get the full execution DAG for a run.

        Returns:
            Dict with "vertices" and "edges" lists.
        """
        await self._ensure_initialized()
        return await self._run_in_executor(self._get_full_trace_sync, run_id)

    async def clear(self) -> None:
        """Delete all vertices and edges."""
        await self._ensure_initialized()
        await self._run_in_executor(self._clear_sync)

    async def close(self) -> None:
        """Close the database connection."""
        await self._run_in_executor(self._close_sync)

    def __repr__(self) -> str:
        return f"SQLiteGraphBackend(db_path={self._db_path!r})"
