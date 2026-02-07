"""Regression tests for the SQLite graph backend.

Tests the SQLiteGraphBackend class which provides persistent execution
tracing using Python's built-in sqlite3 module with async wrappers.

All tests use in-memory SQLite databases (":memory:") so they require
no filesystem access and are fully isolated from each other.
"""

from __future__ import annotations

import pytest

from openagentflow.graph.sqlite import SQLiteGraphBackend


@pytest.fixture
async def backend():
    """Create a fresh in-memory SQLite backend for each test."""
    db = SQLiteGraphBackend(":memory:")
    yield db
    await db.close()


class TestSQLiteGraphBackend:
    """Tests for the SQLiteGraphBackend class."""

    async def test_create_tables(self, backend: SQLiteGraphBackend):
        """Verify that the backend initializes the schema on first use.

        The backend lazily creates tables on the first async operation.
        After a simple vertex insertion (which triggers _ensure_initialized),
        the internal _initialized flag should be True and the connection
        should be established.
        """
        # Trigger initialization by performing an operation.
        await backend.add_vertex("v1", "test", {"run_id": "run-1"})
        assert backend._initialized is True
        assert backend._conn is not None

    async def test_add_vertex(self, backend: SQLiteGraphBackend):
        """Verify that a vertex can be added and subsequently retrieved.

        Adds a vertex with known properties and confirms that query_vertex
        returns a dict with the correct id, label, properties, and run_id.
        """
        await backend.add_vertex(
            "agent-1",
            "agent",
            {"run_id": "run-abc", "name": "planner", "model": "claude-sonnet"},
        )

        result = await backend.query_vertex("agent-1")
        assert result is not None
        assert result["id"] == "agent-1"
        assert result["label"] == "agent"
        assert result["properties"]["name"] == "planner"
        assert result["properties"]["model"] == "claude-sonnet"
        assert result["run_id"] == "run-abc"

    async def test_add_edge(self, backend: SQLiteGraphBackend):
        """Verify that edges can be added between two vertices.

        Creates two vertices and an edge connecting them, then confirms
        that the child vertex is reachable from the parent via query_children.
        """
        await backend.add_vertex("v1", "agent", {"run_id": "run-1"})
        await backend.add_vertex("v2", "tool_call", {"run_id": "run-1"})
        await backend.add_edge("v1", "v2", "invoked", {"step": 1})

        children = await backend.query_children("v1")
        assert len(children) == 1
        assert children[0]["id"] == "v2"
        assert children[0]["edge_label"] == "invoked"
        assert children[0]["edge_properties"]["step"] == 1

    async def test_query_children(self, backend: SQLiteGraphBackend):
        """Verify that query_children returns all outgoing edges.

        Creates a parent vertex with two children and confirms both are
        returned in insertion order.
        """
        await backend.add_vertex("parent", "agent", {"run_id": "run-1"})
        await backend.add_vertex("child-a", "tool_call", {"run_id": "run-1"})
        await backend.add_vertex("child-b", "tool_call", {"run_id": "run-1"})
        await backend.add_edge("parent", "child-a", "called", {})
        await backend.add_edge("parent", "child-b", "called", {})

        children = await backend.query_children("parent")
        assert len(children) == 2
        child_ids = {c["id"] for c in children}
        assert child_ids == {"child-a", "child-b"}

    async def test_query_parents(self, backend: SQLiteGraphBackend):
        """Verify that query_parents returns all incoming edges.

        Creates two parent vertices pointing to the same child and confirms
        both parents are returned.
        """
        await backend.add_vertex("parent-1", "agent", {"run_id": "run-1"})
        await backend.add_vertex("parent-2", "agent", {"run_id": "run-1"})
        await backend.add_vertex("child", "result", {"run_id": "run-1"})
        await backend.add_edge("parent-1", "child", "produced", {})
        await backend.add_edge("parent-2", "child", "produced", {})

        parents = await backend.query_parents("child")
        assert len(parents) == 2
        parent_ids = {p["id"] for p in parents}
        assert parent_ids == {"parent-1", "parent-2"}

    async def test_get_full_trace(self, backend: SQLiteGraphBackend):
        """Verify that get_full_trace returns the complete DAG for a run.

        Creates a three-vertex, two-edge trace and checks that the full
        trace dict contains all vertices and edges belonging to the run_id.
        """
        run_id = "run-trace-1"
        await backend.add_vertex("a", "agent", {"run_id": run_id, "name": "a"})
        await backend.add_vertex("b", "tool_call", {"run_id": run_id, "name": "b"})
        await backend.add_vertex("c", "result", {"run_id": run_id, "name": "c"})
        await backend.add_edge("a", "b", "invoked", {})
        await backend.add_edge("b", "c", "returned", {})

        trace = await backend.get_full_trace(run_id)
        assert "vertices" in trace
        assert "edges" in trace
        assert len(trace["vertices"]) == 3
        assert len(trace["edges"]) == 2

        vertex_ids = {v["id"] for v in trace["vertices"]}
        assert vertex_ids == {"a", "b", "c"}

    async def test_get_full_trace_empty_run(self, backend: SQLiteGraphBackend):
        """Verify that get_full_trace for a nonexistent run returns empty lists."""
        trace = await backend.get_full_trace("nonexistent-run")
        assert trace == {"vertices": [], "edges": []}

    async def test_update_vertex(self, backend: SQLiteGraphBackend):
        """Verify that update_vertex merges new properties into existing ones.

        Creates a vertex, updates it with new properties, and confirms
        that both old and new properties are present.
        """
        await backend.add_vertex(
            "v1", "agent", {"run_id": "run-1", "status": "running"}
        )

        await backend.update_vertex("v1", {"status": "completed", "duration_ms": 150.5})

        result = await backend.query_vertex("v1")
        assert result is not None
        assert result["properties"]["status"] == "completed"
        assert result["properties"]["duration_ms"] == 150.5
        # Original run_id should still be present.
        assert result["properties"]["run_id"] == "run-1"

    async def test_update_nonexistent_vertex(self, backend: SQLiteGraphBackend):
        """Verify that updating a nonexistent vertex does not raise an error.

        The implementation logs a warning but does not crash.
        """
        # Should not raise.
        await backend.update_vertex("nonexistent-id", {"foo": "bar"})

    async def test_clear(self, backend: SQLiteGraphBackend):
        """Verify that clear() removes all vertices and edges.

        After inserting data and calling clear, both query_vertex and
        get_full_trace should return empty results.
        """
        await backend.add_vertex("v1", "agent", {"run_id": "run-1"})
        await backend.add_vertex("v2", "tool", {"run_id": "run-1"})
        await backend.add_edge("v1", "v2", "used", {})

        await backend.clear()

        assert await backend.query_vertex("v1") is None
        assert await backend.query_vertex("v2") is None
        trace = await backend.get_full_trace("run-1")
        assert trace == {"vertices": [], "edges": []}

    async def test_in_memory_mode(self):
        """Verify that ':memory:' path creates an in-memory database.

        The backend should work correctly without touching the filesystem
        and its db_path property should return ':memory:'.
        """
        db = SQLiteGraphBackend(":memory:")
        assert db.db_path == ":memory:"

        await db.add_vertex("v1", "test", {"run_id": "run-mem"})
        result = await db.query_vertex("v1")
        assert result is not None
        assert result["id"] == "v1"

        await db.close()

    async def test_empty_properties(self, backend: SQLiteGraphBackend):
        """Verify that vertices can be created with minimal / empty properties.

        An empty properties dict should be stored as '{}' and should not
        cause errors during retrieval.
        """
        await backend.add_vertex("v-empty", "node", {})
        result = await backend.query_vertex("v-empty")
        assert result is not None
        assert result["properties"] == {}
        assert result["run_id"] is None

    async def test_nonexistent_vertex(self, backend: SQLiteGraphBackend):
        """Verify that querying a nonexistent vertex returns None.

        query_vertex should return None rather than raising an exception
        when the requested vertex ID does not exist.
        """
        result = await backend.query_vertex("does-not-exist")
        assert result is None

    async def test_query_children_of_leaf(self, backend: SQLiteGraphBackend):
        """Verify that query_children on a leaf vertex returns an empty list."""
        await backend.add_vertex("leaf", "result", {"run_id": "run-1"})
        children = await backend.query_children("leaf")
        assert children == []

    async def test_query_parents_of_root(self, backend: SQLiteGraphBackend):
        """Verify that query_parents on a root vertex returns an empty list."""
        await backend.add_vertex("root", "agent", {"run_id": "run-1"})
        parents = await backend.query_parents("root")
        assert parents == []

    async def test_vertex_overwrite(self, backend: SQLiteGraphBackend):
        """Verify that adding a vertex with the same ID replaces the old one.

        The SQLite backend uses INSERT OR REPLACE so re-adding a vertex
        with the same ID should overwrite properties.
        """
        await backend.add_vertex("v1", "agent", {"run_id": "run-1", "version": 1})
        await backend.add_vertex("v1", "agent", {"run_id": "run-1", "version": 2})

        result = await backend.query_vertex("v1")
        assert result is not None
        assert result["properties"]["version"] == 2

    async def test_repr(self, backend: SQLiteGraphBackend):
        """Verify the __repr__ output format."""
        r = repr(backend)
        assert "SQLiteGraphBackend" in r
        assert ":memory:" in r
