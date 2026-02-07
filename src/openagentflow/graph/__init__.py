"""Graph-based execution tracing for Open Agent Flow.

Provides graph backends for recording agent, chain, and swarm execution
DAGs. The default backend uses SQLite (zero external dependencies).

Quick Start:
    from openagentflow.graph import get_default_backend, SQLiteGraphBackend

    # Default (file-based, persistent)
    backend = get_default_backend()

    # In-memory (ephemeral, great for testing)
    backend = SQLiteGraphBackend(":memory:")

    # Optional: Gremlin for production
    from openagentflow.graph import GremlinGraphBackend
    backend = GremlinGraphBackend("ws://localhost:8182/gremlin")
"""

from openagentflow.graph.base import GraphBackend
from openagentflow.graph.sqlite import SQLiteGraphBackend

# Lazy import for Gremlin to avoid import-time failure
# when gremlinpython is not installed.
_default_backend: SQLiteGraphBackend | None = None


def get_default_backend() -> SQLiteGraphBackend:
    """Get or create the default SQLite graph backend.

    Returns a singleton SQLiteGraphBackend using the default
    database path (~/.openagentflow/traces.db).
    """
    global _default_backend
    if _default_backend is None:
        _default_backend = SQLiteGraphBackend()
    return _default_backend


def reset_default_backend() -> None:
    """Reset the default backend singleton (primarily for testing)."""
    global _default_backend
    _default_backend = None


# Lazy accessor for GremlinGraphBackend so it doesn't fail at import
# if gremlinpython is not installed.
def __getattr__(name: str):
    if name == "GremlinGraphBackend":
        from openagentflow.graph.gremlin import GremlinGraphBackend

        return GremlinGraphBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GraphBackend",
    "SQLiteGraphBackend",
    "GremlinGraphBackend",
    "get_default_backend",
    "reset_default_backend",
]
