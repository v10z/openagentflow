"""Fleeting memory -- per-turn working scratchpad.

Fleeting memory is the most ephemeral tier.  It exists only for the
duration of a single ReAct turn and is automatically cleared at the end
of each iteration.  Think of it as a whiteboard the agent scribbles on
while reasoning through a single step.

Characteristics:
- Simple key/value dictionary store.
- Zero persistence: data is discarded when ``clear()`` is called.
- No importance scoring or garbage collection needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openagentflow.memory.base import MemoryEntry, MemoryTier


class FleetingMemory:
    """Per-turn scratchpad memory that is wiped after each ReAct turn.

    This is intentionally the simplest tier: a plain dictionary.  The
    executor calls ``clear()`` at the end of every turn so that stale
    intermediate results never leak into subsequent reasoning steps.

    Example::

        mem = FleetingMemory()
        await mem.remember("scratch", {"partial_result": 42})
        results = await mem.recall("scratch")
        await mem.clear()  # everything gone
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def remember(self, key: str, value: Any) -> None:
        """Store a value in the fleeting scratchpad.

        If the key already exists it is silently overwritten.

        Args:
            key: Identifier for the scratchpad entry.
            value: Any Python object to store.
        """
        self._store[key] = value

    async def recall(self, query: str) -> list[MemoryEntry]:
        """Retrieve entries matching a query string.

        Matching is performed against both keys and the string
        representation of values.  An empty query returns all entries.

        Args:
            query: Search string matched against keys and content.

        Returns:
            A list of ``MemoryEntry`` wrappers around matching entries.
        """
        results: list[MemoryEntry] = []
        query_lower = query.lower()
        now = datetime.now(timezone.utc)

        for key, value in self._store.items():
            if not query or query_lower in key.lower() or query_lower in str(value).lower():
                entry = MemoryEntry(
                    key=key,
                    content=value,
                    tier=MemoryTier.FLEETING,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    importance=0.0,
                )
                results.append(entry)

        return results

    async def forget(self, key: str) -> None:
        """Remove a single entry from the scratchpad.

        Silently succeeds if the key does not exist.

        Args:
            key: Identifier of the entry to remove.
        """
        self._store.pop(key, None)

    async def clear(self) -> None:
        """Wipe all fleeting memory.

        Called by the executor at the end of every ReAct turn so that
        temporary reasoning artifacts do not persist.
        """
        self._store.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of entries currently stored."""
        return len(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store
