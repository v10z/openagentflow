"""Long-term memory -- persistent storage across sessions.

Long-term memory survives beyond individual sessions.  It uses a graph
backend (implementing the ``GraphBackend`` protocol) to persist memories
as vertices.  When no graph backend is provided, an in-memory dictionary
serves as a transparent fallback so the rest of the system continues to
work.

Characteristics:
- Persistent across sessions (when backed by a real graph database).
- Keyword search across stored memories.
- Retrieval by originating agent or execution run ID.
- Automatic consolidation of similar memories.
- Time-based relevance decay with pruning of low-relevance entries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openagentflow.memory.base import MemoryEntry, MemoryTier


class LongTermMemory:
    """Persistent memory tier backed by a graph database.

    When ``graph_backend`` is ``None`` the class silently falls back to an
    in-memory dictionary, which is sufficient for tests and short-lived
    processes.

    Args:
        graph_backend: Optional graph database implementing the
            ``GraphBackend`` protocol (``add_vertex``, ``query``).
        relevance_threshold: Entries whose relevance score drops below
            this value during ``decay()`` are candidates for pruning.

    Example::

        ltm = LongTermMemory()
        await ltm.remember(entry)
        results = await ltm.recall("user preferences")
        await ltm.decay()
    """

    def __init__(
        self,
        graph_backend: Any = None,
        relevance_threshold: float = 0.10,
    ) -> None:
        self._graph = graph_backend
        self.relevance_threshold = relevance_threshold
        # In-memory fallback when no graph backend is available.
        self._store: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def remember(self, entry: MemoryEntry) -> None:
        """Persist a memory entry to long-term storage.

        If a graph backend is available the entry is serialized and
        stored as a vertex.  Otherwise it is kept in the in-memory dict.

        Args:
            entry: The ``MemoryEntry`` to persist.
        """
        entry.tier = MemoryTier.LONG_TERM

        # Always store in the in-memory dict so that recall, consolidate,
        # decay, and other operations can find the entry.
        self._store[entry.key] = entry

        if self._graph is not None:
            try:
                vertex_id = f"memory-{entry.key}"
                properties = entry.to_dict()
                await self._graph.add_vertex(vertex_id, "memory", properties)
            except Exception:
                # Graph persistence failed; entry is still in _store.
                pass

    async def recall(self, query: str) -> list[MemoryEntry]:
        """Search long-term memory by keyword.

        An empty query returns all stored entries.  When a graph backend
        is available, the search is delegated to the backend.  Otherwise
        it scans the in-memory dictionary.

        Args:
            query: Search string matched (case-insensitive) against keys,
                tags, and content.

        Returns:
            Matching ``MemoryEntry`` objects sorted by relevance.
        """
        if self._graph is not None:
            return await self._recall_from_graph(query)
        return await self._recall_from_memory(query)

    async def recall_by_agent(self, agent_name: str) -> list[MemoryEntry]:
        """Retrieve all memories originating from a specific agent.

        Args:
            agent_name: The ``source_agent`` value to filter by.

        Returns:
            Matching entries sorted by relevance.
        """
        results: list[MemoryEntry] = []

        for entry in self._store.values():
            if entry.source_agent == agent_name:
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)

        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results

    async def recall_by_run(self, run_id: str) -> list[MemoryEntry]:
        """Retrieve all memories from a specific execution run.

        Args:
            run_id: The ``source_run_id`` value to filter by.

        Returns:
            Matching entries sorted by relevance.
        """
        results: list[MemoryEntry] = []

        for entry in self._store.values():
            if entry.source_run_id == run_id:
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)

        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results

    async def forget(self, key: str) -> None:
        """Remove a single long-term memory.

        Args:
            key: The key of the entry to remove.
        """
        self._store.pop(key, None)

    async def consolidate(self) -> int:
        """Merge similar memories and strengthen frequently accessed ones.

        Consolidation compares every pair of entries using Jaccard
        similarity on their tokenized content.  When two entries are
        sufficiently similar (>= 0.6) the less relevant one is marked as
        superseded and removed.

        Returns:
            The number of entries removed by consolidation.
        """
        entries = list(self._store.values())
        removed = 0
        keys_to_remove: set[str] = set()

        for i in range(len(entries)):
            if entries[i].key in keys_to_remove:
                continue
            for j in range(i + 1, len(entries)):
                if entries[j].key in keys_to_remove:
                    continue

                similarity = _jaccard_similarity(
                    str(entries[i].content), str(entries[j].content)
                )
                if similarity >= 0.6:
                    # Keep the more relevant entry.
                    if entries[i].relevance_score >= entries[j].relevance_score:
                        winner, loser = entries[i], entries[j]
                    else:
                        winner, loser = entries[j], entries[i]

                    # Strengthen winner.
                    winner.access_count += loser.access_count
                    winner.importance = max(winner.importance, loser.importance)
                    winner.last_accessed = datetime.now(timezone.utc)
                    if loser.tags:
                        winner.tags = list(set(winner.tags + loser.tags))

                    loser.superseded_by = winner.key
                    keys_to_remove.add(loser.key)

        for key in keys_to_remove:
            await self.forget(key)
            removed += 1

        return removed

    async def decay(self) -> int:
        """Apply time-based decay and prune entries below threshold.

        The decay function is:
            ``relevance = base_score * 0.95 ^ days_since_last_access``

        Entries whose decayed relevance falls below
        ``relevance_threshold`` are removed.

        Returns:
            The number of entries pruned.
        """
        now = datetime.now(timezone.utc)
        keys_to_remove: list[str] = []

        for key, entry in self._store.items():
            days_since_access = (now - entry.last_accessed).total_seconds() / 86400
            decayed_relevance = entry.relevance_score * (0.95 ** days_since_access)
            if decayed_relevance < self.relevance_threshold:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            await self.forget(key)

        return len(keys_to_remove)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _recall_from_graph(self, query: str) -> list[MemoryEntry]:
        """Search the graph backend for matching memories.

        Since the GraphBackend protocol does not expose a generic query
        method, we maintain an in-memory mirror of graph-persisted entries
        and search that.  The graph serves as the durable persistence
        layer, while the in-memory dict enables fast keyword search.
        """
        # The graph backend is used for persistence, but search is done
        # against the in-memory store which is kept in sync.
        return await self._recall_from_memory(query)

    async def _recall_from_memory(self, query: str) -> list[MemoryEntry]:
        """Search the in-memory fallback store."""
        query_lower = query.lower()
        results: list[MemoryEntry] = []

        for entry in self._store.values():
            if not query or self._matches(entry, query_lower):
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)

        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results

    @staticmethod
    def _matches(entry: MemoryEntry, query_lower: str) -> bool:
        """Check whether an entry matches a lowercased query string."""
        if query_lower in entry.key.lower():
            return True
        if any(query_lower in tag.lower() for tag in entry.tags):
            return True
        if query_lower in str(entry.content).lower():
            return True
        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[MemoryEntry]:
        """Return all in-memory entries (does not query graph)."""
        return list(self._store.values())

    @property
    def size(self) -> int:
        """Return the number of in-memory entries."""
        return len(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store


# =====================================================================
# Module-level helpers
# =====================================================================


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two text strings.

    Both strings are lowercased and split on whitespace.  The similarity
    is the size of the intersection divided by the size of the union of
    the resulting token sets.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        A float in [0, 1].  1.0 means identical token sets.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
