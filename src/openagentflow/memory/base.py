"""Core types and coordinator for the multi-layer memory hierarchy.

This module defines the foundational types (MemoryTier, MemoryEntry) and the
MemoryManager class that coordinates the three-tier memory system:

- Fleeting: per-turn scratchpad, auto-discarded after each ReAct iteration.
- Short-term: session/conversation context with auto-compression.
- Long-term: persistent storage with relevance-based auto-pruning.

The system works without external dependencies. LLM-powered features
(summarization, memory improvement) are gracefully skipped when no LLM
provider is available. Graph-backed persistence falls back to an in-memory
dictionary when no graph backend is provided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryTier(Enum):
    """Enumeration of memory hierarchy tiers.

    Each tier has different retention semantics:
    - FLEETING: destroyed at the end of every ReAct turn.
    - SHORT_TERM: retained for the duration of a session, periodically
      compressed.
    - LONG_TERM: persisted across sessions, subject to relevance-based
      decay and pruning.
    """

    FLEETING = "fleeting"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class MemoryEntry:
    """A single unit of stored memory.

    Attributes:
        key: Unique identifier for the memory.
        content: The stored payload (must be JSON-serializable for
            long-term persistence).
        tier: Which memory tier this entry belongs to.
        created_at: UTC timestamp of creation.
        last_accessed: UTC timestamp of the most recent read.
        access_count: Number of times this entry has been retrieved.
        importance: Score in [0, 1] indicating how important this memory
            is.  Errors and critical events should be near 1.0; routine
            observations near 0.0.
        source_agent: Name of the agent that created this memory.
        source_run_id: Execution run ID that produced this memory.
        tags: Freeform labels for categorical retrieval.
        superseded_by: Key of a newer entry that replaces this one,
            if applicable.
    """

    key: str
    content: Any
    tier: MemoryTier
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 0.5
    source_agent: str = ""
    source_run_id: str = ""
    tags: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    @property
    def relevance_score(self) -> float:
        """Compute a dynamic relevance score combining multiple signals.

        The score blends four factors:
        - Recency (30%): inverse of days since last access.
        - Frequency (20%): capped access count normalized to 10.
        - Importance (40%): the manually-assigned importance weight.
        - Decay (10%): exponential decay based on age since creation.

        Returns:
            A float in roughly [0, 1] representing current relevance.
        """
        now = datetime.now(timezone.utc)
        days_since_access = (now - self.last_accessed).total_seconds() / 86400
        days_since_creation = (now - self.created_at).total_seconds() / 86400
        recency = 1.0 / (1.0 + days_since_access)
        frequency = min(self.access_count / 10.0, 1.0)
        decay = 0.95 ** days_since_creation
        return recency * 0.3 + frequency * 0.2 + self.importance * 0.4 + decay * 0.1

    # ------------------------------------------------------------------
    # Serialization helpers for graph / JSON storage
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a JSON-compatible dictionary.

        Returns:
            A plain dictionary suitable for ``json.dumps``.
        """
        data = {
            "key": self.key,
            "content": self.content,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "source_agent": self.source_agent,
            "source_run_id": self.source_run_id,
            "tags": list(self.tags),
            "superseded_by": self.superseded_by,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Deserialize a ``MemoryEntry`` from a dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict``.

        Returns:
            A reconstructed ``MemoryEntry`` instance.
        """
        return cls(
            key=data["key"],
            content=data["content"],
            tier=MemoryTier(data["tier"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            source_agent=data.get("source_agent", ""),
            source_run_id=data.get("source_run_id", ""),
            tags=data.get("tags", []),
            superseded_by=data.get("superseded_by"),
        )

    def to_json(self) -> str:
        """Serialize the entry to a JSON string.

        Returns:
            A JSON string representation of this entry.
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> MemoryEntry:
        """Deserialize a ``MemoryEntry`` from a JSON string.

        Args:
            json_str: JSON string previously produced by ``to_json``.

        Returns:
            A reconstructed ``MemoryEntry`` instance.
        """
        return cls.from_dict(json.loads(json_str))


class MemoryManager:
    """Coordinates the three-tier memory hierarchy with automatic management.

    The ``MemoryManager`` is the single entry-point for all memory operations.
    It delegates storage and retrieval to the appropriate tier, and runs the
    ``MemoryGarbageCollector`` for automatic lifecycle management (forget,
    summarize, prune, improve).

    The manager works in a fully degraded mode when optional dependencies are
    absent:
    - Without an ``llm_provider``, LLM-powered summarization and improvement
      are skipped.
    - Without a ``graph_backend``, long-term memory falls back to an
      in-memory dictionary.

    Args:
        graph_backend: Optional graph database backend implementing the
            ``GraphBackend`` protocol.  Used by ``LongTermMemory`` for
            persistent storage.
        llm_provider: Optional LLM provider implementing
            ``BaseLLMProvider``.  Used by the garbage collector for
            summarization and memory improvement.

    Example::

        manager = MemoryManager()
        await manager.remember("user_name", "Alice", tier=MemoryTier.SHORT_TERM)
        results = await manager.recall("user_name")
        await manager.run_gc()
    """

    def __init__(
        self,
        graph_backend: Any = None,
        llm_provider: Any = None,
    ) -> None:
        # Import siblings here to avoid circular imports at module level.
        from openagentflow.memory.fleeting import FleetingMemory
        from openagentflow.memory.gc import MemoryGarbageCollector
        from openagentflow.memory.long_term import LongTermMemory
        from openagentflow.memory.short_term import ShortTermMemory

        self.fleeting = FleetingMemory()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(graph_backend=graph_backend)
        self.gc = MemoryGarbageCollector(self, llm_provider=llm_provider)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def remember(
        self,
        key: str,
        value: Any,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        importance: float = 0.5,
        tags: list[str] | None = None,
        source_agent: str = "",
        source_run_id: str = "",
    ) -> MemoryEntry:
        """Store a memory in the specified tier.

        Args:
            key: Unique identifier for the memory.
            value: The content to store.
            tier: Target memory tier (default: SHORT_TERM).
            importance: Importance weight in [0, 1].
            tags: Optional categorical labels.
            source_agent: Name of the originating agent.
            source_run_id: Execution run ID.

        Returns:
            The newly created ``MemoryEntry``.
        """
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            key=key,
            content=value,
            tier=tier,
            created_at=now,
            last_accessed=now,
            access_count=0,
            importance=importance,
            source_agent=source_agent,
            source_run_id=source_run_id,
            tags=tags or [],
        )

        if tier == MemoryTier.FLEETING:
            await self.fleeting.remember(key, value)
        elif tier == MemoryTier.SHORT_TERM:
            await self.short_term.remember(entry)
        elif tier == MemoryTier.LONG_TERM:
            await self.long_term.remember(entry)

        return entry

    async def recall(
        self,
        query: str,
        tiers: list[MemoryTier] | None = None,
    ) -> list[MemoryEntry]:
        """Search for memories across one or more tiers.

        Args:
            query: A search string matched against keys, tags, and content.
            tiers: Which tiers to search.  Defaults to all tiers.

        Returns:
            A list of matching ``MemoryEntry`` objects sorted by relevance
            (highest first).
        """
        tiers = tiers or [MemoryTier.FLEETING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]
        results: list[MemoryEntry] = []

        if MemoryTier.FLEETING in tiers:
            results.extend(await self.fleeting.recall(query))

        if MemoryTier.SHORT_TERM in tiers:
            results.extend(await self.short_term.recall(query))

        if MemoryTier.LONG_TERM in tiers:
            results.extend(await self.long_term.recall(query))

        # Sort by relevance descending.
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results

    async def forget(
        self,
        key: str,
        tier: MemoryTier | None = None,
    ) -> None:
        """Explicitly remove a memory.

        Args:
            key: The key of the memory to forget.
            tier: If given, remove only from this tier.  Otherwise remove
                from all tiers.
        """
        tiers = [tier] if tier else [MemoryTier.FLEETING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]

        if MemoryTier.FLEETING in tiers:
            await self.fleeting.forget(key)

        if MemoryTier.SHORT_TERM in tiers:
            await self.short_term.forget(key)

        if MemoryTier.LONG_TERM in tiers:
            await self.long_term.forget(key)

    async def get_context_window(self, max_tokens: int) -> list[MemoryEntry]:
        """Build an optimal context window for LLM consumption.

        Selects the most relevant memories from short-term and long-term
        tiers that fit within the given token budget (approximated as
        4 characters per token).

        Args:
            max_tokens: Maximum number of tokens the context may consume.

        Returns:
            A list of ``MemoryEntry`` objects fitting within the budget.
        """
        # Gather candidates from short-term and long-term.
        candidates: list[MemoryEntry] = []
        candidates.extend(await self.short_term.recall(""))
        candidates.extend(await self.long_term.recall(""))

        # Sort by relevance descending.
        candidates.sort(key=lambda e: e.relevance_score, reverse=True)

        # Approximate token budget (4 chars ~ 1 token).
        budget = max_tokens * 4
        selected: list[MemoryEntry] = []
        used = 0

        for entry in candidates:
            content_len = len(str(entry.content))
            if used + content_len > budget:
                continue
            selected.append(entry)
            used += content_len

        return selected

    async def run_gc(self) -> None:
        """Run a full garbage-collection cycle across all tiers.

        This triggers auto-forget, pruning, deduplication, and (if an
        LLM provider is available) memory improvement.
        """
        await self.gc.auto_forget(MemoryTier.SHORT_TERM)
        await self.gc.auto_forget(MemoryTier.LONG_TERM)
        await self.gc.prune_duplicates(MemoryTier.SHORT_TERM)
        await self.gc.prune_duplicates(MemoryTier.LONG_TERM)
        await self.gc.prune_by_staleness(MemoryTier.LONG_TERM)
        await self.gc.improve_memory(MemoryTier.LONG_TERM)
