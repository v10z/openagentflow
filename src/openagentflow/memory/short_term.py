"""Short-term memory -- session / conversation context.

Short-term memory retains information for the duration of a session (or
conversation).  It implements a sliding window of ``MemoryEntry`` objects
with automatic compression: when the entry count exceeds a configurable
maximum the oldest, least-relevant entries are either pruned or (when an
LLM provider is available) progressively summarized into a single
condensed entry.

Characteristics:
- Configurable maximum size (default 100 entries).
- Automatic pruning of entries below a relevance threshold on refresh.
- LLM-powered progressive summarization (optional).
- Efficient substring and tag-based search.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openagentflow.memory.base import MemoryEntry, MemoryTier


class ShortTermMemory:
    """Sliding-window session memory with auto-compression.

    Args:
        max_entries: Maximum number of entries to retain before triggering
            compression.
        relevance_threshold: Entries with a relevance score below this
            value are pruned during a refresh cycle.
        refresh_interval: Number of ``remember`` calls between automatic
            refresh / prune cycles.

    Example::

        stm = ShortTermMemory(max_entries=50)
        await stm.remember(entry)
        recent = await stm.get_recent(5)
    """

    def __init__(
        self,
        max_entries: int = 100,
        relevance_threshold: float = 0.15,
        refresh_interval: int = 10,
    ) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._insertion_order: list[str] = []
        self.max_entries = max_entries
        self.relevance_threshold = relevance_threshold
        self.refresh_interval = refresh_interval
        self._ops_since_refresh: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def remember(self, entry: MemoryEntry) -> None:
        """Add or update an entry in short-term memory.

        If an entry with the same key already exists its content,
        importance, and access metadata are updated.

        A refresh cycle is triggered automatically every
        ``refresh_interval`` insertions or when the store exceeds
        ``max_entries``.

        Args:
            entry: The ``MemoryEntry`` to store.
        """
        if entry.key in self._entries:
            # Update existing entry.
            existing = self._entries[entry.key]
            existing.content = entry.content
            existing.importance = max(existing.importance, entry.importance)
            existing.last_accessed = datetime.now(timezone.utc)
            existing.access_count += 1
            if entry.tags:
                existing.tags = list(set(existing.tags + entry.tags))
        else:
            self._entries[entry.key] = entry
            self._insertion_order.append(entry.key)

        self._ops_since_refresh += 1

        # Auto-refresh when due or when over capacity.
        if (
            self._ops_since_refresh >= self.refresh_interval
            or len(self._entries) > self.max_entries
        ):
            await self._refresh()

    async def recall(self, query: str) -> list[MemoryEntry]:
        """Search short-term memory by key, tags, or content substring.

        An empty query returns all entries.

        Args:
            query: Search string matched (case-insensitive) against the
                entry key, tags, and string representation of content.

        Returns:
            Matching ``MemoryEntry`` objects sorted by relevance
            (descending).
        """
        query_lower = query.lower()
        results: list[MemoryEntry] = []

        for entry in self._entries.values():
            if not query:
                # Empty query -> return everything.
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)
                continue

            # Match against key.
            if query_lower in entry.key.lower():
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)
                continue

            # Match against tags.
            if any(query_lower in tag.lower() for tag in entry.tags):
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)
                continue

            # Match against content.
            if query_lower in str(entry.content).lower():
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)
                continue

        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results

    async def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the *n* most recently inserted entries.

        Args:
            n: Number of entries to return.

        Returns:
            Up to *n* entries in reverse insertion order (newest first).
        """
        recent_keys = self._insertion_order[-n:]
        recent_keys.reverse()
        results: list[MemoryEntry] = []
        for key in recent_keys:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                results.append(entry)
        return results

    async def forget(self, key: str) -> None:
        """Remove a single entry by key.

        Silently succeeds if the key does not exist.

        Args:
            key: Identifier of the entry to remove.
        """
        self._entries.pop(key, None)
        if key in self._insertion_order:
            self._insertion_order.remove(key)

    async def summarize(self, llm_provider: Any = None) -> MemoryEntry | None:
        """Compress older entries into a progressive summary.

        When an LLM provider is available the oldest entries (beyond the
        most recent ``keep_recent`` verbatim entries) are sent to the LLM
        for summarization.  The resulting summary replaces the compressed
        entries, dramatically reducing token consumption.

        Without an LLM provider, a simple concatenation-based summary is
        created instead.

        Args:
            llm_provider: Optional LLM provider with a ``generate`` method.

        Returns:
            The summary ``MemoryEntry``, or ``None`` if there are too few
            entries to warrant summarization.
        """
        keep_recent = min(10, len(self._entries))
        if len(self._entries) <= keep_recent:
            return None

        # Split into old (to summarize) and recent (to keep verbatim).
        all_keys = list(self._insertion_order)
        old_keys = all_keys[:-keep_recent] if keep_recent > 0 else all_keys
        old_entries = [self._entries[k] for k in old_keys if k in self._entries]

        if not old_entries:
            return None

        # Build text block from old entries.
        text_parts: list[str] = []
        for e in old_entries:
            text_parts.append(f"[{e.key}] {e.content}")
        combined_text = "\n".join(text_parts)

        summary_content: str
        if llm_provider is not None:
            # LLM-powered summarization.
            try:
                from openagentflow.core.types import LLMProvider as LP, Message as Msg, ModelConfig

                prompt = (
                    "Summarize the following memory entries into a concise paragraph "
                    "preserving all key facts, decisions, and important details:\n\n"
                    f"{combined_text}"
                )
                # Use a sensible default config; the provider may override
                # the model_id internally.
                default_config = ModelConfig(
                    provider=LP.ANTHROPIC,
                    model_id="claude-sonnet-4-20250514",
                )
                response = await llm_provider.generate(
                    messages=[Msg(role="user", content=prompt)],
                    config=default_config,
                )
                summary_content = response.content
            except Exception:
                # Fallback to simple truncation on any error.
                summary_content = combined_text[:2000]
                if len(combined_text) > 2000:
                    summary_content += "\n... (truncated)"
        else:
            # No LLM -- simple truncation.
            summary_content = combined_text[:2000]
            if len(combined_text) > 2000:
                summary_content += "\n... (truncated)"

        # Remove old entries.
        for key in old_keys:
            self._entries.pop(key, None)
            if key in self._insertion_order:
                self._insertion_order.remove(key)

        # Insert summary entry at the front.
        now = datetime.now(timezone.utc)
        summary_entry = MemoryEntry(
            key="__summary__",
            content=summary_content,
            tier=MemoryTier.SHORT_TERM,
            created_at=now,
            last_accessed=now,
            access_count=0,
            importance=0.8,
            tags=["summary", "auto-generated"],
        )
        # Remove previous summary if present.
        self._entries.pop("__summary__", None)
        if "__summary__" in self._insertion_order:
            self._insertion_order.remove("__summary__")

        self._entries["__summary__"] = summary_entry
        self._insertion_order.insert(0, "__summary__")

        return summary_entry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        """Run a refresh cycle: prune low-relevance entries.

        Entries whose ``relevance_score`` falls below
        ``relevance_threshold`` are removed.  If the store still exceeds
        ``max_entries`` after pruning, the least relevant entries are
        dropped until the size is within bounds.
        """
        self._ops_since_refresh = 0

        # Phase 1: prune below threshold.
        to_remove: list[str] = []
        for key, entry in self._entries.items():
            if entry.relevance_score < self.relevance_threshold:
                to_remove.append(key)

        for key in to_remove:
            self._entries.pop(key, None)
            if key in self._insertion_order:
                self._insertion_order.remove(key)

        # Phase 2: if still over capacity, drop least relevant.
        if len(self._entries) > self.max_entries:
            sorted_entries = sorted(
                self._entries.values(),
                key=lambda e: e.relevance_score,
            )
            excess = len(self._entries) - self.max_entries
            for entry in sorted_entries[:excess]:
                self._entries.pop(entry.key, None)
                if entry.key in self._insertion_order:
                    self._insertion_order.remove(entry.key)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[MemoryEntry]:
        """Return all entries in insertion order."""
        result: list[MemoryEntry] = []
        for key in self._insertion_order:
            entry = self._entries.get(key)
            if entry is not None:
                result.append(entry)
        return result

    @property
    def size(self) -> int:
        """Return the number of entries currently stored."""
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries
