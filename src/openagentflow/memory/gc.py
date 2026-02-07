"""Memory Garbage Collector -- automatic memory lifecycle management.

The ``MemoryGarbageCollector`` provides four categories of automatic
memory management:

1. **Auto-Forget**: remove memories whose relevance score has dropped
   below a configurable threshold.
2. **Summarize**: use an LLM (when available) to progressively compress
   short-term context, falling back to simple truncation otherwise.
3. **Prune**: three pruning strategies -- by relevance ratio, by
   staleness, and by duplicate detection (Jaccard similarity).
4. **Improve**: LLM-powered memory refinement that corrects outdated
   information and extracts higher-level patterns.

Additionally, the collector exposes three lifecycle hooks that the
executor should call at the appropriate moments:

- ``on_turn_end``: clears fleeting memory after each ReAct turn.
- ``on_session_end``: summarizes short-term memory and persists key
  entries to long-term storage.
- ``on_idle``: background maintenance (prune, deduplicate, improve).

All operations are async and the system degrades gracefully when no LLM
provider is available (LLM-powered features are simply skipped).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openagentflow.memory.base import MemoryEntry, MemoryTier


class MemoryGarbageCollector:
    """Automatic memory lifecycle management.

    Args:
        memory_manager: The ``MemoryManager`` whose tiers this collector
            operates on.
        llm_provider: Optional LLM provider for summarization and
            memory improvement.  When ``None``, those features are
            silently skipped.
        relevance_threshold: Entries below this relevance score are
            candidates for auto-forget.
        duplicate_similarity_threshold: Jaccard similarity above which
            two entries are considered duplicates.

    Example::

        gc = MemoryGarbageCollector(manager, llm_provider=my_llm)
        await gc.on_turn_end()
        await gc.on_session_end()
        await gc.on_idle()
    """

    def __init__(
        self,
        memory_manager: Any,
        llm_provider: Any = None,
        relevance_threshold: float = 0.15,
        duplicate_similarity_threshold: float = 0.7,
    ) -> None:
        self.memory = memory_manager
        self.llm = llm_provider
        self.relevance_threshold = relevance_threshold
        self.duplicate_similarity_threshold = duplicate_similarity_threshold

    # ==================================================================
    # Auto-Forget
    # ==================================================================

    async def auto_forget(self, tier: MemoryTier) -> int:
        """Remove memories whose relevance score is below the threshold.

        Args:
            tier: The memory tier to scan.

        Returns:
            Number of entries removed.
        """
        store = self._get_tier(tier)
        if store is None:
            return 0

        entries = self._get_entries(store)
        removed = 0

        for entry in entries:
            if entry.relevance_score < self.relevance_threshold:
                await self._forget_from_tier(store, entry.key)
                removed += 1

        return removed

    # ==================================================================
    # Summarize
    # ==================================================================

    async def summarize_context(
        self,
        entries: list[MemoryEntry],
        max_tokens: int = 2000,
    ) -> str:
        """Produce a progressive summary of the given memory entries.

        When an LLM provider is available the entries are sent for
        intelligent summarization that preserves key facts and decisions.
        Otherwise a simple concatenation with truncation is returned.

        Args:
            entries: Memory entries to summarize.
            max_tokens: Approximate token budget (4 chars ~ 1 token).

        Returns:
            A summary string fitting within the token budget.
        """
        if not entries:
            return ""

        # Build combined text.
        text_parts: list[str] = []
        for entry in entries:
            text_parts.append(f"[{entry.key}] (importance={entry.importance:.2f}) {entry.content}")
        combined = "\n".join(text_parts)

        if self.llm is not None:
            try:
                from openagentflow.core.types import LLMProvider as LP, Message as Msg, ModelConfig

                prompt = (
                    "You are a memory management system. Summarize the following "
                    "memory entries into a concise summary that preserves all key "
                    "facts, decisions, errors, and important context. Remove "
                    f"redundancy. Target length: ~{max_tokens} tokens.\n\n"
                    f"{combined}"
                )
                default_config = ModelConfig(
                    provider=LP.ANTHROPIC,
                    model_id="claude-sonnet-4-20250514",
                )
                response = await self.llm.generate(
                    messages=[Msg(role="user", content=prompt)],
                    config=default_config,
                )
                return response.content
            except Exception:
                pass

        # Fallback: simple truncation.
        max_chars = max_tokens * 4
        if len(combined) <= max_chars:
            return combined
        return combined[:max_chars] + "\n... (truncated)"

    # ==================================================================
    # Prune
    # ==================================================================

    async def prune_by_relevance(
        self,
        tier: MemoryTier,
        keep_ratio: float = 0.7,
    ) -> int:
        """Remove the bottom fraction of entries by relevance score.

        For example, with ``keep_ratio=0.7`` the bottom 30% of entries
        (by relevance) are removed.

        Args:
            tier: The memory tier to prune.
            keep_ratio: Fraction of entries to keep (0.0 - 1.0).

        Returns:
            Number of entries removed.
        """
        store = self._get_tier(tier)
        if store is None:
            return 0

        entries = self._get_entries(store)
        if not entries:
            return 0

        entries.sort(key=lambda e: e.relevance_score)

        remove_count = max(0, int(len(entries) * (1.0 - keep_ratio)))
        removed = 0

        for entry in entries[:remove_count]:
            await self._forget_from_tier(store, entry.key)
            removed += 1

        return removed

    async def prune_by_staleness(
        self,
        tier: MemoryTier,
        max_age_days: int = 30,
    ) -> int:
        """Remove memories not accessed within ``max_age_days``.

        Args:
            tier: The memory tier to prune.
            max_age_days: Maximum number of days since last access.

        Returns:
            Number of entries removed.
        """
        store = self._get_tier(tier)
        if store is None:
            return 0

        now = datetime.now(timezone.utc)
        entries = self._get_entries(store)
        removed = 0

        for entry in entries:
            days_since_access = (now - entry.last_accessed).total_seconds() / 86400
            if days_since_access > max_age_days:
                await self._forget_from_tier(store, entry.key)
                removed += 1

        return removed

    async def prune_duplicates(self, tier: MemoryTier) -> int:
        """Deduplicate similar memories using Jaccard text similarity.

        When two entries exceed the ``duplicate_similarity_threshold`` the
        less relevant one is removed.

        Args:
            tier: The memory tier to deduplicate.

        Returns:
            Number of duplicate entries removed.
        """
        store = self._get_tier(tier)
        if store is None:
            return 0

        entries = self._get_entries(store)
        keys_to_remove: set[str] = set()

        for i in range(len(entries)):
            if entries[i].key in keys_to_remove:
                continue
            for j in range(i + 1, len(entries)):
                if entries[j].key in keys_to_remove:
                    continue

                similarity = _jaccard_similarity(
                    str(entries[i].content),
                    str(entries[j].content),
                )
                if similarity >= self.duplicate_similarity_threshold:
                    # Remove the less relevant one.
                    if entries[i].relevance_score >= entries[j].relevance_score:
                        keys_to_remove.add(entries[j].key)
                    else:
                        keys_to_remove.add(entries[i].key)

        removed = 0
        for key in keys_to_remove:
            await self._forget_from_tier(store, key)
            removed += 1

        return removed

    # ==================================================================
    # Improve
    # ==================================================================

    async def improve_memory(self, tier: MemoryTier) -> int:
        """LLM-powered memory improvement.

        Sends each memory entry to the LLM asking it to:
        - Correct factual errors or outdated information.
        - Extract higher-level patterns or insights.
        - Flag contradictions between entries.

        This operation is a no-op when no LLM provider is configured.

        Args:
            tier: The memory tier to improve.

        Returns:
            Number of entries improved.
        """
        if self.llm is None:
            return 0

        store = self._get_tier(tier)
        if store is None:
            return 0

        entries = self._get_entries(store)
        if not entries:
            return 0

        # Build a context block of all entries for cross-referencing.
        context_parts: list[str] = []
        for entry in entries:
            context_parts.append(
                f"[{entry.key}] importance={entry.importance:.2f}: {entry.content}"
            )
        context_block = "\n".join(context_parts)

        improved = 0

        for entry in entries:
            try:
                from openagentflow.core.types import LLMProvider as LP, Message as Msg, ModelConfig

                prompt = (
                    "You are a memory improvement system. Given the following "
                    "memory entry and the full memory context, improve this "
                    "entry by:\n"
                    "1. Correcting any outdated or incorrect information\n"
                    "2. Making the content more precise and concise\n"
                    "3. Flagging any contradictions with other entries\n\n"
                    f"Full context:\n{context_block}\n\n"
                    f"Entry to improve:\n[{entry.key}] {entry.content}\n\n"
                    "Return ONLY the improved content text. If no improvement "
                    "is needed, return the original text unchanged."
                )
                default_config = ModelConfig(
                    provider=LP.ANTHROPIC,
                    model_id="claude-sonnet-4-20250514",
                )
                response = await self.llm.generate(
                    messages=[Msg(role="user", content=prompt)],
                    config=default_config,
                )
                new_content = response.content.strip()
                if new_content and new_content != str(entry.content):
                    entry.content = new_content
                    entry.last_accessed = datetime.now(timezone.utc)
                    improved += 1
            except Exception:
                # Skip entries that fail improvement.
                continue

        return improved

    # ==================================================================
    # Lifecycle Hooks
    # ==================================================================

    async def on_turn_end(self) -> None:
        """Called after each ReAct turn.

        Clears all fleeting memory so that per-turn scratchpad data does
        not leak into subsequent reasoning steps.
        """
        await self.memory.fleeting.clear()

    async def on_session_end(self) -> None:
        """Called when a session ends.

        Performs two operations:
        1. Summarizes short-term memory (using the LLM if available).
        2. Persists important short-term entries to long-term storage.
        """
        # Step 1: summarize short-term.
        await self.memory.short_term.summarize(llm_provider=self.llm)

        # Step 2: promote important entries to long-term.
        entries = self.memory.short_term.entries
        for entry in entries:
            if entry.importance >= 0.5 or entry.access_count >= 3:
                lt_entry = MemoryEntry(
                    key=entry.key,
                    content=entry.content,
                    tier=MemoryTier.LONG_TERM,
                    created_at=entry.created_at,
                    last_accessed=entry.last_accessed,
                    access_count=entry.access_count,
                    importance=entry.importance,
                    source_agent=entry.source_agent,
                    source_run_id=entry.source_run_id,
                    tags=list(entry.tags),
                )
                await self.memory.long_term.remember(lt_entry)

        # Step 3: clear fleeting (if not already cleared).
        await self.memory.fleeting.clear()

    async def on_idle(self) -> None:
        """Background maintenance during idle periods.

        Runs a comprehensive maintenance cycle on long-term memory:
        1. Auto-forget low-relevance entries.
        2. Prune stale entries (not accessed in 30 days).
        3. Deduplicate similar entries.
        4. Consolidate remaining entries.
        5. Improve memories using the LLM (if available).
        """
        await self.auto_forget(MemoryTier.LONG_TERM)
        await self.prune_by_staleness(MemoryTier.LONG_TERM, max_age_days=30)
        await self.prune_duplicates(MemoryTier.LONG_TERM)
        await self.memory.long_term.consolidate()
        await self.improve_memory(MemoryTier.LONG_TERM)

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _get_tier(self, tier: MemoryTier) -> Any:
        """Return the tier-specific store object.

        Args:
            tier: The tier to resolve.

        Returns:
            The corresponding store, or ``None`` for FLEETING (which has
            no relevance-based management).
        """
        if tier == MemoryTier.FLEETING:
            return self.memory.fleeting
        elif tier == MemoryTier.SHORT_TERM:
            return self.memory.short_term
        elif tier == MemoryTier.LONG_TERM:
            return self.memory.long_term
        return None

    @staticmethod
    def _get_entries(store: Any) -> list[MemoryEntry]:
        """Extract a list of ``MemoryEntry`` objects from a tier store.

        Handles the different internal representations of each tier.

        Args:
            store: A tier store (FleetingMemory, ShortTermMemory, or
                LongTermMemory).

        Returns:
            A list of current entries.
        """
        if hasattr(store, "entries"):
            return list(store.entries)
        if hasattr(store, "_store"):
            # FleetingMemory stores raw values; wrap them.
            from openagentflow.memory.base import MemoryEntry as ME, MemoryTier as MT

            results: list[MemoryEntry] = []
            now = datetime.now(timezone.utc)
            for key, value in store._store.items():
                if isinstance(value, MemoryEntry):
                    results.append(value)
                else:
                    results.append(
                        ME(
                            key=key,
                            content=value,
                            tier=MT.FLEETING,
                            created_at=now,
                            last_accessed=now,
                            importance=0.0,
                        )
                    )
            return results
        return []

    @staticmethod
    async def _forget_from_tier(store: Any, key: str) -> None:
        """Remove an entry from a tier store by key.

        Args:
            store: A tier store.
            key: The key to remove.
        """
        if hasattr(store, "forget"):
            await store.forget(key)


# =====================================================================
# Module-level helpers
# =====================================================================


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two text strings.

    Both strings are lowercased and tokenized on whitespace.  The
    similarity is ``|intersection| / |union|``.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        A float in [0, 1].
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
