"""Regression tests for the multi-tier memory system.

Tests cover:
- FleetingMemory: per-turn scratchpad storage.
- ShortTermMemory: session/conversation context with auto-pruning.
- MemoryEntry: data model serialization and relevance scoring.
- MemoryManager: coordinating tier, routing remember/recall/forget.
- MemoryGarbageCollector: lifecycle hooks and pruning strategies.

All tests are fully self-contained and require no external services.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from openagentflow.memory.base import MemoryEntry, MemoryManager, MemoryTier
from openagentflow.memory.fleeting import FleetingMemory
from openagentflow.memory.gc import MemoryGarbageCollector
from openagentflow.memory.short_term import ShortTermMemory


# =====================================================================
# FleetingMemory tests
# =====================================================================


class TestFleetingMemory:
    """Tests for the FleetingMemory per-turn scratchpad."""

    async def test_fleeting_remember_recall(self):
        """Verify that values stored via remember() are retrievable via recall().

        Stores a simple value and confirms it appears in recall results
        when searching by a matching query string.
        """
        mem = FleetingMemory()
        await mem.remember("scratch", {"partial_result": 42})

        results = await mem.recall("scratch")
        assert len(results) == 1
        assert results[0].key == "scratch"
        assert results[0].content == {"partial_result": 42}
        assert results[0].tier == MemoryTier.FLEETING

    async def test_fleeting_recall_all(self):
        """Verify that an empty query returns all entries."""
        mem = FleetingMemory()
        await mem.remember("key_a", "value_a")
        await mem.remember("key_b", "value_b")

        results = await mem.recall("")
        assert len(results) == 2

    async def test_fleeting_recall_no_match(self):
        """Verify that a non-matching query returns an empty list."""
        mem = FleetingMemory()
        await mem.remember("key_a", "value_a")

        results = await mem.recall("nonexistent")
        assert len(results) == 0

    async def test_fleeting_clear(self):
        """Verify that clear() wipes all entries in the scratchpad.

        After storing entries and calling clear, both the size and recall
        should indicate the store is empty.
        """
        mem = FleetingMemory()
        await mem.remember("a", 1)
        await mem.remember("b", 2)
        assert mem.size == 2

        await mem.clear()
        assert mem.size == 0
        assert len(await mem.recall("")) == 0

    async def test_fleeting_forget(self):
        """Verify that forget() removes a single entry by key."""
        mem = FleetingMemory()
        await mem.remember("keep", "yes")
        await mem.remember("remove", "no")

        await mem.forget("remove")
        assert "remove" not in mem
        assert "keep" in mem

    async def test_fleeting_overwrite(self):
        """Verify that re-remembering the same key overwrites the value."""
        mem = FleetingMemory()
        await mem.remember("x", "old")
        await mem.remember("x", "new")

        results = await mem.recall("x")
        assert len(results) == 1
        assert results[0].content == "new"


# =====================================================================
# ShortTermMemory tests
# =====================================================================


class TestShortTermMemory:
    """Tests for the ShortTermMemory sliding-window session store."""

    def _make_entry(self, key: str, content: str, importance: float = 0.5) -> MemoryEntry:
        """Helper to create a MemoryEntry for testing."""
        now = datetime.now(timezone.utc)
        return MemoryEntry(
            key=key,
            content=content,
            tier=MemoryTier.SHORT_TERM,
            created_at=now,
            last_accessed=now,
            access_count=0,
            importance=importance,
        )

    async def test_short_term_remember_recall(self):
        """Verify that entries stored via remember() are searchable via recall().

        Stores an entry and confirms it is found by a matching key query.
        """
        stm = ShortTermMemory()
        entry = self._make_entry("user_name", "Alice")
        await stm.remember(entry)

        results = await stm.recall("user_name")
        assert len(results) == 1
        assert results[0].key == "user_name"
        assert results[0].content == "Alice"

    async def test_short_term_recall_by_content(self):
        """Verify that recall matches against content, not just key."""
        stm = ShortTermMemory()
        entry = self._make_entry("info", "The capital of France is Paris")
        await stm.remember(entry)

        results = await stm.recall("Paris")
        assert len(results) == 1

    async def test_short_term_recall_by_tag(self):
        """Verify that recall matches against tags."""
        stm = ShortTermMemory()
        entry = self._make_entry("fact", "Water boils at 100C")
        entry.tags = ["science", "chemistry"]
        await stm.remember(entry)

        results = await stm.recall("chemistry")
        assert len(results) == 1

    async def test_short_term_get_recent(self):
        """Verify that get_recent() returns the N most recently inserted entries.

        Inserts five entries and retrieves the last three, checking that
        they appear in newest-first order.
        """
        stm = ShortTermMemory()
        for i in range(5):
            await stm.remember(self._make_entry(f"entry_{i}", f"content_{i}"))

        recent = await stm.get_recent(3)
        assert len(recent) == 3
        # Most recent first.
        assert recent[0].key == "entry_4"
        assert recent[1].key == "entry_3"
        assert recent[2].key == "entry_2"

    async def test_short_term_overflow(self):
        """Verify that auto-pruning kicks in when max_entries is exceeded.

        Creates a ShortTermMemory with max_entries=5, inserts 8 entries,
        and confirms the store does not exceed the limit after the
        automatic refresh.
        """
        stm = ShortTermMemory(max_entries=5, refresh_interval=3)
        for i in range(8):
            await stm.remember(
                self._make_entry(f"item_{i}", f"data_{i}", importance=i * 0.1)
            )

        # After auto-pruning, size should be <= max_entries.
        assert stm.size <= 5

    async def test_short_term_forget(self):
        """Verify that forget() removes an entry by key."""
        stm = ShortTermMemory()
        await stm.remember(self._make_entry("keep", "yes"))
        await stm.remember(self._make_entry("drop", "no"))

        await stm.forget("drop")
        assert "drop" not in stm
        assert "keep" in stm

    async def test_short_term_update_existing(self):
        """Verify that remembering an existing key updates content and metadata."""
        stm = ShortTermMemory()
        original = self._make_entry("key1", "old_content", importance=0.3)
        await stm.remember(original)

        updated = self._make_entry("key1", "new_content", importance=0.7)
        await stm.remember(updated)

        # Should still be one entry.
        assert stm.size == 1
        results = await stm.recall("key1")
        assert results[0].content == "new_content"
        # Importance should be max(old, new).
        assert results[0].importance == 0.7


# =====================================================================
# MemoryEntry tests
# =====================================================================


class TestMemoryEntry:
    """Tests for the MemoryEntry data model."""

    def test_memory_entry_relevance_score(self):
        """Verify that the relevance_score property computes a sensible value.

        A freshly created entry with importance=0.8, access_count=5 should
        produce a score in (0, 1] that reflects those weights.
        """
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            key="test",
            content="test content",
            tier=MemoryTier.SHORT_TERM,
            created_at=now,
            last_accessed=now,
            access_count=5,
            importance=0.8,
        )

        score = entry.relevance_score
        # Score is composed of:
        #   recency (0.3 * 1/(1+0)) = 0.30
        #   frequency (0.2 * min(5/10, 1.0)) = 0.10
        #   importance (0.4 * 0.8) = 0.32
        #   decay (0.1 * 0.95^0) = 0.10
        # Total ~0.82
        assert 0.0 < score <= 1.0
        assert score == pytest.approx(0.82, abs=0.05)

    def test_memory_entry_relevance_score_old_entry(self):
        """Verify that relevance decays for entries not accessed recently."""
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        entry = MemoryEntry(
            key="old",
            content="old content",
            tier=MemoryTier.LONG_TERM,
            created_at=old_time,
            last_accessed=old_time,
            access_count=1,
            importance=0.5,
        )

        score = entry.relevance_score
        # Old entries should have lower relevance.
        assert score < 0.5

    def test_memory_entry_serialization(self):
        """Verify that to_json/from_json produce a faithful roundtrip.

        Serializes a MemoryEntry to JSON and reconstructs it, checking
        that all fields survive the roundtrip unchanged.
        """
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            key="round_trip",
            content={"nested": "data", "count": 42},
            tier=MemoryTier.SHORT_TERM,
            created_at=now,
            last_accessed=now,
            access_count=3,
            importance=0.75,
            source_agent="test_agent",
            source_run_id="run-xyz",
            tags=["tag1", "tag2"],
            superseded_by=None,
        )

        json_str = entry.to_json()
        restored = MemoryEntry.from_json(json_str)

        assert restored.key == entry.key
        assert restored.content == entry.content
        assert restored.tier == entry.tier
        assert restored.access_count == entry.access_count
        assert restored.importance == entry.importance
        assert restored.source_agent == entry.source_agent
        assert restored.source_run_id == entry.source_run_id
        assert restored.tags == entry.tags
        assert restored.superseded_by is None

    def test_memory_entry_to_dict_from_dict(self):
        """Verify that to_dict/from_dict produce a faithful roundtrip."""
        entry = MemoryEntry(
            key="dict_test",
            content="simple text",
            tier=MemoryTier.FLEETING,
        )
        data = entry.to_dict()
        assert isinstance(data, dict)
        assert data["key"] == "dict_test"
        assert data["tier"] == "fleeting"

        restored = MemoryEntry.from_dict(data)
        assert restored.key == entry.key
        assert restored.tier == entry.tier


# =====================================================================
# MemoryManager tests
# =====================================================================


class TestMemoryManager:
    """Tests for the MemoryManager coordinator."""

    async def test_memory_manager_remember_recall(self):
        """Verify the full remember-recall flow through the manager.

        Stores a memory in SHORT_TERM via the manager and retrieves it
        using a matching query.
        """
        manager = MemoryManager()
        await manager.remember(
            "user_preference",
            "dark mode",
            tier=MemoryTier.SHORT_TERM,
            importance=0.6,
        )

        results = await manager.recall("user_preference")
        assert len(results) >= 1
        found = [r for r in results if r.key == "user_preference"]
        assert len(found) == 1
        assert found[0].content == "dark mode"

    async def test_memory_manager_remember_fleeting(self):
        """Verify that the manager routes FLEETING memories to FleetingMemory."""
        manager = MemoryManager()
        await manager.remember(
            "scratch_data",
            {"intermediate": True},
            tier=MemoryTier.FLEETING,
        )

        results = await manager.recall(
            "scratch_data",
            tiers=[MemoryTier.FLEETING],
        )
        assert len(results) == 1
        assert results[0].content == {"intermediate": True}

    async def test_memory_manager_forget(self):
        """Verify that explicit forget() removes the memory.

        After storing and then forgetting a key, recall should return
        no matching entries.
        """
        manager = MemoryManager()
        await manager.remember("temp", "temporary", tier=MemoryTier.SHORT_TERM)

        await manager.forget("temp")
        results = await manager.recall(
            "temp", tiers=[MemoryTier.SHORT_TERM]
        )
        # Should have been removed.
        found = [r for r in results if r.key == "temp"]
        assert len(found) == 0

    async def test_memory_manager_forget_specific_tier(self):
        """Verify that forget with a specific tier only removes from that tier."""
        manager = MemoryManager()
        await manager.remember("multi", "data", tier=MemoryTier.SHORT_TERM)
        await manager.remember("multi", "data", tier=MemoryTier.FLEETING)

        await manager.forget("multi", tier=MemoryTier.FLEETING)

        # Should still be in short-term.
        results = await manager.recall("multi", tiers=[MemoryTier.SHORT_TERM])
        found = [r for r in results if r.key == "multi"]
        assert len(found) == 1

    async def test_memory_manager_get_context_window(self):
        """Verify that get_context_window returns entries within token budget."""
        manager = MemoryManager()
        await manager.remember("item1", "A" * 100, tier=MemoryTier.SHORT_TERM)
        await manager.remember("item2", "B" * 100, tier=MemoryTier.SHORT_TERM)

        # Budget of 200 tokens ~ 800 chars, should fit both entries.
        context = await manager.get_context_window(max_tokens=200)
        assert len(context) >= 1

    async def test_memory_manager_run_gc(self):
        """Verify that run_gc completes without error when no LLM is available.

        The garbage collector should degrade gracefully with no LLM --
        LLM-powered features are simply skipped.
        """
        manager = MemoryManager()
        await manager.remember("gc_test", "value", tier=MemoryTier.SHORT_TERM)
        # Should not raise.
        await manager.run_gc()


# =====================================================================
# MemoryGarbageCollector tests
# =====================================================================


class TestMemoryGarbageCollector:
    """Tests for the MemoryGarbageCollector lifecycle hooks and strategies."""

    async def test_gc_on_turn_end(self):
        """Verify that on_turn_end clears all fleeting memory.

        After storing fleeting entries and triggering on_turn_end, the
        fleeting store should be empty.
        """
        manager = MemoryManager()
        await manager.remember("scratch", "temp_value", tier=MemoryTier.FLEETING)
        assert manager.fleeting.size == 1

        await manager.gc.on_turn_end()
        assert manager.fleeting.size == 0

    async def test_gc_on_turn_end_preserves_short_term(self):
        """Verify that on_turn_end does NOT clear short-term memory."""
        manager = MemoryManager()
        await manager.remember("st_data", "persistent", tier=MemoryTier.SHORT_TERM)
        await manager.remember("fl_data", "ephemeral", tier=MemoryTier.FLEETING)

        await manager.gc.on_turn_end()

        # Short-term should survive.
        assert manager.short_term.size == 1
        # Fleeting should be cleared.
        assert manager.fleeting.size == 0

    async def test_gc_prune_by_relevance(self):
        """Verify that prune_by_relevance removes the bottom fraction.

        Creates entries with varying importance and verifies that with
        keep_ratio=0.5, roughly half the entries are removed.
        """
        manager = MemoryManager()

        # Create entries with different importance levels.
        for i in range(10):
            await manager.remember(
                f"entry_{i}",
                f"content_{i}",
                tier=MemoryTier.SHORT_TERM,
                importance=i * 0.1,
            )

        initial_size = manager.short_term.size
        removed = await manager.gc.prune_by_relevance(
            MemoryTier.SHORT_TERM, keep_ratio=0.5
        )

        assert removed > 0
        assert manager.short_term.size < initial_size

    async def test_gc_prune_by_staleness(self):
        """Verify that prune_by_staleness removes entries older than max_age_days.

        Creates entries with artificially old last_accessed timestamps and
        confirms they are removed while recent entries survive.
        """
        manager = MemoryManager()
        now = datetime.now(timezone.utc)

        # Fresh entry.
        fresh_entry = MemoryEntry(
            key="fresh",
            content="I am fresh",
            tier=MemoryTier.SHORT_TERM,
            created_at=now,
            last_accessed=now,
            importance=0.5,
        )
        await manager.short_term.remember(fresh_entry)

        # Stale entry (60 days old).
        stale_entry = MemoryEntry(
            key="stale",
            content="I am stale",
            tier=MemoryTier.SHORT_TERM,
            created_at=now - timedelta(days=60),
            last_accessed=now - timedelta(days=60),
            importance=0.5,
        )
        await manager.short_term.remember(stale_entry)

        removed = await manager.gc.prune_by_staleness(
            MemoryTier.SHORT_TERM, max_age_days=30
        )

        assert removed == 1
        assert "stale" not in manager.short_term
        assert "fresh" in manager.short_term

    async def test_gc_auto_forget(self):
        """Verify that auto_forget removes entries below the relevance threshold.

        Creates an entry with zero importance and very old timestamps so
        its relevance score is extremely low, then confirms auto_forget
        removes it.
        """
        manager = MemoryManager()
        old_time = datetime.now(timezone.utc) - timedelta(days=365)

        low_relevance = MemoryEntry(
            key="irrelevant",
            content="outdated fact",
            tier=MemoryTier.SHORT_TERM,
            created_at=old_time,
            last_accessed=old_time,
            access_count=0,
            importance=0.0,
        )
        await manager.short_term.remember(low_relevance)

        gc = MemoryGarbageCollector(manager, relevance_threshold=0.5)
        removed = await gc.auto_forget(MemoryTier.SHORT_TERM)

        assert removed >= 1
        assert "irrelevant" not in manager.short_term

    async def test_gc_prune_duplicates(self):
        """Verify that prune_duplicates removes near-duplicate entries.

        Creates two entries with almost identical content and confirms
        the less relevant one is removed.
        """
        manager = MemoryManager()
        now = datetime.now(timezone.utc)

        e1 = MemoryEntry(
            key="dup1",
            content="the quick brown fox jumps over the lazy dog",
            tier=MemoryTier.SHORT_TERM,
            created_at=now,
            last_accessed=now,
            importance=0.8,
        )
        e2 = MemoryEntry(
            key="dup2",
            content="the quick brown fox jumps over the lazy dog",
            tier=MemoryTier.SHORT_TERM,
            created_at=now,
            last_accessed=now,
            importance=0.3,
        )
        await manager.short_term.remember(e1)
        await manager.short_term.remember(e2)

        gc = MemoryGarbageCollector(
            manager, duplicate_similarity_threshold=0.7
        )
        removed = await gc.prune_duplicates(MemoryTier.SHORT_TERM)

        assert removed == 1
        # The more relevant entry (dup1) should survive.
        assert "dup1" in manager.short_term
        assert "dup2" not in manager.short_term

    async def test_gc_improve_memory_no_llm(self):
        """Verify that improve_memory is a no-op when no LLM is available."""
        manager = MemoryManager()
        await manager.remember("fact", "Earth is round", tier=MemoryTier.SHORT_TERM)

        gc = MemoryGarbageCollector(manager, llm_provider=None)
        improved = await gc.improve_memory(MemoryTier.SHORT_TERM)
        assert improved == 0

    async def test_gc_on_session_end(self):
        """Verify that on_session_end clears fleeting and summarizes short-term.

        After calling on_session_end, fleeting memory should be empty and
        the operation should complete without error.
        """
        manager = MemoryManager()
        await manager.remember("fl", "temp", tier=MemoryTier.FLEETING)
        await manager.remember("st", "session_data", tier=MemoryTier.SHORT_TERM)

        await manager.gc.on_session_end()
        assert manager.fleeting.size == 0
