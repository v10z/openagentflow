"""Multi-layer memory hierarchy for OpenAgentFlow.

This package implements a three-tier memory system with automatic lifecycle
management:

Tiers:
    - **FleetingMemory**: per-turn scratchpad, automatically cleared after
      each ReAct iteration.
    - **ShortTermMemory**: session/conversation context with a sliding
      window and progressive summarization.
    - **LongTermMemory**: persistent storage across sessions with
      relevance-based decay and pruning.

Automatic Management:
    The ``MemoryGarbageCollector`` provides four auto-management
    capabilities:

    - **Auto-Forget**: remove entries below a relevance threshold.
    - **Auto-Summarize**: LLM-powered progressive compression of context
      (falls back to truncation when no LLM is available).
    - **Auto-Prune**: remove stale, duplicate, and low-relevance entries.
    - **Auto-Improve**: LLM-powered correction and refinement of stored
      memories.

Usage::

    from openagentflow.memory import MemoryManager, MemoryTier

    manager = MemoryManager()
    await manager.remember("user_name", "Alice", tier=MemoryTier.SHORT_TERM)
    results = await manager.recall("user_name")
    await manager.run_gc()
"""

from openagentflow.memory.base import MemoryEntry, MemoryManager, MemoryTier
from openagentflow.memory.fleeting import FleetingMemory
from openagentflow.memory.gc import MemoryGarbageCollector
from openagentflow.memory.long_term import LongTermMemory
from openagentflow.memory.short_term import ShortTermMemory

__all__ = [
    "MemoryManager",
    "MemoryTier",
    "MemoryEntry",
    "FleetingMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryGarbageCollector",
]
