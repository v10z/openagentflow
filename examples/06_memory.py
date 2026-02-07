"""
OpenAgentFlow Example 06: Memory System

Demonstrates:
- 3-tier memory (fleeting, short-term, long-term)
- Storing and recalling memories
- Relevance scoring
- Context window building
- Garbage collection
"""
import asyncio
from openagentflow.memory import MemoryManager, MemoryTier, MemoryEntry


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 06: Memory System")
    print("=" * 60)

    # Create a memory manager (works without any dependencies)
    memory = MemoryManager()

    # --- Fleeting Memory (per-turn scratchpad) ---
    print("\n--- Tier 1: Fleeting Memory ---")
    await memory.remember("scratch_calc", {"step": 1, "value": 42}, tier=MemoryTier.FLEETING)
    await memory.remember("temp_note", "intermediate result", tier=MemoryTier.FLEETING)

    results = await memory.recall("scratch")
    print(f"Recalled 'scratch': {len(results)} entries")
    for r in results:
        print(f"  [{r.tier.value}] {r.key}: {r.content}")

    # Clear fleeting (simulates end of turn)
    await memory.gc.on_turn_end()
    results = await memory.recall("scratch")
    print(f"After turn end: {len(results)} entries (fleeting cleared)")

    # --- Short-Term Memory (session context) ---
    print("\n--- Tier 2: Short-Term Memory ---")
    await memory.remember(
        "user_preference", "prefers concise answers",
        tier=MemoryTier.SHORT_TERM, importance=0.8
    )
    await memory.remember(
        "project_context", "building a REST API with FastAPI",
        tier=MemoryTier.SHORT_TERM, importance=0.9
    )
    await memory.remember(
        "minor_note", "user mentioned coffee",
        tier=MemoryTier.SHORT_TERM, importance=0.2
    )

    results = await memory.recall("project")
    print(f"Recalled 'project': {len(results)} entries")
    for r in results:
        print(f"  [{r.tier.value}] {r.key}: {r.content} (relevance={r.relevance_score:.2f})")

    # --- Long-Term Memory (persistent) ---
    print("\n--- Tier 3: Long-Term Memory ---")
    await memory.remember(
        "architecture_decision", "Using event-driven microservices with Kafka",
        tier=MemoryTier.LONG_TERM, importance=0.95,
        tags=["architecture", "decision"],
    )
    await memory.remember(
        "debugging_lesson", "Always check for None before calling .split()",
        tier=MemoryTier.LONG_TERM, importance=0.7,
        tags=["debugging", "python"],
    )

    results = await memory.recall("architecture")
    print(f"Recalled 'architecture': {len(results)} entries")
    for r in results:
        print(f"  [{r.tier.value}] {r.key}: {r.content} (relevance={r.relevance_score:.2f})")

    # --- Context Window ---
    print("\n--- Context Window ---")
    context = await memory.get_context_window(max_tokens=4000)
    print(f"Context entries: {len(context)} (sorted by relevance)")
    for entry in context:
        print(f"  {entry.key}: relevance={entry.relevance_score:.2f}, importance={entry.importance}")

    # --- Garbage Collection ---
    print("\n--- Garbage Collection ---")
    print("Running GC cycle...")
    await memory.run_gc()
    print("GC complete. Low-relevance entries pruned.")

    # Check what survived
    all_results = await memory.recall("")
    print(f"Surviving entries: {len(all_results)}")
    for r in all_results:
        print(f"  [{r.tier.value}] {r.key} (relevance={r.relevance_score:.2f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
