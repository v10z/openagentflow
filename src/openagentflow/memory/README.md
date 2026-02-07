# OpenAgentFlow Memory System

## Overview

The OpenAgentFlow memory system implements a **three-tier memory hierarchy**
with automatic lifecycle management. It is inspired by human cognitive
architecture: a fleeting working memory for immediate reasoning, a short-term
buffer for session context, and a persistent long-term store that survives
across sessions.

All tiers are coordinated by a single `MemoryManager` and maintained by an
automatic `MemoryGarbageCollector` that handles forgetting, summarization,
pruning, deduplication, and LLM-powered memory improvement.

The system is designed to degrade gracefully:

- **Without an LLM provider**: summarization and memory improvement are
  skipped; simple truncation is used instead.
- **Without a graph backend**: long-term memory falls back to an in-memory
  dictionary, which is sufficient for tests and short-lived processes.
- **Without any configuration**: all three tiers work out of the box with
  sensible defaults.

---

## Architecture

```
+----------------------------------------------------------------------+
|                         MemoryManager                                |
|                                                                      |
|   remember(key, value, tier, importance)                             |
|   recall(query, tiers) -> list[MemoryEntry]                         |
|   forget(key, tier)                                                  |
|   get_context_window(max_tokens) -> list[MemoryEntry]               |
|   run_gc()                                                           |
|                                                                      |
+----------------+-----------------+-----------------------------------+
|   FLEETING     |   SHORT-TERM    |          LONG-TERM                |
|   (per-turn)   |   (session)     |        (persistent)               |
|                |                 |                                    |
|  dict store    | sliding window  |  graph backend                    |
|  key -> value  | max 100 entries |  (or in-memory fallback)          |
|  auto-clear    | auto-summary    |  auto-decay (0.95^days)           |
|  on turn end   | auto-prune      |  consolidation                    |
|                | refresh/10 ops  |  agent/run recall                  |
|                |                 |                                    |
|  FleetingMemory| ShortTermMemory | LongTermMemory                   |
|  (fleeting.py) | (short_term.py) | (long_term.py)                   |
+----------------+-----------------+-----------------------------------+
|                                                                      |
|                    MemoryGarbageCollector (gc.py)                     |
|                                                                      |
|   auto_forget    | Remove entries below relevance threshold          |
|   summarize      | LLM-powered progressive compression              |
|   prune_relevance| Drop bottom N% by relevance score                |
|   prune_staleness| Drop entries not accessed in N days              |
|   prune_dupes    | Jaccard similarity deduplication                 |
|   improve_memory | LLM-powered correction and refinement            |
|                                                                      |
|   Lifecycle Hooks:                                                   |
|     on_turn_end()    -> clear fleeting                               |
|     on_session_end() -> summarize short-term, promote to long-term   |
|     on_idle()        -> full maintenance cycle on long-term          |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## Tier Details

### Tier 1: Fleeting Memory

**Purpose**: Per-turn scratchpad for intermediate reasoning artifacts.

Think of this as the whiteboard an agent scribbles on while working through
a single ReAct step. Once the step is complete, the whiteboard is wiped clean.

| Property           | Value                                  |
|--------------------|----------------------------------------|
| Backing store      | `dict[str, Any]`                       |
| Persistence        | None (cleared every turn)              |
| Auto-management    | `clear()` called by `gc.on_turn_end()` |
| Relevance scoring  | Not applicable (importance = 0.0)      |
| Max lifetime       | One ReAct iteration                    |

```python
# FleetingMemory API
mem = FleetingMemory()
await mem.remember("scratch", {"partial_result": 42})
results = await mem.recall("scratch")     # substring search on keys + values
await mem.forget("scratch")               # remove one entry
await mem.clear()                         # wipe everything
print(mem.size)                           # 0
```

**Implementation**: `openagentflow/memory/fleeting.py`

### Tier 2: Short-Term Memory

**Purpose**: Session/conversation context with automatic compression.

Retains information for the duration of a session. When the entry count
exceeds `max_entries` (default: 100) or every `refresh_interval` operations
(default: 10), a refresh cycle prunes low-relevance entries. When a session
ends, older entries are progressively summarized (using the LLM if available)
and important entries are promoted to long-term storage.

| Property           | Value                                              |
|--------------------|----------------------------------------------------|
| Backing store      | `dict[str, MemoryEntry]` + insertion order list    |
| Persistence        | Session duration                                   |
| Max entries        | 100 (configurable)                                 |
| Auto-refresh       | Every 10 `remember()` calls or on overflow         |
| Relevance threshold| 0.15 (entries below this are pruned)               |
| Summarization      | LLM-powered progressive compression (or truncation)|

```python
# ShortTermMemory API
stm = ShortTermMemory(max_entries=50, relevance_threshold=0.15, refresh_interval=10)
await stm.remember(entry)                # MemoryEntry object
results = await stm.recall("user prefs") # substring search, sorted by relevance
recent = await stm.get_recent(5)         # 5 most recently inserted
await stm.forget("old_key")
summary = await stm.summarize(llm_provider=provider)  # compress old entries
print(stm.size)
```

**Key behaviors**:

- **Auto-refresh**: On every 10th insertion (or when over capacity), entries
  with `relevance_score < 0.15` are pruned. If still over capacity, the
  least relevant entries are dropped until size is within bounds.
- **Progressive summarization**: When `summarize()` is called, the oldest
  entries (beyond the 10 most recent) are sent to the LLM for intelligent
  compression. The summary replaces the compressed entries as a single
  `__summary__` entry with importance 0.8.
- **Recall updates metadata**: Every successful recall updates
  `last_accessed` and increments `access_count`, which feeds back into
  the relevance score.

**Implementation**: `openagentflow/memory/short_term.py`

### Tier 3: Long-Term Memory

**Purpose**: Persistent storage across sessions with relevance-based decay.

Uses a graph backend (implementing the `GraphBackend` protocol) to persist
memories as vertices. Supports keyword search, agent-based retrieval,
run-based retrieval, automatic consolidation, and time-based decay.

| Property           | Value                                              |
|--------------------|----------------------------------------------------|
| Backing store      | Graph backend (or in-memory dict fallback)         |
| Persistence        | Across sessions (with real graph backend)          |
| Relevance threshold| 0.10 (entries below this are pruned during decay)  |
| Decay function     | `relevance * 0.95 ^ days_since_last_access`        |
| Consolidation      | Jaccard similarity >= 0.6 triggers merge           |
| Graph label        | `"memory"` (vertices)                              |

```python
# LongTermMemory API
ltm = LongTermMemory(graph_backend=backend, relevance_threshold=0.10)
await ltm.remember(entry)                    # persist to graph (or dict)
results = await ltm.recall("user preferences")  # keyword search
results = await ltm.recall_by_agent("researcher")  # filter by source agent
results = await ltm.recall_by_run("run-abc123")    # filter by run ID
removed = await ltm.consolidate()            # merge similar memories
removed = await ltm.decay()                  # prune decayed entries
print(ltm.size)
```

**Key behaviors**:

- **Graph persistence**: Entries are serialized via `MemoryEntry.to_dict()`
  and stored as graph vertices with `label="memory"`. On graph errors, the
  system silently falls back to in-memory storage.
- **Consolidation**: Pairs of entries with Jaccard similarity >= 0.6 are
  merged. The more relevant entry absorbs the access count, importance, and
  tags of the less relevant one; the loser is marked `superseded_by` and
  removed.
- **Decay**: `relevance * 0.95 ^ days_since_last_access`. Entries whose
  decayed relevance drops below `relevance_threshold` (0.10) are pruned.

**Implementation**: `openagentflow/memory/long_term.py`

---

## The MemoryEntry

Every piece of stored information is wrapped in a `MemoryEntry` dataclass:

```python
@dataclass
class MemoryEntry:
    key: str                    # Unique identifier
    content: Any                # The stored payload (JSON-serializable for long-term)
    tier: MemoryTier            # FLEETING | SHORT_TERM | LONG_TERM
    created_at: datetime        # UTC timestamp of creation
    last_accessed: datetime     # UTC timestamp of most recent read
    access_count: int           # Number of retrievals
    importance: float           # Manual weight in [0, 1]
    source_agent: str           # Which agent created this memory
    source_run_id: str          # Which execution run produced it
    tags: list[str]             # Freeform labels for categorical retrieval
    superseded_by: str | None   # Key of newer entry that replaces this one
```

Serialization is supported via `to_dict()`, `from_dict()`, `to_json()`, and
`from_json()` for graph storage and wire transfer.

---

## Relevance Scoring

Every `MemoryEntry` computes a dynamic `relevance_score` property that blends
four signals:

```
relevance = recency * 0.3 + frequency * 0.2 + importance * 0.4 + decay * 0.1
```

### Component Breakdown

| Component    | Weight | Formula                                  | Intuition                              |
|-------------|--------|------------------------------------------|----------------------------------------|
| **Recency**  | 0.30   | `1.0 / (1.0 + days_since_last_access)`   | Recently used memories score higher    |
| **Frequency**| 0.20   | `min(access_count / 10.0, 1.0)`          | Frequently accessed memories matter    |
| **Importance**| 0.40  | `self.importance` (manually set)         | Critical facts always rank high        |
| **Decay**    | 0.10   | `0.95 ^ days_since_creation`             | Old memories gradually fade            |

### Score Examples

| Scenario                                    | Recency | Freq | Import | Decay | Total |
|---------------------------------------------|---------|------|--------|-------|-------|
| Just created, importance=0.8                | 0.30    | 0.00 | 0.32   | 0.10  | 0.72  |
| 1 day old, accessed 5 times, importance=0.5 | 0.15    | 0.10 | 0.20   | 0.10  | 0.55  |
| 7 days old, accessed 10 times, importance=1 | 0.04    | 0.20 | 0.40   | 0.07  | 0.71  |
| 30 days old, accessed once, importance=0.2  | 0.01    | 0.02 | 0.08   | 0.02  | 0.13  |
| 60 days old, never accessed, importance=0.1 | 0.005   | 0.00 | 0.04   | 0.005 | 0.05  |

Entries with a relevance score below the threshold (0.15 for short-term,
0.10 for long-term after decay) are candidates for automatic pruning.

---

## Context Refresh Timeline

The memory system evolves as an agent processes turns. Here is how the
three tiers interact across a typical session:

```
Turn 1
  |-- Fleeting: Agent stores scratch notes ("tool output", "intermediate calc")
  |-- Short-term: First entry stored (e.g., "user asked about X")
  |-- Long-term: (unchanged, loaded from previous sessions)
  |-- GC: on_turn_end() -> fleeting.clear()
  |
Turn 5
  |-- Fleeting: Fresh scratchpad for this turn
  |-- Short-term: 5 entries, no refresh yet
  |-- Long-term: (unchanged)
  |-- GC: on_turn_end() -> fleeting.clear()
  |
Turn 10
  |-- Fleeting: Fresh scratchpad
  |-- Short-term: ~10 entries -> auto-refresh triggered (prune below 0.15)
  |-- Long-term: (unchanged)
  |-- GC: on_turn_end() -> fleeting.clear()
  |
Turn 20
  |-- Fleeting: Fresh scratchpad
  |-- Short-term: ~15-18 entries (some pruned at turns 10, 20)
  |-- Long-term: (unchanged)
  |-- GC: on_turn_end() -> fleeting.clear()
  |
Turn 50
  |-- Fleeting: Fresh scratchpad
  |-- Short-term: ~20-30 entries (heavily pruned, possibly summarized)
  |             summary entry captures compressed older context
  |-- Long-term: (unchanged)
  |-- GC: on_turn_end() -> fleeting.clear()
  |
Session End
  |-- GC: on_session_end() triggers:
  |     1. short_term.summarize(llm_provider) -> compress old entries
  |     2. Promote important entries (importance >= 0.5 OR access_count >= 3)
  |        from short-term to long-term
  |     3. fleeting.clear()
  |
Idle Period (between sessions)
  |-- GC: on_idle() triggers full long-term maintenance:
  |     1. auto_forget: remove entries below relevance threshold
  |     2. prune_by_staleness: remove entries not accessed in 30 days
  |     3. prune_duplicates: Jaccard dedup (similarity >= 0.7)
  |     4. consolidate: merge similar entries (similarity >= 0.6)
  |     5. improve_memory: LLM reviews and corrects stored memories
  |
Next Session
  |-- Fleeting: Empty
  |-- Short-term: Empty (fresh session)
  |-- Long-term: Cleaned, consolidated, improved memories from prior sessions
```

---

## Usage Examples

### Basic Usage

```python
from openagentflow.memory import MemoryManager, MemoryTier

# Create manager (no dependencies required)
memory = MemoryManager()

# Store memories in different tiers
await memory.remember("scratch_calc", 42, tier=MemoryTier.FLEETING)
await memory.remember("user_name", "Alice", tier=MemoryTier.SHORT_TERM, importance=0.7)
await memory.remember(
    "api_key_location",
    "stored in .env file under PROJECT_ROOT",
    tier=MemoryTier.LONG_TERM,
    importance=0.8,
    tags=["config", "security"],
)

# Recall across all tiers
results = await memory.recall("api key")
for entry in results:
    print(f"  [{entry.tier.value}] {entry.key}: {entry.content} (relevance={entry.relevance_score:.2f})")

# Forget explicitly
await memory.forget("scratch_calc")

# Run garbage collection
await memory.run_gc()
```

### With LLM Provider and Graph Backend

```python
from openagentflow.memory import MemoryManager, MemoryTier
from openagentflow.graph import SQLiteGraphBackend
from openagentflow.llm.providers.anthropic_ import AnthropicProvider

# Full-featured setup
backend = SQLiteGraphBackend(":memory:")
provider = AnthropicProvider(api_key="sk-ant-...")
memory = MemoryManager(graph_backend=backend, llm_provider=provider)

# Long-term memories are persisted to the graph
await memory.remember(
    "project_architecture",
    "Microservices with event-driven communication via Kafka",
    tier=MemoryTier.LONG_TERM,
    importance=0.9,
    tags=["architecture", "decision"],
    source_agent="architect",
)

# GC now uses the LLM for summarization and improvement
await memory.run_gc()
```

### Building a Context Window

```python
# Get the most relevant memories that fit within a token budget
context = await memory.get_context_window(max_tokens=4000)

# The context is sorted by relevance and fits within ~4000 tokens
# (approximated as 4 characters per token)
for entry in context:
    print(f"  {entry.key}: {entry.content[:60]}... (relevance={entry.relevance_score:.2f})")
```

### Querying by Agent or Run

```python
# Retrieve all memories from a specific agent
agent_memories = await memory.long_term.recall_by_agent("researcher")

# Retrieve all memories from a specific execution run
run_memories = await memory.long_term.recall_by_run("run-abc-123")
```

---

## Garbage Collection

The `MemoryGarbageCollector` provides four categories of automatic memory
management and three lifecycle hooks.

### Management Categories

#### 1. Auto-Forget

Removes entries whose `relevance_score` has dropped below
`relevance_threshold` (default: 0.15).

```python
removed = await gc.auto_forget(MemoryTier.SHORT_TERM)
print(f"Removed {removed} low-relevance entries")
```

#### 2. Summarize

Uses the LLM to progressively compress context. Falls back to simple
concatenation + truncation when no LLM is available.

```python
summary = await gc.summarize_context(entries, max_tokens=2000)
# With LLM: intelligent summary preserving key facts
# Without:  concatenated text, truncated to ~8000 chars
```

The LLM prompt instructs: "Summarize the following memory entries into a
concise summary that preserves all key facts, decisions, errors, and
important context. Remove redundancy."

#### 3. Prune

Three pruning strategies:

```python
# Drop bottom 30% by relevance score
await gc.prune_by_relevance(MemoryTier.LONG_TERM, keep_ratio=0.7)

# Drop entries not accessed in 30 days
await gc.prune_by_staleness(MemoryTier.LONG_TERM, max_age_days=30)

# Deduplicate: Jaccard similarity >= 0.7 triggers removal of less relevant entry
await gc.prune_duplicates(MemoryTier.SHORT_TERM)
```

The Jaccard similarity function tokenizes both texts (lowercase, whitespace
split) and computes `|intersection| / |union|`.

#### 4. Improve

LLM-powered memory refinement. For each entry, the LLM is asked to:
- Correct factual errors or outdated information
- Make content more precise and concise
- Flag contradictions with other entries

This is a no-op when no LLM provider is configured.

```python
improved = await gc.improve_memory(MemoryTier.LONG_TERM)
print(f"Improved {improved} entries")
```

### Lifecycle Hooks

The executor should call these hooks at the appropriate moments:

```python
# After each ReAct turn:
await gc.on_turn_end()
# -> Clears all fleeting memory

# When a session ends:
await gc.on_session_end()
# -> Summarizes short-term memory (LLM if available)
# -> Promotes important entries to long-term (importance >= 0.5 OR access_count >= 3)
# -> Clears fleeting memory

# During idle periods (between sessions):
await gc.on_idle()
# -> auto_forget on long-term
# -> prune_by_staleness (30 days)
# -> prune_duplicates on long-term
# -> consolidate similar entries
# -> improve_memory via LLM
```

### Lifecycle Hook Flow

```
on_turn_end()
    |
    +-> fleeting.clear()
    |
    Done.

on_session_end()
    |
    +-> short_term.summarize(llm_provider)
    |       |
    |       +-> [LLM available?]
    |       |       YES: Send entries to LLM for intelligent summary
    |       |       NO:  Concatenate + truncate to 2000 chars
    |       |
    |       +-> Replace old entries with __summary__ entry (importance=0.8)
    |
    +-> For each short-term entry where importance >= 0.5 OR access_count >= 3:
    |       +-> Copy to long-term with all metadata preserved
    |
    +-> fleeting.clear()
    |
    Done.

on_idle()
    |
    +-> auto_forget(LONG_TERM)          -- relevance < 0.15
    +-> prune_by_staleness(LONG_TERM)   -- not accessed in 30 days
    +-> prune_duplicates(LONG_TERM)     -- Jaccard >= 0.7
    +-> long_term.consolidate()         -- Jaccard >= 0.6, merge metadata
    +-> improve_memory(LONG_TERM)       -- LLM refinement (if available)
    |
    Done.
```

---

## Integration with Agents

### The @agent Decorator

The `@agent` decorator accepts a `memory` parameter of type `MemoryConfig`:

```python
from openagentflow import agent
from openagentflow.core.types import MemoryConfig

@agent(
    model="claude-sonnet-4-20250514",
    memory=MemoryConfig(
        short_term_enabled=True,
        short_term_strategy="summarization",
        short_term_max_tokens=8000,
        long_term_enabled=True,
        long_term_backend="sqlite",
        working_enabled=True,
    ),
)
async def researcher(query: str) -> str:
    """Research agent with full memory."""
    ...
```

### How Memory Plugs into the Executor

```
User calls: result = await researcher("What is quantum computing?")
    |
    v
@agent decorator -> AgentSpec (with MemoryConfig)
    |
    v
AgentExecutor.run(spec, input_data)
    |
    v
_react_loop():
    |
    +-- Turn 1:
    |     |-- memory.fleeting: store tool outputs, scratch notes
    |     |-- memory.short_term: store user query, key observations
    |     |-- LLM call with context from memory.get_context_window()
    |     |-- gc.on_turn_end() -> clear fleeting
    |
    +-- Turn 2:
    |     |-- memory.fleeting: fresh scratchpad
    |     |-- memory.short_term: accumulating context
    |     |-- LLM call with updated context window
    |     |-- gc.on_turn_end() -> clear fleeting
    |
    +-- ... (repeat until done)
    |
    +-- Final output produced
    |
    v
gc.on_session_end()
    |-- summarize short-term
    |-- promote important entries to long-term
    |-- clear fleeting
    |
    v
AgentResult returned to user
```

### Memory in Multi-Agent Systems (Swarms/Chains)

When agents are composed into chains or swarms, memory provides continuity:

```
Agent A (researcher)                Agent B (writer)
    |                                   |
    |-- memory.remember(                |-- results = memory.recall(
    |     "finding_1",                  |     "finding",
    |     "Quantum uses qubits",        |     tiers=[LONG_TERM]
    |     tier=LONG_TERM,               |   )
    |     importance=0.9,               |
    |     source_agent="researcher"     |-- Uses findings to write report
    |   )                               |
    |                                   |
    +-- shared graph backend <----------+
```

Agents sharing the same `graph_backend` can read each other's long-term
memories, enabling knowledge transfer across the swarm.

---

## Configuration Reference

### MemoryManager

| Parameter       | Type               | Default | Description                          |
|-----------------|--------------------|---------|--------------------------------------|
| `graph_backend` | `GraphBackend`     | `None`  | Graph DB for long-term persistence   |
| `llm_provider`  | `BaseLLMProvider`  | `None`  | LLM for summarization/improvement    |

### ShortTermMemory

| Parameter             | Type    | Default | Description                            |
|-----------------------|---------|---------|----------------------------------------|
| `max_entries`         | `int`   | 100     | Max entries before forced compression  |
| `relevance_threshold` | `float` | 0.15    | Entries below this are pruned          |
| `refresh_interval`    | `int`   | 10      | Operations between auto-refresh cycles |

### LongTermMemory

| Parameter             | Type           | Default | Description                          |
|-----------------------|----------------|---------|--------------------------------------|
| `graph_backend`       | `GraphBackend` | `None`  | Graph DB for persistence             |
| `relevance_threshold` | `float`        | 0.10    | Threshold for decay-based pruning    |

### MemoryGarbageCollector

| Parameter                        | Type    | Default | Description                          |
|----------------------------------|---------|---------|--------------------------------------|
| `memory_manager`                 | `MemoryManager` | required | The manager to operate on     |
| `llm_provider`                   | `BaseLLMProvider` | `None` | LLM for summarize/improve     |
| `relevance_threshold`            | `float` | 0.15    | Auto-forget threshold                |
| `duplicate_similarity_threshold` | `float` | 0.7     | Jaccard threshold for dedup          |

---

## Module Structure

```
openagentflow/memory/
    __init__.py         # Public API: MemoryManager, MemoryTier, MemoryEntry, etc.
    base.py             # MemoryTier enum, MemoryEntry dataclass, MemoryManager
    fleeting.py         # FleetingMemory (per-turn dict scratchpad)
    short_term.py       # ShortTermMemory (sliding window + summarization)
    long_term.py        # LongTermMemory (graph-backed persistent store)
    gc.py               # MemoryGarbageCollector (auto-forget, prune, improve)
```

---

## Further Reading

- `openagentflow/core/types.py` -- `MemoryConfig`, `MemoryType` enums
- `openagentflow/runtime/executor.py` -- `AgentExecutor` that calls GC lifecycle hooks
- `openagentflow/core/agent.py` -- `@agent` decorator with `memory` parameter
- `openagentflow/graph/` -- Graph backends (SQLite, TinkerPop) for long-term persistence
- `openagentflow/reasoning/` -- Reasoning engines that interact with the memory system
