# OpenAgentFlow Memory System

Full documentation: [docs/memory.md](../../../docs/memory.md)

## Quick Reference

Three-tier memory hierarchy with automatic lifecycle management:

| Tier | Purpose | Persistence |
|------|---------|-------------|
| Fleeting | Per-turn scratchpad | Cleared every turn |
| Short-Term | Session context | Session duration |
| Long-Term | Persistent storage | Across sessions (graph-backed) |

## Basic Usage

```python
from openagentflow.memory import MemoryManager, MemoryTier

memory = MemoryManager()
await memory.remember("pattern", "always check imports", importance=0.9)
results = await memory.recall("import issues")
context = await memory.get_context_window(max_tokens=4000)
await memory.run_gc()
```
