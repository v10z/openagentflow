---
description: Use OpenAgentFlow framework to create agents, tools, reasoning engines, memory, and graph traces
allowed-tools: Bash(python:*), Read, Glob, Grep, Write, Edit
---

# OpenAgentFlow Skill

You have access to the OpenAgentFlow framework (v0.2.0). Use it to build and run AI agents, tools, reasoning engines, and memory systems.

## Quick Imports

```python
# Core decorators
from openagentflow import agent, tool, chain, swarm, configure

# Reasoning engines (10 available)
from openagentflow.reasoning import (
    DialecticalSpiral, DreamWakeCycle, MetaCognitiveLoop,
    AdversarialSelfPlay, EvolutionaryThought, FractalRecursion,
    ResonanceNetwork, TemporalRecursion, SimulatedAnnealing,
    SocraticInterrogation, ReasoningEngine, ReasoningTrace,
)

# Memory system
from openagentflow.memory import MemoryManager, MemoryTier, MemoryEntry

# Graph tracing
from openagentflow.graph import SQLiteGraphBackend

# Meta-agent (JIT tool creation)
from openagentflow.meta import Sandbox, ToolFactory

# Distributed compute
from openagentflow.distributed import ComputeCluster, ComputeNode, ComputeBackend

# Built-in tools (99 across 9 categories)
from openagentflow.tools import text, code, data, web, math, media, datetime, ai, system

# LLM providers
from openagentflow.llm.providers import OllamaProvider, is_ollama_available
```

## Usage Patterns

### 1. Define and Use a Tool

```python
from openagentflow import tool

@tool
def analyze(code: str) -> dict:
    """Analyze code quality."""
    return {"lines": len(code.splitlines()), "chars": len(code)}

# Call directly (returns raw result)
result = analyze("print('hello')")

# Access auto-generated spec
print(analyze._tool_spec.name)          # "analyze"
print(analyze._tool_spec.input_schema)  # JSON Schema from type hints
```

### 2. Run Built-in Tools

```python
from openagentflow.tools import text, math, code

text.extract_emails("hi bob@test.com")    # ["bob@test.com"]
text.text_to_slug("Hello World!")          # "hello-world"
math.is_prime(17)                          # True
math.fibonacci(10)                         # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
code.calculate_complexity(source_code)     # Cyclomatic complexity score
```

### 3. Define and Run an Agent

```python
import asyncio
from openagentflow import agent, tool

@tool
def search(query: str) -> list[dict]:
    """Search for information."""
    return [{"title": "Result", "url": "https://example.com"}]

@agent(model="claude-sonnet-4-20250514", tools=[search])
async def researcher(question: str) -> str:
    """Research agent."""
    pass  # ReAct loop handles execution

result = asyncio.run(researcher("What are AI trends?"))
print(result.output)
print(result.status)        # AgentStatus.SUCCEEDED
print(result.total_tokens)  # Token usage
```

### 4. Use a Reasoning Engine

```python
import asyncio
from openagentflow.reasoning import AdversarialSelfPlay

engine = AdversarialSelfPlay(max_rounds=5)

async def run():
    trace = await engine.reason("Design a secure auth system", llm_provider)
    print(trace.final_output)
    print(f"Steps: {len(trace.steps)}, LLM calls: {trace.total_llm_calls}")
    # Inspect the DAG
    dag = trace.to_dag()
    for step in trace.get_steps_by_type("judge"):
        print(f"Judge verdict: {step.content[:100]}")

asyncio.run(run())
```

### 5. Use the Memory System

```python
import asyncio
from openagentflow.memory import MemoryManager, MemoryTier

async def run():
    mm = MemoryManager()
    await mm.remember("pattern", "always check imports", tier=MemoryTier.SHORT_TERM, importance=0.9)
    await mm.remember("scratch", "temp calc", tier=MemoryTier.FLEETING)

    results = await mm.recall("imports")
    context = await mm.get_context_window(max_tokens=4000)

    # Garbage collection
    await mm.gc.on_turn_end()   # Clears fleeting
    await mm.run_gc()           # Full GC cycle

asyncio.run(run())
```

### 6. Graph Tracing

```python
import asyncio
from openagentflow.graph import SQLiteGraphBackend

async def run():
    backend = SQLiteGraphBackend(":memory:")
    await backend.add_vertex("agent-1", "agent", {"name": "planner", "run_id": "run-1"})
    await backend.add_vertex("tool-1", "tool", {"name": "search", "run_id": "run-1"})
    await backend.add_edge("agent-1", "tool-1", "CALLED", {"duration_ms": 150})

    trace = await backend.get_full_trace("run-1")
    print(f"Vertices: {len(trace['vertices'])}, Edges: {len(trace['edges'])}")

asyncio.run(run())
```

### 7. JIT Tool Creation (Meta-Agent)

```python
from openagentflow.meta import ToolFactory, Sandbox

# Validate code safety first
sandbox = Sandbox()
safe, reason = sandbox.validate_source("def double(n: int) -> int:\n    return n * 2")

# Create a tool at runtime
factory = ToolFactory()
spec = factory.create_tool(
    name="double",
    description="Double a number",
    source_code="def double(n: int) -> int:\n    return n * 2",
)
result = factory.test_tool("double", {"n": 21})  # Returns 42
```

### 8. Distributed Compute

```python
from openagentflow.distributed import ComputeCluster, ComputeNode, ComputeBackend

cluster = ComputeCluster(name="inference")
cluster.add_node(ComputeNode(
    node_id="gpu1",
    backend=ComputeBackend.HTTP,
    endpoint="http://gpu1:11434",
))
available = cluster.get_available_nodes()
```

## Important Gotchas

- `FleetingMemory.recall()` is **async** -- always `await` it
- `OllamaProvider.__init__` takes `base_url`, not `model`
- `OllamaProvider.estimate_cost()` requires 3 args: `(input_tokens, output_tokens, model_id)`
- `python-dateutil` is required by datetime tools (in core deps since v0.2.0)
- Tool decorator returns a **sync wrapper** for direct calls; use `._async_call()` for `ToolResult`
- Agents with `pass` body let the ReAct executor loop handle LLM calls
- All graph operations are async (SQLite uses `run_in_executor`)
- Reasoning engines need an LLM provider passed to `engine.reason(prompt, provider)`

## Reasoning Engine Selection Guide

| Need | Engine |
|------|--------|
| Deep analysis | `DialecticalSpiral` |
| Creative solutions | `DreamWakeCycle` |
| Complex planning | `MetaCognitiveLoop` |
| Robust outputs | `AdversarialSelfPlay` |
| Optimization | `EvolutionaryThought` |
| Hierarchical tasks | `FractalRecursion` |
| Coherent synthesis | `ResonanceNetwork` |
| Risk planning | `TemporalRecursion` |
| Escaping local optima | `SimulatedAnnealing` |
| Critical thinking | `SocraticInterrogation` |

## Project Location

- Source: `src/openagentflow/`
- Tests: `tests/`
- Run tests: `python -m pytest tests/ -v`
