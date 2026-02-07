# Graph Tracing

## Overview

OpenAgentFlow records every agent execution as a directed acyclic graph (DAG) of vertices and edges. This provides full auditability, debugging, and post-hoc analysis of agent behavior.

Each run produces a trace consisting of typed vertices (agents, tools, reasoning steps, memory operations) connected by typed edges (calls, sequential flow, adversarial challenges, synthesis). Traces can be queried, exported, and visualized after execution completes.

## Backends

### SQLite (Default)

Zero-configuration, file-based or in-memory graph storage. Suitable for development, testing, and single-process deployments.

```python
from openagentflow.graph import SQLiteGraphBackend

backend = SQLiteGraphBackend(":memory:")  # or "traces.db"
await backend.add_vertex("agent-1", "agent", {"name": "planner", "run_id": "run-1"})
await backend.add_vertex("tool-1", "tool", {"name": "search", "run_id": "run-1"})
await backend.add_edge("agent-1", "tool-1", "CALLED", {"duration_ms": 150})

trace = await backend.get_full_trace("run-1")
print(f"Vertices: {len(trace['vertices'])}, Edges: {len(trace['edges'])}")
```

### Gremlin (Production)

For production deployments, use Apache TinkerPop-compatible graph databases such as JanusGraph, Amazon Neptune, or Azure CosmosDB.

**Install:**

```bash
pip install openagentflow[gremlin]
```

**Usage:**

```python
from openagentflow.graph import GremlinGraphBackend

backend = GremlinGraphBackend(
    endpoint="wss://your-neptune-endpoint:8182/gremlin",
    traversal_source="g",
)

await backend.add_vertex("agent-1", "agent", {"name": "planner", "run_id": "run-1"})
await backend.add_vertex("tool-1", "tool", {"name": "search", "run_id": "run-1"})
await backend.add_edge("agent-1", "tool-1", "CALLED", {"duration_ms": 150})

trace = await backend.get_full_trace("run-1")
```

## Vertex Types

| Type        | Description             |
|-------------|-------------------------|
| `agent`     | Agent execution node    |
| `tool`      | Tool invocation node    |
| `reasoning` | Reasoning step node     |
| `memory`    | Memory operation node   |

## Edge Types

| Type           | Description              |
|----------------|--------------------------|
| `CALLED`       | Agent called a tool      |
| `LEADS_TO`     | Sequential flow          |
| `CHALLENGES`   | Adversarial reasoning    |
| `SYNTHESIZES`  | Combining results        |

## Querying Traces

### Full Trace Retrieval

Retrieve all vertices and edges for a given run:

```python
trace = await backend.get_full_trace("run-1")

for vertex in trace["vertices"]:
    print(f"  {vertex['id']} ({vertex['label']})")

for edge in trace["edges"]:
    print(f"  {edge['from']} --{edge['label']}--> {edge['to']}")
```

### Filtering by Vertex Label

```python
tool_vertices = await backend.get_vertices_by_label("tool", run_id="run-1")
for v in tool_vertices:
    print(f"Tool: {v['properties']['name']}")
```

### Querying Outbound Edges

```python
edges = await backend.get_edges_from("agent-1")
for e in edges:
    print(f"  --{e['label']}--> {e['to']}")
```

## API Reference

### `GraphBackend` (Abstract Base)

All backends implement the following interface:

| Method                   | Description                                           |
|--------------------------|-------------------------------------------------------|
| `add_vertex(id, label, properties)` | Add a vertex to the graph                   |
| `add_edge(from_id, to_id, label, properties)` | Add a directed edge between two vertices |
| `get_full_trace(run_id)` | Return all vertices and edges for a run              |
| `get_vertices_by_label(label, run_id)` | Return vertices matching a label and run    |
| `get_edges_from(vertex_id)` | Return all outbound edges from a vertex           |

### `SQLiteGraphBackend`

- **Constructor:** `SQLiteGraphBackend(db_path: str)` -- pass `":memory:"` for in-memory or a file path for persistent storage.

### `GremlinGraphBackend`

- **Constructor:** `GremlinGraphBackend(endpoint: str, traversal_source: str = "g")` -- connect to any TinkerPop-compatible server.
