"""
OpenAgentFlow Example 07: Graph Tracing

Demonstrates:
- Recording execution as a DAG
- Adding vertices (agents, tools, reasoning steps)
- Adding edges (CALLED, LEADS_TO, SYNTHESIZES)
- Querying traces by run_id
"""
import asyncio
from openagentflow.graph import SQLiteGraphBackend


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 07: Graph Tracing")
    print("=" * 60)

    # Create an in-memory graph backend
    backend = SQLiteGraphBackend(":memory:")

    run_id = "run-example-001"

    # --- Build an execution trace ---
    print(f"\nBuilding trace for run: {run_id}")

    # Agent vertex
    await backend.add_vertex("agent-planner", "agent", {
        "name": "planner",
        "model": "claude-opus-4-6",
        "run_id": run_id,
    })

    # Tool vertices
    await backend.add_vertex("tool-search", "tool", {
        "name": "search_knowledge",
        "run_id": run_id,
    })
    await backend.add_vertex("tool-analyze", "tool", {
        "name": "analyze_text",
        "run_id": run_id,
    })

    # Reasoning step vertices
    await backend.add_vertex("step-think-1", "reasoning", {
        "step_type": "thought",
        "content": "I need to search for information about the topic",
        "run_id": run_id,
    })
    await backend.add_vertex("step-think-2", "reasoning", {
        "step_type": "observation",
        "content": "Found 3 relevant articles",
        "run_id": run_id,
    })
    await backend.add_vertex("step-synthesis", "reasoning", {
        "step_type": "synthesis",
        "content": "Combining findings into a coherent answer",
        "run_id": run_id,
    })

    # Edges: Agent -> Tool calls
    await backend.add_edge("agent-planner", "tool-search", "CALLED", {
        "duration_ms": 150,
        "input": "AI trends",
    })
    await backend.add_edge("agent-planner", "tool-analyze", "CALLED", {
        "duration_ms": 80,
        "input": "article text",
    })

    # Edges: Reasoning flow
    await backend.add_edge("step-think-1", "tool-search", "LEADS_TO", {})
    await backend.add_edge("tool-search", "step-think-2", "LEADS_TO", {})
    await backend.add_edge("step-think-2", "tool-analyze", "LEADS_TO", {})
    await backend.add_edge("tool-analyze", "step-synthesis", "LEADS_TO", {})
    await backend.add_edge("step-think-1", "step-synthesis", "SYNTHESIZES", {})

    # --- Query the trace ---
    print("\n--- Full Trace ---")
    trace = await backend.get_full_trace(run_id)

    print(f"Vertices: {len(trace['vertices'])}")
    for v in trace["vertices"]:
        print(f"  [{v['label']}] {v['id']}: {v['properties'].get('name', v['properties'].get('step_type', ''))}")

    print(f"\nEdges: {len(trace['edges'])}")
    for e in trace["edges"]:
        print(f"  {e['source_id']} --{e['label']}--> {e['target_id']}")

    # --- Query by label ---
    print("\n--- Query by Label ---")
    agents = [v for v in trace["vertices"] if v["label"] == "agent"]
    tools = [v for v in trace["vertices"] if v["label"] == "tool"]
    reasoning = [v for v in trace["vertices"] if v["label"] == "reasoning"]
    print(f"Agents: {len(agents)}, Tools: {len(tools)}, Reasoning steps: {len(reasoning)}")

    print("\n" + "=" * 60)
    print("Trace recorded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
