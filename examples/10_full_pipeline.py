"""
OpenAgentFlow Example 10: Full Pipeline

Demonstrates an end-to-end workflow combining:
- Custom tools with @tool
- Memory system (store, recall, context window, GC)
- Graph tracing (vertices, edges, full trace)

This is the most comprehensive example showing how the
infrastructure components work together without requiring
an LLM provider.
"""
import asyncio
from openagentflow import tool
from openagentflow.memory import MemoryManager, MemoryTier
from openagentflow.graph import SQLiteGraphBackend


@tool
def analyze_codebase(path: str) -> dict:
    """Analyze a codebase and return metrics."""
    # Simulated analysis
    return {
        "files": 42,
        "total_lines": 8500,
        "languages": ["Python", "TypeScript"],
        "test_coverage": 0.78,
        "complexity_avg": 4.2,
    }


@tool
def check_dependencies(manifest: str) -> list[dict]:
    """Check dependencies for known vulnerabilities."""
    # Simulated vulnerability check
    return [
        {"package": "requests", "version": "2.28.0", "severity": "low", "cve": "CVE-2023-XXXX"},
        {"package": "pyyaml", "version": "5.4", "severity": "high", "cve": "CVE-2023-YYYY"},
    ]


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 10: Full Pipeline")
    print("=" * 60)

    # --- Setup Infrastructure ---
    print("\n--- Setting up infrastructure ---")

    # Graph backend for tracing
    graph = SQLiteGraphBackend(":memory:")
    print("  Graph backend: SQLite (in-memory)")

    # Memory system
    memory = MemoryManager(graph_backend=graph)
    print("  Memory manager: initialized")

    # --- Store Context in Memory ---
    print("\n--- Storing context in memory ---")
    await memory.remember(
        "project_goal",
        "Audit the e-commerce platform for security and quality",
        tier=MemoryTier.SHORT_TERM,
        importance=0.9,
    )
    await memory.remember(
        "past_finding",
        "Previous audit found SQL injection in user search endpoint",
        tier=MemoryTier.LONG_TERM,
        importance=0.8,
        tags=["security", "sql-injection"],
    )
    print("  Stored project goal and past findings")

    # --- Run Tools Directly ---
    print("\n--- Running analysis tools ---")
    metrics = analyze_codebase("/path/to/project")
    print(f"  Codebase: {metrics['files']} files, {metrics['total_lines']} lines")
    print(f"  Languages: {', '.join(metrics['languages'])}")
    print(f"  Test coverage: {metrics['test_coverage']:.0%}")

    vulns = check_dependencies("requirements.txt")
    print(f"  Vulnerabilities found: {len(vulns)}")
    for v in vulns:
        print(f"    [{v['severity'].upper()}] {v['package']} {v['version']} ({v['cve']})")

    # Store findings in memory
    await memory.remember(
        "current_metrics",
        metrics,
        tier=MemoryTier.SHORT_TERM,
        importance=0.7,
    )
    await memory.remember(
        "current_vulns",
        f"Found {len(vulns)} vulnerabilities including {vulns[0]['cve']}",
        tier=MemoryTier.SHORT_TERM,
        importance=0.85,
    )

    # --- Record in Graph ---
    print("\n--- Recording execution trace ---")
    run_id = "audit-run-001"

    await graph.add_vertex("audit-agent", "agent", {
        "name": "security_auditor",
        "model": "claude-opus-4-6",
        "run_id": run_id,
    })
    await graph.add_vertex("tool-analyze", "tool", {
        "name": "analyze_codebase",
        "run_id": run_id,
    })
    await graph.add_vertex("tool-deps", "tool", {
        "name": "check_dependencies",
        "run_id": run_id,
    })
    await graph.add_edge("audit-agent", "tool-analyze", "CALLED", {"duration_ms": 250})
    await graph.add_edge("audit-agent", "tool-deps", "CALLED", {"duration_ms": 180})

    trace = await graph.get_full_trace(run_id)
    print(f"  Trace: {len(trace['vertices'])} vertices, {len(trace['edges'])} edges")

    # --- Build Context Window ---
    print("\n--- Building context window ---")
    context = await memory.get_context_window(max_tokens=4000)
    print(f"  Context entries: {len(context)}")
    for entry in context:
        print(f"    [{entry.tier.value}] {entry.key} (relevance={entry.relevance_score:.2f})")

    # --- Cleanup ---
    print("\n--- Running garbage collection ---")
    await memory.gc.on_turn_end()
    await memory.run_gc()
    print("  GC complete")

    print("\n" + "=" * 60)
    print("Full pipeline demo complete!")
    print("=" * 60)
    print("\nTo extend this with LLM-powered agents, see examples 02-04.")
    print("Set ANTHROPIC_API_KEY or install Ollama for local inference.")


if __name__ == "__main__":
    asyncio.run(main())
