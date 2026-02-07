"""
@swarm decorator for parallel multi-agent coordination.

Swarms execute agents in parallel with configurable consensus mechanisms:
- voting: Agents vote on the best response
- synthesis: A coordinator synthesizes all responses
- first: Return the first successful response

When a graph backend is provided, the swarm execution DAG is recorded:
swarm vertex -> parallel agent vertices via "PARALLEL" edges,
then -> synthesis vertex (for synthesis strategy).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

from openagentflow.core.types import (
    AgentResult,
    AgentStatus,
    ExecutionHash,
    SwarmSpec,
)
from openagentflow.exceptions import AgentError

if TYPE_CHECKING:
    from openagentflow.graph.base import GraphBackend

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Swarm registry
_swarm_registry: dict[str, SwarmSpec] = {}


@dataclass
class SwarmResult:
    """Result from a swarm execution."""

    swarm_name: str
    status: AgentStatus
    output: Any = None
    consensus_method: str = "voting"
    agreement_score: float = 0.0
    error: str | None = None
    agent_results: list[AgentResult] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_ms: float = 0.0
    execution_hash: ExecutionHash | None = None


def get_swarm(name: str) -> SwarmSpec | None:
    """Get a registered swarm by name."""
    return _swarm_registry.get(name)


async def _trace_safe(coro: Any) -> None:
    """Execute a tracing coroutine, catching and logging any errors.

    Tracing failures must never crash agent/swarm execution.
    """
    try:
        await coro
    except Exception:
        logger.warning("Graph tracing operation failed in swarm", exc_info=True)


async def _record_swarm_results(
    graph_backend: GraphBackend,
    swarm_vertex_id: str,
    run_id: str,
    agent_results: list[AgentResult],
    agent_names: list[str],
    strategy: str,
    output: Any,
    agreement: float,
    total_tokens: int,
    total_cost: float,
    duration_ms: float,
) -> None:
    """Record all swarm agent results and synthesis into the graph.

    This is separated out so it can be called once after parallel
    execution completes.
    """
    # Record each agent result as a vertex with PARALLEL edge
    for i, result in enumerate(agent_results):
        agent_name = result.agent_name or (agent_names[i] if i < len(agent_names) else f"agent_{i}")
        agent_vertex_id = f"agent-{agent_name}-{result.run_id[:12] if result.run_id else i}"

        await _trace_safe(
            graph_backend.add_vertex(agent_vertex_id, "agent", {
                "run_id": run_id,
                "agent_name": agent_name,
                "parallel_index": i,
                "status": result.status.name,
                "total_tokens": result.total_tokens,
                "total_cost": result.total_cost,
                "duration_ms": result.duration_ms,
                "output": str(result.output)[:2000] if result.output else "",
                "error": result.error or "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        )
        await _trace_safe(
            graph_backend.add_edge(
                swarm_vertex_id, agent_vertex_id,
                "PARALLEL", {
                    "parallel_index": i,
                    "agent_name": agent_name,
                })
        )

    # Record synthesis/consensus vertex
    synthesis_vertex_id = f"synthesis-{swarm_vertex_id}"
    await _trace_safe(
        graph_backend.add_vertex(synthesis_vertex_id, "synthesis", {
            "run_id": run_id,
            "strategy": strategy,
            "agreement_score": agreement,
            "output": str(output)[:2000] if output else "",
            "num_agents": len(agent_results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    )

    # Link all agent vertices to synthesis vertex
    for i, result in enumerate(agent_results):
        agent_name = result.agent_name or (agent_names[i] if i < len(agent_names) else f"agent_{i}")
        agent_vertex_id = f"agent-{agent_name}-{result.run_id[:12] if result.run_id else i}"
        await _trace_safe(
            graph_backend.add_edge(
                agent_vertex_id, synthesis_vertex_id,
                "CONTRIBUTES_TO", {
                    "agent_name": agent_name,
                    "parallel_index": i,
                })
        )

    # Update swarm vertex with final results
    await _trace_safe(
        graph_backend.update_vertex(swarm_vertex_id, {
            "status": AgentStatus.SUCCEEDED.name,
            "output": str(output)[:2000] if output else "",
            "agreement_score": agreement,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "duration_ms": duration_ms,
            "agents_completed": len(agent_results),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    )


def swarm(
    agents: list[str] | None = None,
    *,
    name: str | None = None,
    strategy: Literal["voting", "synthesis", "first"] = "voting",
    min_agreement: float = 0.5,
    timeout: float = 600.0,
) -> Callable[[F], F]:
    """
    Decorator to define a parallel multi-agent swarm.

    Example:
        @agent(model="claude-sonnet-4-20250514")
        async def analyst1(data: str) -> str:
            '''First analyst perspective.'''
            pass

        @agent(model="claude-sonnet-4-20250514")
        async def analyst2(data: str) -> str:
            '''Second analyst perspective.'''
            pass

        @swarm(agents=["analyst1", "analyst2"], strategy="voting")
        async def consensus(data: str) -> str:
            '''Get consensus from multiple analysts.'''
            pass

        # Usage:
        result = await consensus("Analyze this data...")
        # Runs analyst1 and analyst2 in parallel, returns majority response

        # With graph tracing:
        from openagentflow.graph import SQLiteGraphBackend
        backend = SQLiteGraphBackend(":memory:")
        result = await consensus("Analyze this data...", graph_backend=backend)

    Strategies:
        - voting: Return the most common response
        - synthesis: Combine all responses (returns list of outputs)
        - first: Return the first successful response

    Args:
        agents: List of agent names to execute in parallel
        name: Swarm name (defaults to function name)
        strategy: Consensus strategy
        min_agreement: Minimum agreement for voting (0-1)
        timeout: Total execution timeout in seconds
    """

    def decorator(fn: F) -> F:
        swarm_name = name or fn.__name__
        agent_list = agents or []

        spec = SwarmSpec(
            name=swarm_name,
            agents=agent_list,
            consensus_strategy=strategy,
            min_agreement=min_agreement,
            timeout_seconds=timeout,
        )

        _swarm_registry[swarm_name] = spec

        @functools.wraps(fn)
        async def wrapper(
            *args: Any,
            parent_hash: str | list[str] | None = None,
            graph_backend: GraphBackend | None = None,
            **kwargs: Any,
        ) -> SwarmResult:
            """Execute all agents in parallel and apply consensus."""
            from openagentflow.core.agent import get_agent

            start_time = time.time()

            # Generate swarm execution hash
            parent_hashes = (
                [parent_hash] if isinstance(parent_hash, str) else (parent_hash or [])
            )
            swarm_hash = ExecutionHash.generate(parent_hashes)
            run_id = swarm_hash.value

            # Swarm vertex ID
            swarm_vertex_id = f"swarm-{swarm_name}-{swarm_hash.value[:12]}"

            # Record swarm vertex (if tracing)
            if graph_backend is not None:
                await _trace_safe(
                    graph_backend.add_vertex(swarm_vertex_id, "swarm", {
                        "run_id": run_id,
                        "swarm_name": swarm_name,
                        "agents": json.dumps(agent_list),
                        "strategy": strategy,
                        "min_agreement": min_agreement,
                        "timeout": timeout,
                        "status": AgentStatus.THINKING.name,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    })
                )

            # Get input
            input_data = args[0] if args else kwargs.get("input", "")

            # Create tasks for all agents
            tasks = []
            task_agent_names = []
            for agent_name in agent_list:
                agent_spec = get_agent(agent_name)
                if agent_spec is None:
                    continue

                # Get the agent wrapper (includes parent_hash support)
                if agent_spec.wrapper:
                    agent_func = agent_spec.wrapper
                else:
                    agent_func = agent_spec.func
                    if hasattr(agent_func, "_async_call"):
                        agent_func = agent_func._async_call

                # Each agent gets the swarm hash as parent
                task = asyncio.create_task(
                    agent_func(input_data, parent_hash=swarm_hash.value)
                )
                tasks.append((agent_name, task))
                task_agent_names.append(agent_name)

            if not tasks:
                duration_ms = (time.time() - start_time) * 1000
                if graph_backend is not None:
                    await _trace_safe(
                        graph_backend.update_vertex(swarm_vertex_id, {
                            "status": AgentStatus.FAILED.name,
                            "error": "No valid agents found in swarm",
                            "duration_ms": duration_ms,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                return SwarmResult(
                    swarm_name=swarm_name,
                    status=AgentStatus.FAILED,
                    error="No valid agents found in swarm",
                    consensus_method=strategy,
                    duration_ms=duration_ms,
                    execution_hash=swarm_hash,
                )

            # Execute all in parallel with timeout
            try:
                results_with_names = await asyncio.wait_for(
                    asyncio.gather(
                        *[task for _, task in tasks],
                        return_exceptions=True,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                if graph_backend is not None:
                    await _trace_safe(
                        graph_backend.update_vertex(swarm_vertex_id, {
                            "status": AgentStatus.TIMEOUT.name,
                            "error": f"Swarm timed out after {timeout}s",
                            "duration_ms": duration_ms,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                return SwarmResult(
                    swarm_name=swarm_name,
                    status=AgentStatus.TIMEOUT,
                    error=f"Swarm timed out after {timeout}s",
                    consensus_method=strategy,
                    duration_ms=duration_ms,
                    execution_hash=swarm_hash,
                )

            # Collect successful results
            agent_results: list[AgentResult] = []
            total_tokens = 0
            total_cost = 0.0

            for i, result in enumerate(results_with_names):
                if isinstance(result, Exception):
                    continue
                if isinstance(result, AgentResult):
                    agent_results.append(result)
                    total_tokens += result.total_tokens
                    total_cost += result.total_cost

            if not agent_results:
                duration_ms = (time.time() - start_time) * 1000
                if graph_backend is not None:
                    await _trace_safe(
                        graph_backend.update_vertex(swarm_vertex_id, {
                            "status": AgentStatus.FAILED.name,
                            "error": "All agents failed",
                            "duration_ms": duration_ms,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                return SwarmResult(
                    swarm_name=swarm_name,
                    status=AgentStatus.FAILED,
                    error="All agents failed",
                    agent_results=agent_results,
                    consensus_method=strategy,
                    duration_ms=duration_ms,
                    execution_hash=swarm_hash,
                )

            # Apply consensus strategy
            if strategy == "first":
                # Return first successful result
                output = agent_results[0].output
                agreement = 1.0 / len(agent_results)
            elif strategy == "synthesis":
                # Return all outputs as a list
                output = [r.output for r in agent_results]
                agreement = 1.0
            else:  # voting
                # Vote on most common output
                outputs = [str(r.output) for r in agent_results]
                counter = Counter(outputs)
                most_common, count = counter.most_common(1)[0]
                output = most_common
                agreement = count / len(outputs)

                # Find the original output (not stringified)
                for r in agent_results:
                    if str(r.output) == most_common:
                        output = r.output
                        break

            duration_ms = (time.time() - start_time) * 1000

            # Record swarm results in graph (if tracing)
            if graph_backend is not None:
                try:
                    await _record_swarm_results(
                        graph_backend=graph_backend,
                        swarm_vertex_id=swarm_vertex_id,
                        run_id=run_id,
                        agent_results=agent_results,
                        agent_names=task_agent_names,
                        strategy=strategy,
                        output=output,
                        agreement=agreement,
                        total_tokens=total_tokens,
                        total_cost=total_cost,
                        duration_ms=duration_ms,
                    )
                except Exception:
                    logger.warning("Failed to record swarm results in graph", exc_info=True)

            return SwarmResult(
                swarm_name=swarm_name,
                status=AgentStatus.SUCCEEDED,
                output=output,
                consensus_method=strategy,
                agreement_score=agreement,
                agent_results=agent_results,
                total_tokens=total_tokens,
                total_cost=total_cost,
                duration_ms=duration_ms,
                execution_hash=swarm_hash,
            )

        wrapper._swarm_spec = spec  # type: ignore
        return wrapper  # type: ignore

    return decorator


async def run_swarm(
    swarm_name: str,
    input_data: Any,
    parent_hash: str | list[str] | None = None,
    graph_backend: GraphBackend | None = None,
) -> SwarmResult:
    """Run a registered swarm by name.

    Args:
        swarm_name: Name of the registered swarm to run.
        input_data: Input data for all agents in the swarm.
        parent_hash: Parent execution hash(es) for lineage tracking.
        graph_backend: Optional graph backend for execution tracing.
    """
    spec = get_swarm(swarm_name)
    if spec is None:
        raise AgentError(f"Swarm not found: {swarm_name}")

    from openagentflow.core.agent import get_agent

    start_time = time.time()
    parent_hashes = [parent_hash] if isinstance(parent_hash, str) else (parent_hash or [])
    swarm_hash = ExecutionHash.generate(parent_hashes)
    run_id = swarm_hash.value

    # Swarm vertex ID
    swarm_vertex_id = f"swarm-{swarm_name}-{swarm_hash.value[:12]}"

    # Record swarm vertex (if tracing)
    if graph_backend is not None:
        await _trace_safe(
            graph_backend.add_vertex(swarm_vertex_id, "swarm", {
                "run_id": run_id,
                "swarm_name": swarm_name,
                "agents": json.dumps(spec.agents),
                "strategy": spec.consensus_strategy,
                "min_agreement": spec.min_agreement,
                "timeout": spec.timeout_seconds,
                "status": AgentStatus.THINKING.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })
        )

    tasks = []
    task_agent_names = []
    for agent_name in spec.agents:
        agent_spec = get_agent(agent_name)
        if agent_spec is None:
            continue

        if agent_spec.wrapper:
            agent_func = agent_spec.wrapper
        else:
            agent_func = agent_spec.func
            if hasattr(agent_func, "_async_call"):
                agent_func = agent_func._async_call

        task = asyncio.create_task(agent_func(input_data, parent_hash=swarm_hash.value))
        tasks.append(task)
        task_agent_names.append(agent_name)

    if not tasks:
        duration_ms = (time.time() - start_time) * 1000
        if graph_backend is not None:
            await _trace_safe(
                graph_backend.update_vertex(swarm_vertex_id, {
                    "status": AgentStatus.FAILED.name,
                    "error": "No valid agents found",
                    "duration_ms": duration_ms,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
            )
        return SwarmResult(
            swarm_name=swarm_name,
            status=AgentStatus.FAILED,
            error="No valid agents found",
            consensus_method=spec.consensus_strategy,
            duration_ms=duration_ms,
            execution_hash=swarm_hash,
        )

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=spec.timeout_seconds,
        )
    except asyncio.TimeoutError:
        duration_ms = (time.time() - start_time) * 1000
        if graph_backend is not None:
            await _trace_safe(
                graph_backend.update_vertex(swarm_vertex_id, {
                    "status": AgentStatus.TIMEOUT.name,
                    "error": f"Swarm timed out after {spec.timeout_seconds}s",
                    "duration_ms": duration_ms,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
            )
        return SwarmResult(
            swarm_name=swarm_name,
            status=AgentStatus.TIMEOUT,
            error=f"Swarm timed out after {spec.timeout_seconds}s",
            consensus_method=spec.consensus_strategy,
            duration_ms=duration_ms,
            execution_hash=swarm_hash,
        )

    agent_results = [r for r in results if isinstance(r, AgentResult)]
    total_tokens = sum(r.total_tokens for r in agent_results)
    total_cost = sum(r.total_cost for r in agent_results)

    if not agent_results:
        duration_ms = (time.time() - start_time) * 1000
        if graph_backend is not None:
            await _trace_safe(
                graph_backend.update_vertex(swarm_vertex_id, {
                    "status": AgentStatus.FAILED.name,
                    "error": "All agents failed",
                    "duration_ms": duration_ms,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
            )
        return SwarmResult(
            swarm_name=swarm_name,
            status=AgentStatus.FAILED,
            error="All agents failed",
            consensus_method=spec.consensus_strategy,
            duration_ms=duration_ms,
            execution_hash=swarm_hash,
        )

    strategy = spec.consensus_strategy
    if strategy == "first":
        output = agent_results[0].output
        agreement = 1.0 / len(agent_results)
    elif strategy == "synthesis":
        output = [r.output for r in agent_results]
        agreement = 1.0
    else:
        outputs = [str(r.output) for r in agent_results]
        counter = Counter(outputs)
        most_common, count = counter.most_common(1)[0]
        output = most_common
        agreement = count / len(outputs)
        for r in agent_results:
            if str(r.output) == most_common:
                output = r.output
                break

    duration_ms = (time.time() - start_time) * 1000

    # Record swarm results in graph (if tracing)
    if graph_backend is not None:
        try:
            await _record_swarm_results(
                graph_backend=graph_backend,
                swarm_vertex_id=swarm_vertex_id,
                run_id=run_id,
                agent_results=agent_results,
                agent_names=task_agent_names,
                strategy=strategy,
                output=output,
                agreement=agreement,
                total_tokens=total_tokens,
                total_cost=total_cost,
                duration_ms=duration_ms,
            )
        except Exception:
            logger.warning("Failed to record swarm results in graph", exc_info=True)

    return SwarmResult(
        swarm_name=swarm_name,
        status=AgentStatus.SUCCEEDED,
        output=output,
        consensus_method=strategy,
        agreement_score=agreement,
        agent_results=agent_results,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_ms=duration_ms,
        execution_hash=swarm_hash,
    )
