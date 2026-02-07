"""
@chain decorator for sequential agent pipelines.

Chains execute agents in order, passing output from one to the next.
Each agent's output becomes the input for the next agent.
Lineage is tracked through ExecutionHash.

When a graph backend is provided, the chain execution DAG is recorded:
chain vertex -> agent vertices (in order) via "STEP_N" edges.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from openagentflow.core.types import (
    AgentResult,
    AgentStatus,
    ChainSpec,
    ExecutionHash,
)
from openagentflow.exceptions import AgentError

if TYPE_CHECKING:
    from openagentflow.graph.base import GraphBackend

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Chain registry
_chain_registry: dict[str, ChainSpec] = {}


@dataclass
class ChainResult:
    """Result from a chain execution."""

    chain_name: str
    status: AgentStatus
    output: Any = None
    error: str | None = None
    agent_results: list[AgentResult] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_ms: float = 0.0
    execution_hash: ExecutionHash | None = None


def get_chain(name: str) -> ChainSpec | None:
    """Get a registered chain by name."""
    return _chain_registry.get(name)


async def _trace_safe(coro: Any) -> None:
    """Execute a tracing coroutine, catching and logging any errors.

    Tracing failures must never crash agent/chain execution.
    """
    try:
        await coro
    except Exception:
        logger.warning("Graph tracing operation failed in chain", exc_info=True)


def chain(
    agents: list[str] | None = None,
    *,
    name: str | None = None,
    pass_output: bool = True,
    stop_on_failure: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to define a sequential agent chain.

    Example:
        @agent(model="claude-sonnet-4-20250514")
        async def planner(task: str) -> str:
            '''Plan the task.'''
            pass

        @agent(model="claude-sonnet-4-20250514")
        async def executor(plan: str) -> str:
            '''Execute the plan.'''
            pass

        @chain(agents=["planner", "executor"])
        async def pipeline(task: str) -> str:
            '''Plan then execute.'''
            pass

        # Usage:
        result = await pipeline("Build a website")
        # Runs: planner("Build a website") -> executor(planner_output)

        # With graph tracing:
        from openagentflow.graph import SQLiteGraphBackend
        backend = SQLiteGraphBackend(":memory:")
        result = await pipeline("Build a website", graph_backend=backend)

    Args:
        agents: List of agent names to execute in order
        name: Chain name (defaults to function name)
        pass_output: Whether to pass output from one agent to the next
        stop_on_failure: Whether to stop if an agent fails
    """

    def decorator(fn: F) -> F:
        chain_name = name or fn.__name__
        agent_list = agents or []

        spec = ChainSpec(
            name=chain_name,
            agents=agent_list,
            pass_output=pass_output,
            stop_on_failure=stop_on_failure,
        )

        _chain_registry[chain_name] = spec

        @functools.wraps(fn)
        async def wrapper(
            *args: Any,
            parent_hash: str | list[str] | None = None,
            graph_backend: GraphBackend | None = None,
            **kwargs: Any,
        ) -> ChainResult:
            """Execute the chain of agents sequentially."""
            from openagentflow.core.agent import get_agent

            start_time = time.time()

            # Generate chain execution hash
            parent_hashes = (
                [parent_hash] if isinstance(parent_hash, str) else (parent_hash or [])
            )
            chain_hash = ExecutionHash.generate(parent_hashes)
            run_id = chain_hash.value

            # Chain vertex ID
            chain_vertex_id = f"chain-{chain_name}-{chain_hash.value[:12]}"

            # Record chain vertex (if tracing)
            if graph_backend is not None:
                await _trace_safe(
                    graph_backend.add_vertex(chain_vertex_id, "chain", {
                        "run_id": run_id,
                        "chain_name": chain_name,
                        "agents": json.dumps(agent_list),
                        "pass_output": pass_output,
                        "stop_on_failure": stop_on_failure,
                        "status": AgentStatus.THINKING.name,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    })
                )

            agent_results: list[AgentResult] = []
            total_tokens = 0
            total_cost = 0.0
            current_input = args[0] if args else kwargs.get("input", "")
            last_hash = chain_hash.value

            for step_index, agent_name in enumerate(agent_list):
                # Get agent from registry
                agent_spec = get_agent(agent_name)
                if agent_spec is None:
                    error_msg = f"Agent '{agent_name}' not found in chain '{chain_name}'"
                    if stop_on_failure:
                        duration_ms = (time.time() - start_time) * 1000
                        if graph_backend is not None:
                            await _trace_safe(
                                graph_backend.update_vertex(chain_vertex_id, {
                                    "status": AgentStatus.FAILED.name,
                                    "error": error_msg,
                                    "duration_ms": duration_ms,
                                    "completed_at": datetime.now(timezone.utc).isoformat(),
                                })
                            )
                        return ChainResult(
                            chain_name=chain_name,
                            status=AgentStatus.FAILED,
                            error=error_msg,
                            agent_results=agent_results,
                            total_tokens=total_tokens,
                            total_cost=total_cost,
                            duration_ms=duration_ms,
                            execution_hash=chain_hash,
                        )
                    continue

                # Get the agent wrapper (includes parent_hash support)
                if agent_spec.wrapper:
                    agent_func = agent_spec.wrapper
                else:
                    agent_func = agent_spec.func
                    if hasattr(agent_func, "_async_call"):
                        agent_func = agent_func._async_call

                # Execute agent with lineage
                try:
                    result = await agent_func(current_input, parent_hash=last_hash)
                    agent_results.append(result)
                    total_tokens += result.total_tokens
                    total_cost += result.total_cost

                    # Record agent vertex and STEP_N edge (if tracing)
                    if graph_backend is not None:
                        agent_vertex_id = f"agent-{agent_name}-{result.run_id[:12] if result.run_id else step_index}"
                        await _trace_safe(
                            graph_backend.add_vertex(agent_vertex_id, "agent", {
                                "run_id": run_id,
                                "agent_name": agent_name,
                                "step_index": step_index,
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
                                chain_vertex_id, agent_vertex_id,
                                f"STEP_{step_index}", {
                                    "step_index": step_index,
                                    "agent_name": agent_name,
                                })
                        )
                        # Also link sequential agents
                        if step_index > 0 and len(agent_results) >= 2:
                            prev_result = agent_results[-2]
                            prev_agent_name = agent_list[step_index - 1]
                            prev_vertex_id = f"agent-{prev_agent_name}-{prev_result.run_id[:12] if prev_result.run_id else step_index - 1}"
                            await _trace_safe(
                                graph_backend.add_edge(
                                    prev_vertex_id, agent_vertex_id,
                                    "NEXT", {
                                        "from_step": step_index - 1,
                                        "to_step": step_index,
                                    })
                            )

                    if result.status != AgentStatus.SUCCEEDED:
                        if stop_on_failure:
                            duration_ms = (time.time() - start_time) * 1000
                            if graph_backend is not None:
                                await _trace_safe(
                                    graph_backend.update_vertex(chain_vertex_id, {
                                        "status": result.status.name,
                                        "error": result.error or f"Agent '{agent_name}' did not succeed",
                                        "duration_ms": duration_ms,
                                        "total_tokens": total_tokens,
                                        "total_cost": total_cost,
                                        "completed_at": datetime.now(timezone.utc).isoformat(),
                                    })
                                )
                            return ChainResult(
                                chain_name=chain_name,
                                status=result.status,
                                error=result.error,
                                agent_results=agent_results,
                                total_tokens=total_tokens,
                                total_cost=total_cost,
                                duration_ms=duration_ms,
                                execution_hash=chain_hash,
                            )

                    # Pass output to next agent
                    if pass_output and result.output:
                        current_input = result.output

                    # Update lineage
                    if result.execution_hash:
                        last_hash = result.execution_hash.value

                except Exception as e:
                    if stop_on_failure:
                        duration_ms = (time.time() - start_time) * 1000
                        if graph_backend is not None:
                            await _trace_safe(
                                graph_backend.update_vertex(chain_vertex_id, {
                                    "status": AgentStatus.FAILED.name,
                                    "error": f"Agent '{agent_name}' failed: {e}",
                                    "duration_ms": duration_ms,
                                    "total_tokens": total_tokens,
                                    "total_cost": total_cost,
                                    "completed_at": datetime.now(timezone.utc).isoformat(),
                                })
                            )
                        return ChainResult(
                            chain_name=chain_name,
                            status=AgentStatus.FAILED,
                            error=f"Agent '{agent_name}' failed: {e}",
                            agent_results=agent_results,
                            total_tokens=total_tokens,
                            total_cost=total_cost,
                            duration_ms=duration_ms,
                            execution_hash=chain_hash,
                        )

            # All agents completed
            final_output = agent_results[-1].output if agent_results else None
            duration_ms = (time.time() - start_time) * 1000

            # Update chain vertex with success (if tracing)
            if graph_backend is not None:
                await _trace_safe(
                    graph_backend.update_vertex(chain_vertex_id, {
                        "status": AgentStatus.SUCCEEDED.name,
                        "output": str(final_output)[:2000] if final_output else "",
                        "duration_ms": duration_ms,
                        "total_tokens": total_tokens,
                        "total_cost": total_cost,
                        "steps_completed": len(agent_results),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                )

            return ChainResult(
                chain_name=chain_name,
                status=AgentStatus.SUCCEEDED,
                output=final_output,
                agent_results=agent_results,
                total_tokens=total_tokens,
                total_cost=total_cost,
                duration_ms=duration_ms,
                execution_hash=chain_hash,
            )

        wrapper._chain_spec = spec  # type: ignore
        return wrapper  # type: ignore

    return decorator


async def run_chain(
    chain_name: str,
    input_data: Any,
    parent_hash: str | list[str] | None = None,
    graph_backend: GraphBackend | None = None,
) -> ChainResult:
    """Run a registered chain by name.

    Args:
        chain_name: Name of the registered chain to run.
        input_data: Input data for the first agent in the chain.
        parent_hash: Parent execution hash(es) for lineage tracking.
        graph_backend: Optional graph backend for execution tracing.
    """
    spec = get_chain(chain_name)
    if spec is None:
        raise AgentError(f"Chain not found: {chain_name}")

    # Create and execute chain dynamically
    from openagentflow.core.agent import get_agent

    start_time = time.time()
    parent_hashes = [parent_hash] if isinstance(parent_hash, str) else (parent_hash or [])
    chain_hash = ExecutionHash.generate(parent_hashes)
    run_id = chain_hash.value

    # Chain vertex ID
    chain_vertex_id = f"chain-{chain_name}-{chain_hash.value[:12]}"

    # Record chain vertex (if tracing)
    if graph_backend is not None:
        await _trace_safe(
            graph_backend.add_vertex(chain_vertex_id, "chain", {
                "run_id": run_id,
                "chain_name": chain_name,
                "agents": json.dumps(spec.agents),
                "pass_output": spec.pass_output,
                "stop_on_failure": spec.stop_on_failure,
                "status": AgentStatus.THINKING.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })
        )

    agent_results: list[AgentResult] = []
    total_tokens = 0
    total_cost = 0.0
    current_input = input_data
    last_hash = chain_hash.value

    for step_index, agent_name in enumerate(spec.agents):
        agent_spec = get_agent(agent_name)
        if agent_spec is None:
            if spec.stop_on_failure:
                duration_ms = (time.time() - start_time) * 1000
                error_msg = f"Agent '{agent_name}' not found"
                if graph_backend is not None:
                    await _trace_safe(
                        graph_backend.update_vertex(chain_vertex_id, {
                            "status": AgentStatus.FAILED.name,
                            "error": error_msg,
                            "duration_ms": duration_ms,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                return ChainResult(
                    chain_name=chain_name,
                    status=AgentStatus.FAILED,
                    error=error_msg,
                    agent_results=agent_results,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    duration_ms=duration_ms,
                    execution_hash=chain_hash,
                )
            continue

        if agent_spec.wrapper:
            agent_func = agent_spec.wrapper
        else:
            agent_func = agent_spec.func
            if hasattr(agent_func, "_async_call"):
                agent_func = agent_func._async_call

        try:
            result = await agent_func(current_input, parent_hash=last_hash)
            agent_results.append(result)
            total_tokens += result.total_tokens
            total_cost += result.total_cost

            # Record agent vertex and STEP_N edge (if tracing)
            if graph_backend is not None:
                agent_vertex_id = f"agent-{agent_name}-{result.run_id[:12] if result.run_id else step_index}"
                await _trace_safe(
                    graph_backend.add_vertex(agent_vertex_id, "agent", {
                        "run_id": run_id,
                        "agent_name": agent_name,
                        "step_index": step_index,
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
                        chain_vertex_id, agent_vertex_id,
                        f"STEP_{step_index}", {
                            "step_index": step_index,
                            "agent_name": agent_name,
                        })
                )
                # Link sequential agents
                if step_index > 0 and len(agent_results) >= 2:
                    prev_result = agent_results[-2]
                    prev_agent_name = spec.agents[step_index - 1]
                    prev_vertex_id = f"agent-{prev_agent_name}-{prev_result.run_id[:12] if prev_result.run_id else step_index - 1}"
                    await _trace_safe(
                        graph_backend.add_edge(
                            prev_vertex_id, agent_vertex_id,
                            "NEXT", {
                                "from_step": step_index - 1,
                                "to_step": step_index,
                            })
                    )

            if result.status != AgentStatus.SUCCEEDED and spec.stop_on_failure:
                duration_ms = (time.time() - start_time) * 1000
                if graph_backend is not None:
                    await _trace_safe(
                        graph_backend.update_vertex(chain_vertex_id, {
                            "status": result.status.name,
                            "error": result.error or "",
                            "duration_ms": duration_ms,
                            "total_tokens": total_tokens,
                            "total_cost": total_cost,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                return ChainResult(
                    chain_name=chain_name,
                    status=result.status,
                    error=result.error,
                    agent_results=agent_results,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    duration_ms=duration_ms,
                    execution_hash=chain_hash,
                )

            if spec.pass_output and result.output:
                current_input = result.output
            if result.execution_hash:
                last_hash = result.execution_hash.value

        except Exception as e:
            if spec.stop_on_failure:
                duration_ms = (time.time() - start_time) * 1000
                if graph_backend is not None:
                    await _trace_safe(
                        graph_backend.update_vertex(chain_vertex_id, {
                            "status": AgentStatus.FAILED.name,
                            "error": str(e),
                            "duration_ms": duration_ms,
                            "total_tokens": total_tokens,
                            "total_cost": total_cost,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                return ChainResult(
                    chain_name=chain_name,
                    status=AgentStatus.FAILED,
                    error=str(e),
                    agent_results=agent_results,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    duration_ms=duration_ms,
                    execution_hash=chain_hash,
                )

    duration_ms = (time.time() - start_time) * 1000

    # Update chain vertex with success (if tracing)
    if graph_backend is not None:
        await _trace_safe(
            graph_backend.update_vertex(chain_vertex_id, {
                "status": AgentStatus.SUCCEEDED.name,
                "output": str(agent_results[-1].output)[:2000] if agent_results else "",
                "duration_ms": duration_ms,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "steps_completed": len(agent_results),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        )

    return ChainResult(
        chain_name=chain_name,
        status=AgentStatus.SUCCEEDED,
        output=agent_results[-1].output if agent_results else None,
        agent_results=agent_results,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_ms=duration_ms,
        execution_hash=chain_hash,
    )
