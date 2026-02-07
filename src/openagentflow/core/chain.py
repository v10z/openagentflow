"""
@chain decorator for sequential agent pipelines.

Chains execute agents in order, passing output from one to the next.
Each agent's output becomes the input for the next agent.
Lineage is tracked through ExecutionHash.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from openagentflow.core.types import (
    AgentResult,
    AgentStatus,
    ChainSpec,
    ExecutionHash,
)
from openagentflow.exceptions import AgentError

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

            agent_results: list[AgentResult] = []
            total_tokens = 0
            total_cost = 0.0
            current_input = args[0] if args else kwargs.get("input", "")
            last_hash = chain_hash.value

            for agent_name in agent_list:
                # Get agent from registry
                agent_spec = get_agent(agent_name)
                if agent_spec is None:
                    error_msg = f"Agent '{agent_name}' not found in chain '{chain_name}'"
                    if stop_on_failure:
                        return ChainResult(
                            chain_name=chain_name,
                            status=AgentStatus.FAILED,
                            error=error_msg,
                            agent_results=agent_results,
                            total_tokens=total_tokens,
                            total_cost=total_cost,
                            duration_ms=(time.time() - start_time) * 1000,
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

                    if result.status != AgentStatus.SUCCEEDED:
                        if stop_on_failure:
                            return ChainResult(
                                chain_name=chain_name,
                                status=result.status,
                                error=result.error,
                                agent_results=agent_results,
                                total_tokens=total_tokens,
                                total_cost=total_cost,
                                duration_ms=(time.time() - start_time) * 1000,
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
                        return ChainResult(
                            chain_name=chain_name,
                            status=AgentStatus.FAILED,
                            error=f"Agent '{agent_name}' failed: {e}",
                            agent_results=agent_results,
                            total_tokens=total_tokens,
                            total_cost=total_cost,
                            duration_ms=(time.time() - start_time) * 1000,
                            execution_hash=chain_hash,
                        )

            # All agents completed
            final_output = agent_results[-1].output if agent_results else None

            return ChainResult(
                chain_name=chain_name,
                status=AgentStatus.SUCCEEDED,
                output=final_output,
                agent_results=agent_results,
                total_tokens=total_tokens,
                total_cost=total_cost,
                duration_ms=(time.time() - start_time) * 1000,
                execution_hash=chain_hash,
            )

        wrapper._chain_spec = spec  # type: ignore
        return wrapper  # type: ignore

    return decorator


async def run_chain(
    chain_name: str,
    input_data: Any,
    parent_hash: str | list[str] | None = None,
) -> ChainResult:
    """Run a registered chain by name."""
    spec = get_chain(chain_name)
    if spec is None:
        raise AgentError(f"Chain not found: {chain_name}")

    # Create and execute chain dynamically
    from openagentflow.core.agent import get_agent

    start_time = time.time()
    parent_hashes = [parent_hash] if isinstance(parent_hash, str) else (parent_hash or [])
    chain_hash = ExecutionHash.generate(parent_hashes)

    agent_results: list[AgentResult] = []
    total_tokens = 0
    total_cost = 0.0
    current_input = input_data
    last_hash = chain_hash.value

    for agent_name in spec.agents:
        agent_spec = get_agent(agent_name)
        if agent_spec is None:
            if spec.stop_on_failure:
                return ChainResult(
                    chain_name=chain_name,
                    status=AgentStatus.FAILED,
                    error=f"Agent '{agent_name}' not found",
                    agent_results=agent_results,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    duration_ms=(time.time() - start_time) * 1000,
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

            if result.status != AgentStatus.SUCCEEDED and spec.stop_on_failure:
                return ChainResult(
                    chain_name=chain_name,
                    status=result.status,
                    error=result.error,
                    agent_results=agent_results,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    duration_ms=(time.time() - start_time) * 1000,
                    execution_hash=chain_hash,
                )

            if spec.pass_output and result.output:
                current_input = result.output
            if result.execution_hash:
                last_hash = result.execution_hash.value

        except Exception as e:
            if spec.stop_on_failure:
                return ChainResult(
                    chain_name=chain_name,
                    status=AgentStatus.FAILED,
                    error=str(e),
                    agent_results=agent_results,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    duration_ms=(time.time() - start_time) * 1000,
                    execution_hash=chain_hash,
                )

    return ChainResult(
        chain_name=chain_name,
        status=AgentStatus.SUCCEEDED,
        output=agent_results[-1].output if agent_results else None,
        agent_results=agent_results,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_ms=(time.time() - start_time) * 1000,
        execution_hash=chain_hash,
    )
