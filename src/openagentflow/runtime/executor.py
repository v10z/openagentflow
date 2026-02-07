"""Agent executor implementing the ReAct reasoning loop.

The executor runs the core agent loop:
1. Send messages to LLM
2. If LLM returns tool calls, execute them
3. Add tool results to messages
4. Repeat until done or max iterations reached

Supports optional graph tracing -- when a graph_backend is provided,
the executor records agent and tool vertices/edges into the execution DAG.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from openagentflow.core.tool import execute_tool, get_tool
from openagentflow.core.types import (
    AgentResult,
    AgentSpec,
    AgentState,
    AgentStatus,
    ExecutionHash,
    Message,
    ModelConfig,
    ToolCall,
    ToolResult,
)
from openagentflow.exceptions import AgentError, TimeoutError, ToolError
from openagentflow.llm.base import BaseLLMProvider

if TYPE_CHECKING:
    from openagentflow.graph.base import GraphBackend

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for a single agent execution."""

    agent_spec: AgentSpec
    provider: BaseLLMProvider
    execution_hash: ExecutionHash
    start_time: float = field(default_factory=time.time)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0


class AgentExecutor:
    """Executes agents using the ReAct (Reason + Act) loop.

    The executor handles:
    - LLM calls with retry and timeout
    - Tool execution
    - State management
    - Token/cost tracking
    - Optional graph tracing (execution DAG recording)

    Example:
        executor = AgentExecutor(provider)
        result = await executor.run(
            spec=agent_spec,
            input_data={"query": "What's the weather?"},
            parent_hash=None,
        )

        # With graph tracing:
        from openagentflow.graph import SQLiteGraphBackend
        backend = SQLiteGraphBackend(":memory:")
        executor = AgentExecutor(provider, graph_backend=backend)
        result = await executor.run(spec=agent_spec, input_data={"query": "Hello"})
        trace = await backend.get_full_trace(result.run_id)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        graph_backend: GraphBackend | None = None,
    ):
        """Initialize the executor.

        Args:
            provider: The LLM provider to use.
            graph_backend: Optional graph backend for execution tracing.
                If None, tracing is disabled (zero overhead).
        """
        self.provider = provider
        self._graph: GraphBackend | None = graph_backend

    async def _trace_safe(self, coro: Any) -> None:
        """Execute a tracing coroutine, catching and logging any errors.

        Tracing failures must never crash agent execution.
        """
        try:
            await coro
        except Exception:
            logger.warning("Graph tracing operation failed", exc_info=True)

    async def run(
        self,
        spec: AgentSpec,
        input_data: dict[str, Any],
        parent_hash: str | list[str] | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        """Run an agent with the given input.

        Args:
            spec: The agent specification.
            input_data: Input parameters for the agent.
            parent_hash: Parent execution hash(es) for lineage tracking.
            run_id: Optional run ID for grouping traces. If not provided,
                the execution hash value is used.

        Returns:
            AgentResult with output, trace, and metrics.
        """
        # Generate execution hash for lineage
        parent_hashes = (
            [parent_hash] if isinstance(parent_hash, str) else parent_hash
        )
        execution_hash = ExecutionHash.generate(parent_hashes)

        # Determine run_id for trace grouping
        effective_run_id = run_id or execution_hash.value

        # Create execution context
        ctx = ExecutionContext(
            agent_spec=spec,
            provider=self.provider,
            execution_hash=execution_hash,
        )

        # Generate agent vertex ID
        agent_vertex_id = f"agent-{spec.name}-{execution_hash.value[:12]}"

        # Initialize state
        state = AgentState(
            status=AgentStatus.THINKING,
            iteration=0,
            messages=[],
            tool_calls_history=[],
            accumulated_tokens=0,
        )

        # Build initial user message from input
        user_content = self._format_input(input_data)
        state.messages.append(Message(role="user", content=user_content))

        # Record agent vertex at start (if tracing)
        if self._graph is not None:
            # Build a content hash from inputs + timestamp for deterministic ID
            hash_content = json.dumps(input_data, sort_keys=True, default=str)
            hash_content += str(ctx.start_time)
            content_hash = hashlib.md5(
                hash_content.encode(), usedforsecurity=False
            ).hexdigest()

            agent_properties: dict[str, Any] = {
                "run_id": effective_run_id,
                "agent_name": spec.name,
                "content_hash": content_hash,
                "input_data": json.dumps(input_data, default=str),
                "model": spec.model.model_id,
                "provider": spec.model.provider.value,
                "system_prompt": spec.system_prompt or "",
                "status": AgentStatus.THINKING.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "parent_hashes": json.dumps(parent_hashes or []),
            }
            await self._trace_safe(
                self._graph.add_vertex(agent_vertex_id, "agent", agent_properties)
            )

        try:
            # Run the ReAct loop
            output = await self._react_loop(
                ctx, state, agent_vertex_id, effective_run_id
            )

            duration_ms = (time.time() - ctx.start_time) * 1000

            # Update agent vertex with final output (if tracing)
            if self._graph is not None:
                await self._trace_safe(
                    self._graph.update_vertex(agent_vertex_id, {
                        "status": AgentStatus.SUCCEEDED.name,
                        "output": str(output)[:2000],  # Truncate large outputs
                        "total_tokens": ctx.total_input_tokens + ctx.total_output_tokens,
                        "total_cost": ctx.total_cost,
                        "duration_ms": duration_ms,
                        "iterations": state.iteration,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                )

            return AgentResult(
                agent_name=spec.name,
                run_id=execution_hash.value,
                status=AgentStatus.SUCCEEDED,
                output=output,
                trace_id=execution_hash.value,
                total_tokens=ctx.total_input_tokens + ctx.total_output_tokens,
                total_cost=ctx.total_cost,
                duration_ms=duration_ms,
                execution_hash=execution_hash,
            )

        except TimeoutError as e:
            duration_ms = (time.time() - ctx.start_time) * 1000

            if self._graph is not None:
                await self._trace_safe(
                    self._graph.update_vertex(agent_vertex_id, {
                        "status": AgentStatus.TIMEOUT.name,
                        "error": str(e),
                        "total_tokens": ctx.total_input_tokens + ctx.total_output_tokens,
                        "total_cost": ctx.total_cost,
                        "duration_ms": duration_ms,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                )

            return AgentResult(
                agent_name=spec.name,
                run_id=execution_hash.value,
                status=AgentStatus.TIMEOUT,
                output=None,
                error=str(e),
                trace_id=execution_hash.value,
                total_tokens=ctx.total_input_tokens + ctx.total_output_tokens,
                total_cost=ctx.total_cost,
                duration_ms=duration_ms,
                execution_hash=execution_hash,
            )

        except Exception as e:
            duration_ms = (time.time() - ctx.start_time) * 1000

            if self._graph is not None:
                await self._trace_safe(
                    self._graph.update_vertex(agent_vertex_id, {
                        "status": AgentStatus.FAILED.name,
                        "error": str(e),
                        "total_tokens": ctx.total_input_tokens + ctx.total_output_tokens,
                        "total_cost": ctx.total_cost,
                        "duration_ms": duration_ms,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                )

            return AgentResult(
                agent_name=spec.name,
                run_id=execution_hash.value,
                status=AgentStatus.FAILED,
                output=None,
                error=str(e),
                trace_id=execution_hash.value,
                total_tokens=ctx.total_input_tokens + ctx.total_output_tokens,
                total_cost=ctx.total_cost,
                duration_ms=duration_ms,
                execution_hash=execution_hash,
            )

    async def _react_loop(
        self,
        ctx: ExecutionContext,
        state: AgentState,
        agent_vertex_id: str,
        run_id: str,
    ) -> str:
        """Run the ReAct reasoning loop.

        1. Call LLM with current messages
        2. If tool calls, execute them and add results
        3. Repeat until model says done or max iterations
        """
        spec = ctx.agent_spec
        model_config = spec.model

        while state.iteration < spec.max_iterations:
            state.iteration += 1
            state.status = AgentStatus.THINKING

            # Check timeout
            elapsed = time.time() - ctx.start_time
            if elapsed > spec.timeout_seconds:
                raise TimeoutError(
                    f"Agent timed out after {elapsed:.1f}s "
                    f"(limit: {spec.timeout_seconds}s)"
                )

            # Call LLM
            response = await self.provider.generate(
                messages=state.messages,
                config=model_config,
                tools=spec.tools if spec.tools else None,
                system_prompt=spec.system_prompt,
            )

            # Track tokens and cost
            ctx.total_input_tokens += response.input_tokens
            ctx.total_output_tokens += response.output_tokens
            ctx.total_cost += self.provider.estimate_cost(
                response.input_tokens,
                response.output_tokens,
                model_config.model_id,
            )
            state.accumulated_tokens = ctx.total_input_tokens + ctx.total_output_tokens

            # Add assistant response to messages
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            state.messages.append(assistant_msg)

            # If no tool calls, we're done
            if not response.has_tool_calls:
                return response.content

            # Execute tool calls
            state.status = AgentStatus.TOOL_CALLING
            tool_results = await self._execute_tools(
                response.tool_calls, spec, agent_vertex_id, run_id, state.iteration
            )

            # Add tool results to messages
            for result in tool_results:
                state.tool_calls_history.append(result.tool_call)
                result_msg = Message(
                    role="tool",
                    content=self._format_tool_result(result),
                    tool_call_id=result.tool_call.id,
                )
                state.messages.append(result_msg)

        # Max iterations reached
        raise AgentError(
            f"Agent reached max iterations ({spec.max_iterations}) without completing"
        )

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        spec: AgentSpec,
        agent_vertex_id: str,
        run_id: str,
        iteration: int,
    ) -> list[ToolResult]:
        """Execute tool calls and return results.

        When graph tracing is enabled, records each tool call as a vertex
        linked to the agent vertex via a "CALLED" edge.
        """
        results = []

        for tc in tool_calls:
            tool_start_time = time.time()

            # Find the tool
            tool_spec = None
            for t in spec.tools or []:
                if t.name == tc.tool_name:
                    tool_spec = t
                    break

            if tool_spec is None:
                # Tool not found
                result = ToolResult(
                    tool_call=tc,
                    output=None,
                    error=f"Tool '{tc.tool_name}' not found",
                    success=False,
                )
                results.append(result)

                # Record failed tool call in graph
                if self._graph is not None:
                    tool_vertex_id = f"tool-{tc.tool_name}-{tc.id}"
                    await self._trace_safe(
                        self._graph.add_vertex(tool_vertex_id, "tool", {
                            "run_id": run_id,
                            "tool_name": tc.tool_name,
                            "tool_call_id": tc.id,
                            "arguments": json.dumps(tc.arguments, default=str),
                            "success": False,
                            "error": f"Tool '{tc.tool_name}' not found",
                            "duration_ms": (time.time() - tool_start_time) * 1000,
                            "iteration": iteration,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                    await self._trace_safe(
                        self._graph.add_edge(agent_vertex_id, tool_vertex_id, "CALLED", {
                            "iteration": iteration,
                            "tool_call_id": tc.id,
                        })
                    )
                continue

            # Execute the tool using the registry
            result = await execute_tool(tc, spec.tools)
            results.append(result)

            tool_duration_ms = (time.time() - tool_start_time) * 1000

            # Record tool call in graph (if tracing)
            if self._graph is not None:
                tool_vertex_id = f"tool-{tc.tool_name}-{tc.id}"
                tool_properties: dict[str, Any] = {
                    "run_id": run_id,
                    "tool_name": tc.tool_name,
                    "tool_call_id": tc.id,
                    "arguments": json.dumps(tc.arguments, default=str),
                    "success": result.success,
                    "duration_ms": tool_duration_ms,
                    "iteration": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if result.success:
                    # Truncate large outputs
                    output_str = str(result.output)
                    tool_properties["output"] = output_str[:2000]
                else:
                    tool_properties["error"] = result.error or ""

                await self._trace_safe(
                    self._graph.add_vertex(tool_vertex_id, "tool", tool_properties)
                )
                await self._trace_safe(
                    self._graph.add_edge(agent_vertex_id, tool_vertex_id, "CALLED", {
                        "iteration": iteration,
                        "tool_call_id": tc.id,
                        "tool_name": tc.tool_name,
                    })
                )

        return results

    def _format_input(self, input_data: dict[str, Any]) -> str:
        """Format input data as a user message."""
        if len(input_data) == 1:
            # Single parameter - just use the value
            return str(list(input_data.values())[0])

        # Multiple parameters - format as key: value
        parts = [f"{k}: {v}" for k, v in input_data.items()]
        return "\n".join(parts)

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format a tool result for the LLM."""
        if result.success:
            # Convert output to string
            output = result.output
            if isinstance(output, dict):
                import json

                return json.dumps(output, indent=2)
            return str(output)
        else:
            return f"Error: {result.error}"


async def run_agent_with_provider(
    spec: AgentSpec,
    input_data: dict[str, Any],
    provider: BaseLLMProvider,
    parent_hash: str | list[str] | None = None,
    graph_backend: GraphBackend | None = None,
) -> AgentResult:
    """Convenience function to run an agent with a provider.

    Example:
        from openagentflow.llm.providers import AnthropicProvider
        from openagentflow.graph import SQLiteGraphBackend

        provider = AnthropicProvider()
        backend = SQLiteGraphBackend(":memory:")
        result = await run_agent_with_provider(
            spec=my_agent_spec,
            input_data={"query": "Hello"},
            provider=provider,
            graph_backend=backend,
        )
    """
    executor = AgentExecutor(provider, graph_backend=graph_backend)
    return await executor.run(spec, input_data, parent_hash)
