"""Tests for the agent executor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openagentflow.core.types import (
    AgentSpec,
    AgentStatus,
    LLMProvider,
    Message,
    ModelConfig,
    ReasoningStrategy,
    ToolCall,
    ToolSpec,
)
from openagentflow.llm.base import LLMResponse


class TestAgentExecutor:
    """Tests for AgentExecutor."""

    def _create_spec(
        self,
        name: str = "test_agent",
        tools: list[ToolSpec] | None = None,
        max_iterations: int = 5,
    ) -> AgentSpec:
        """Create a test agent spec."""
        return AgentSpec(
            name=name,
            description="Test agent",
            func=lambda x: x,
            model=ModelConfig(provider=LLMProvider.ANTHROPIC, model_id="claude-sonnet-4-20250514"),
            tools=tools or [],
            reasoning_strategy=ReasoningStrategy.REACT,
            max_iterations=max_iterations,
            timeout_seconds=30.0,
        )

    @pytest.mark.asyncio
    async def test_simple_run_no_tools(self):
        """Test simple run without tool calls."""
        from openagentflow.runtime.executor import AgentExecutor

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Hello! How can I help?",
                tool_calls=[],
                stop_reason="end_turn",
                input_tokens=10,
                output_tokens=8,
                model_id="claude-sonnet-4-20250514",
            )
        )
        mock_provider.estimate_cost = MagicMock(return_value=0.0001)

        executor = AgentExecutor(mock_provider)
        spec = self._create_spec()

        result = await executor.run(spec, {"query": "Hi!"})

        assert result.status == AgentStatus.SUCCEEDED
        assert result.output == "Hello! How can I help?"
        assert result.agent_name == "test_agent"
        assert result.total_tokens == 18

    @pytest.mark.asyncio
    async def test_run_with_tool_call(self):
        """Test run with a tool call."""
        from openagentflow.runtime.executor import AgentExecutor
        from openagentflow.core.tool import tool, _tool_registry

        # Clear tool registry and create a test tool
        _tool_registry.clear()

        @tool
        def get_weather(city: str) -> dict:
            """Get weather for a city."""
            return {"city": city, "temp": 72, "condition": "sunny"}

        tool_spec = get_weather._tool_spec

        # Mock provider - first call returns tool use, second returns final response
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="Let me check the weather for Tokyo.",
                    tool_calls=[
                        ToolCall(id="call_1", tool_name="get_weather", arguments={"city": "Tokyo"}),
                    ],
                    stop_reason="tool_use",
                    input_tokens=20,
                    output_tokens=15,
                ),
                LLMResponse(
                    content="The weather in Tokyo is 72°F and sunny!",
                    tool_calls=[],
                    stop_reason="end_turn",
                    input_tokens=50,
                    output_tokens=12,
                ),
            ]
        )
        mock_provider.estimate_cost = MagicMock(return_value=0.0001)

        executor = AgentExecutor(mock_provider)
        spec = self._create_spec(tools=[tool_spec])

        result = await executor.run(spec, {"query": "What's the weather in Tokyo?"})

        assert result.status == AgentStatus.SUCCEEDED
        assert "72" in result.output or "sunny" in result.output
        assert mock_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self):
        """Test that max iterations raises error."""
        from openagentflow.runtime.executor import AgentExecutor

        # Mock provider that always returns tool calls
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Calling tool again...",
                tool_calls=[
                    ToolCall(id="call_1", tool_name="unknown_tool", arguments={}),
                ],
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=5,
            )
        )
        mock_provider.estimate_cost = MagicMock(return_value=0.0001)

        executor = AgentExecutor(mock_provider)
        spec = self._create_spec(max_iterations=3)

        result = await executor.run(spec, {"query": "Loop forever"})

        assert result.status == AgentStatus.FAILED
        assert "max iterations" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test that timeout is handled between iterations."""
        from openagentflow.runtime.executor import AgentExecutor
        import asyncio

        call_count = 0

        # Mock provider that returns tool calls, requiring multiple iterations
        # Each call takes 0.3s, and we have a 0.25s timeout
        # First call completes, but before second iteration starts, timeout triggers
        async def multi_iteration_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Add delay that will exceed timeout before next iteration check
            await asyncio.sleep(0.3)
            if call_count == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_1", tool_name="unknown_tool", arguments={}),
                    ],
                    stop_reason="tool_use",
                    input_tokens=10,
                    output_tokens=5,
                )
            else:
                return LLMResponse(content="Done", input_tokens=10, output_tokens=5)

        mock_provider = MagicMock()
        mock_provider.generate = multi_iteration_generate
        mock_provider.estimate_cost = MagicMock(return_value=0.0001)

        executor = AgentExecutor(mock_provider)
        spec = AgentSpec(
            name="timeout_agent",
            description="Timeout agent",
            func=lambda x: x,
            model=ModelConfig(provider=LLMProvider.ANTHROPIC, model_id="claude-sonnet-4-20250514"),
            tools=[],
            reasoning_strategy=ReasoningStrategy.REACT,
            max_iterations=5,
            timeout_seconds=0.25,  # Will timeout before second iteration can start
        )

        result = await executor.run(spec, {"query": "Wait"})

        # Should timeout between iterations (after first LLM call completes,
        # before second iteration begins, timeout is checked)
        assert result.status == AgentStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test handling of unknown tool."""
        from openagentflow.runtime.executor import AgentExecutor

        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_1", tool_name="nonexistent_tool", arguments={}),
                    ],
                    stop_reason="tool_use",
                    input_tokens=10,
                    output_tokens=5,
                ),
                LLMResponse(
                    content="Tool not found, but I'll continue.",
                    tool_calls=[],
                    stop_reason="end_turn",
                    input_tokens=20,
                    output_tokens=10,
                ),
            ]
        )
        mock_provider.estimate_cost = MagicMock(return_value=0.0001)

        executor = AgentExecutor(mock_provider)
        spec = self._create_spec()

        result = await executor.run(spec, {"query": "Use unknown tool"})

        # Should still complete, just with error in tool result
        assert result.status == AgentStatus.SUCCEEDED

    def test_format_input_single_param(self):
        """Test formatting single parameter input."""
        from openagentflow.runtime.executor import AgentExecutor

        executor = AgentExecutor(MagicMock())
        formatted = executor._format_input({"query": "Hello world"})
        assert formatted == "Hello world"

    def test_format_input_multiple_params(self):
        """Test formatting multiple parameter input."""
        from openagentflow.runtime.executor import AgentExecutor

        executor = AgentExecutor(MagicMock())
        formatted = executor._format_input({"city": "Tokyo", "date": "today"})
        assert "city: Tokyo" in formatted
        assert "date: today" in formatted


class TestExecutionHash:
    """Tests for execution hash generation."""

    def test_hash_without_parents(self):
        """Test hash generation without parents."""
        from openagentflow.core.types import ExecutionHash

        hash1 = ExecutionHash.generate()
        hash2 = ExecutionHash.generate()

        assert hash1.value != hash2.value  # Should be unique
        assert len(hash1.value) == 32  # MD5 hex length
        assert hash1.parents == ()

    def test_hash_with_parents(self):
        """Test hash generation with parent hashes."""
        from openagentflow.core.types import ExecutionHash

        parent = ExecutionHash.generate()
        child = ExecutionHash.generate([parent.value])

        assert child.parents == (parent.value,)
        assert child.value != parent.value
