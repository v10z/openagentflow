"""Tests for the @tool decorator."""

from __future__ import annotations

import pytest

from openagentflow import tool, get_tool, get_all_tools, execute_tool


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_registration(self):
        """Test that decorated tools are registered."""
        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        spec = get_tool("add_numbers")
        assert spec is not None
        assert spec.name == "add_numbers"
        assert "Add two numbers" in spec.description

    def test_tool_input_schema_generation(self):
        """Test that input schema is auto-generated from type hints."""
        @tool
        def search(query: str, limit: int = 10) -> list:
            """Search for items."""
            return []

        spec = get_tool("search")
        assert spec.input_schema is not None
        assert "properties" in spec.input_schema
        assert "query" in spec.input_schema["properties"]
        assert spec.input_schema["properties"]["query"]["type"] == "string"

    def test_tool_execution(self):
        """Test tool execution."""
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        # Direct call still works
        result = multiply(3, 4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_execute_tool_async(self):
        """Test async tool execution via registry."""
        from openagentflow.core.types import ToolCall

        @tool
        def divide(a: float, b: float) -> float:
            """Divide two numbers."""
            return a / b

        tool_call = ToolCall(id="test", tool_name="divide", arguments={"a": 10.0, "b": 2.0})
        result = await execute_tool(tool_call)
        assert result.success is True
        assert result.output == 5.0

    def test_tool_custom_name(self):
        """Test tool with custom name."""
        @tool(name="custom_calculator")
        def calc(x: int) -> int:
            """Calculate."""
            return x * 2

        spec = get_tool("custom_calculator")
        assert spec is not None
        assert spec.name == "custom_calculator"

    def test_tool_with_timeout(self):
        """Test tool with timeout setting."""
        @tool(timeout=60)
        def slow_task(data: str) -> str:
            """A slow task."""
            return data

        spec = get_tool("slow_task")
        assert spec.timeout_seconds == 60

    def test_tool_with_retry(self):
        """Test tool with retry setting."""
        @tool(retry=5)
        def flaky_task(data: str) -> str:
            """A flaky task."""
            return data

        spec = get_tool("flaky_task")
        assert spec.max_retries == 5

    def test_tool_has_spec_attribute(self):
        """Test that decorated function has _tool_spec attribute."""
        @tool
        def test_func(x: int) -> int:
            """Test function."""
            return x

        assert hasattr(test_func, "_tool_spec")
        assert test_func._tool_spec.name == "test_func"


class TestGetAllTools:
    """Tests for get_all_tools function."""

    def test_get_all_returns_dict(self):
        """Test that get_all_tools returns a dict."""
        tools = get_all_tools()
        assert isinstance(tools, dict)

    def test_get_all_includes_registered(self):
        """Test that get_all_tools includes registered tools."""
        @tool
        def listed_tool(x: int) -> int:
            """Listed tool."""
            return x

        tools = get_all_tools()
        assert "listed_tool" in tools
