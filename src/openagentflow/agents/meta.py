"""
Meta-Agent - Creates new tools on the fly using the ToolFactory.

This module defines a meta-agent that can dynamically create, test, list,
and remove tools at runtime. It uses the ToolFactory and Sandbox to ensure
all generated code is validated and executed safely.

The meta-agent is useful when an agent workflow needs capabilities that
were not anticipated at design time. Instead of stopping and waiting for
a developer to add a new tool, the meta-agent can create one on the spot.

Usage:
    from openagentflow.agents.meta import tool_maker, create_tool, list_tools

    # Use the meta-agent (requires LLM)
    result = await tool_maker("Create a tool that calculates compound interest")

    # Or use the tools directly
    create_tool(
        name="compound_interest",
        description="Calculate compound interest",
        source_code=(
            "def compound_interest(principal: float, rate: float, "
            "years: int, compounds_per_year: int = 12) -> float:\\n"
            "    return principal * (1 + rate / compounds_per_year) "
            "** (compounds_per_year * years)\\n"
        ),
    )
"""

from __future__ import annotations

import json

from openagentflow.core.agent import agent
from openagentflow.core.tool import get_all_tools, tool
from openagentflow.meta.sandbox import Sandbox
from openagentflow.meta.tool_factory import ToolFactory

# Shared factory instance for this module's tools.
# Each use of the meta-agent tools operates on the same factory,
# so dynamically created tools persist across calls within the same process.
_factory = ToolFactory()


@tool
def create_tool(name: str, description: str, source_code: str) -> dict:
    """Create a new tool from Python source code.

    The source_code should define a Python function with type hints.
    The function will be validated for safety (no I/O, no network, no dangerous
    operations) and then registered as an available tool.

    Args:
        name: Unique name for the tool.
        description: Human-readable description of what the tool does.
        source_code: Python source code defining a function with type hints.
            Example: 'def add(a: int, b: int) -> int:\\n    return a + b'

    Returns:
        A dict with 'success' (bool), and either 'tool_name' and 'description'
        on success, or 'error' on failure.
    """
    try:
        spec = _factory.create_tool(name, description, source_code)
        return {
            "success": True,
            "tool_name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def list_tools() -> list[dict]:
    """List all available tools (both built-in and dynamically created).

    Returns:
        A list of dicts, each containing 'name', 'description', and
        'is_dynamic' (bool indicating if it was created at runtime).
    """
    all_tools = get_all_tools()
    dynamic_names = set(_factory._dynamic_tools.keys())

    result = []
    for tool_name, spec in all_tools.items():
        result.append({
            "name": tool_name,
            "description": spec.description,
            "is_dynamic": tool_name in dynamic_names,
        })

    return result


@tool
def test_tool(tool_name: str, test_input: str) -> dict:
    """Test a dynamically created tool with JSON-formatted input.

    Executes the tool in a sandbox with the provided input arguments
    and returns the result. Use this to verify a newly created tool
    works correctly before relying on it.

    Args:
        tool_name: Name of the tool to test.
        test_input: JSON string of input arguments.
            Example: '{"a": 3, "b": 4}'

    Returns:
        A dict with 'success' (bool), and either 'result' on success
        or 'error' on failure.
    """
    try:
        args = json.loads(test_input)
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON in test_input: {e}",
        }

    if not isinstance(args, dict):
        return {
            "success": False,
            "error": (
                "test_input must be a JSON object (dict), "
                f"got {type(args).__name__}"
            ),
        }

    try:
        result = _factory.test_tool(tool_name, args)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def remove_tool(tool_name: str) -> dict:
    """Remove a dynamically created tool.

    Only tools created at runtime via the meta-agent can be removed.
    Built-in tools cannot be removed through this interface.

    Args:
        tool_name: Name of the tool to remove.

    Returns:
        A dict with 'success' (bool) and 'tool_name'.
    """
    success = _factory.remove_tool(tool_name)
    if success:
        return {"success": True, "tool_name": tool_name}
    else:
        return {
            "success": False,
            "tool_name": tool_name,
            "error": (
                f"Tool '{tool_name}' not found in dynamic tools. "
                f"Only dynamically created tools can be removed."
            ),
        }


@tool
def get_tool_source(tool_name: str) -> dict:
    """Get the source code of a dynamically created tool.

    Args:
        tool_name: Name of the tool whose source code to retrieve.

    Returns:
        A dict with 'success' (bool), and either 'source_code' on success
        or 'error' on failure.
    """
    source = _factory.get_tool_source(tool_name)
    if source is not None:
        return {"success": True, "tool_name": tool_name, "source_code": source}
    else:
        return {
            "success": False,
            "error": (
                f"Tool '{tool_name}' not found in dynamic tools. "
                f"Available dynamic tools: "
                f"{[s.name for s in _factory.list_dynamic_tools()]}"
            ),
        }


@agent(
    model="claude-sonnet-4-20250514",
    tools=[create_tool, list_tools, test_tool, remove_tool, get_tool_source],
    system_prompt="""You are a meta-agent that creates new tools on the fly.

When asked to create a tool, generate clean Python source code with proper type hints.
Always test newly created tools before reporting success.

Follow these rules for generated code:
- Only use Python stdlib modules (math, re, json, collections, itertools,
  functools, string, textwrap, unicodedata, datetime, decimal, fractions,
  random, statistics, operator, copy, dataclasses, enum, typing, abc,
  bisect, heapq)
- Include proper type hints on all parameters and return values
- Include a docstring describing what the tool does
- Keep functions pure (no side effects, no I/O, no network)
- Handle edge cases gracefully (empty inputs, zero division, etc.)
- Use descriptive variable names
- Prefer simple, readable implementations over clever ones

Workflow for creating a tool:
1. Generate the Python source code for the requested functionality
2. Call create_tool with the name, description, and source code
3. Call test_tool with representative test inputs to verify correctness
4. If the test fails, remove the tool and try again with fixed code
5. Report the result to the user

When listing tools, use list_tools to show what is available.
When asked to inspect a tool, use get_tool_source to show its code.""",
)
async def tool_maker(request: str) -> str:
    """Meta-agent that creates new tools on the fly based on natural language requests.

    This agent can:
    - Create new tools from descriptions
    - Test tools to verify correctness
    - List all available tools
    - Remove tools that are no longer needed
    - Show source code of dynamic tools

    Args:
        request: Natural language description of what tool to create or action to take.

    Returns:
        A string describing the result of the operation.
    """
    pass  # ReAct loop handles execution


def get_factory() -> ToolFactory:
    """Get the shared ToolFactory instance used by the meta-agent tools.

    This is useful for programmatic access to the same factory that
    the meta-agent tools operate on.

    Returns:
        The module-level ToolFactory instance.
    """
    return _factory
