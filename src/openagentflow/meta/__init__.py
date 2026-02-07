"""
OpenAgentFlow Meta - JIT Meta-Agent System

Provides runtime tool creation capabilities:

- Sandbox: Restricted execution environment for dynamically created tools.
  Validates code via AST analysis, blocks dangerous operations, and enforces
  execution timeouts.

- ToolFactory: Creates new tools at runtime from Python source code or
  natural language descriptions (via LLM). Validates all code through the
  Sandbox before execution.

- Meta-agent tools (in openagentflow.agents.meta): A meta-agent that uses
  the ToolFactory to create, test, list, and remove tools on the fly.

Usage:
    from openagentflow.meta import Sandbox, ToolFactory

    # Create a sandbox and factory
    sandbox = Sandbox()
    factory = ToolFactory(sandbox=sandbox)

    # Create a tool from source code
    spec = factory.create_tool(
        name="fibonacci",
        description="Compute the nth Fibonacci number",
        source_code=(
            "def fibonacci(n: int) -> int:\\n"
            "    if n <= 1:\\n"
            "        return n\\n"
            "    a, b = 0, 1\\n"
            "    for _ in range(2, n + 1):\\n"
            "        a, b = b, a + b\\n"
            "    return b\\n"
        ),
    )

    # Test it
    result = factory.test_tool("fibonacci", {"n": 10})
    assert result == 55
"""

from openagentflow.meta.sandbox import (
    Sandbox,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
)
from openagentflow.meta.tool_factory import ToolFactory, ToolFactoryError

__all__ = [
    "Sandbox",
    "SandboxValidationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    "ToolFactory",
    "ToolFactoryError",
]
