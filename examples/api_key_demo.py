"""
API Key Configuration Demo

This demo shows multiple ways to provide API keys to Open Agent Flow.

Options:
1. Environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY)
2. openagentflow.configure(anthropic_api_key="...")
3. openagentflow.load_dotenv() to load from .env file
4. @agent(api_key="...") for per-agent keys

Usage:
    # With environment variable set:
    python examples/api_key_demo.py

    # Or set key inline in code (see Option 2 below)
"""

import asyncio
import os

import openagentflow
from openagentflow import agent, tool, AgentStatus


# Check if we have an API key configured
def check_api_keys():
    """Check which API keys are available."""
    print("=" * 60)
    print("API Key Configuration Check")
    print("=" * 60)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        print(f"[OK] ANTHROPIC_API_KEY: {anthropic_key[:12]}...")
    else:
        print("[--] ANTHROPIC_API_KEY: Not set")

    if openai_key:
        print(f"[OK] OPENAI_API_KEY: {openai_key[:12]}...")
    else:
        print("[--] OPENAI_API_KEY: Not set")

    # Check via openagentflow config
    print("\nChecking via openagentflow.is_configured():")
    print(f"  Anthropic: {openagentflow.is_configured('anthropic')}")
    print(f"  OpenAI: {openagentflow.is_configured('openai')}")

    return anthropic_key or openai_key


# =============================================================================
# OPTION 1: Environment Variables (already set)
# =============================================================================

# Just use @agent - it will find ANTHROPIC_API_KEY automatically
@agent(model="claude-sonnet-4-20250514")
async def env_agent(question: str) -> str:
    """Agent using environment variable key."""
    pass


# =============================================================================
# OPTION 2: Configure programmatically
# =============================================================================

# openagentflow.configure(
#     anthropic_api_key="sk-ant-your-key-here",
#     # Or for OpenAI:
#     # openai_api_key="sk-your-openai-key",
# )


# =============================================================================
# OPTION 3: Load from .env file
# =============================================================================

# Create .env file with:
#   ANTHROPIC_API_KEY=sk-ant-your-key-here
#   OPENAI_API_KEY=sk-your-openai-key
#
# Then call:
# openagentflow.load_dotenv()


# =============================================================================
# OPTION 4: Per-agent API key
# =============================================================================

# @agent(model="claude-sonnet-4-20250514", api_key="sk-ant-specific-key")
# async def specific_agent(question: str) -> str:
#     """Agent with specific API key."""
#     pass


# =============================================================================
# TOOLS
# =============================================================================

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    allowed = set("0123456789+-*/.(). ")
    if all(c in allowed for c in expression):
        try:
            return float(eval(expression))
        except Exception:
            pass
    return 0.0


# =============================================================================
# AGENT WITH TOOLS
# =============================================================================

@agent(
    model="claude-sonnet-4-20250514",
    tools=[get_current_time, calculate],
    max_iterations=3,
    system_prompt="You are a helpful assistant. Use tools when needed.",
)
async def assistant(question: str) -> str:
    """Helpful assistant with tools."""
    pass


# =============================================================================
# DEMO
# =============================================================================

async def main():
    has_key = check_api_keys()

    if not has_key:
        print("\n" + "=" * 60)
        print("No API key found. Here's how to set one:")
        print("=" * 60)
        print("""
Option 1 - Environment variable:
    export ANTHROPIC_API_KEY=sk-ant-your-key-here
    python examples/api_key_demo.py

Option 2 - In code:
    import openagentflow
    openagentflow.configure(anthropic_api_key="sk-ant-...")

Option 3 - .env file:
    echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
    # Then in code: openagentflow.load_dotenv()

Option 4 - Per-agent:
    @agent(model="claude-sonnet-4-20250514", api_key="sk-ant-...")
    async def my_agent(q: str) -> str:
        pass
""")
        print("Falling back to MockProvider for demo...")

    print("\n" + "=" * 60)
    print("Running Agent")
    print("=" * 60)

    print("\n[Query]: What is 25 * 4?")
    result = await assistant("What is 25 * 4?")

    print(f"\n[Status]: {result.status.name}")
    print(f"[Output]: {result.output}")
    print(f"[Tokens]: {result.total_tokens}")
    print(f"[Cost]: ${result.total_cost:.6f}")
    print(f"[Duration]: {result.duration_ms:.1f}ms")

    if result.tool_calls:
        print(f"[Tool Calls]: {len(result.tool_calls)}")
        for tc in result.tool_calls:
            print(f"  - {tc.tool_name}({tc.arguments})")


if __name__ == "__main__":
    asyncio.run(main())
