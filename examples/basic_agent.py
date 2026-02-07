"""
Basic agent example for Open Agent Flow.

This example demonstrates:
- Defining tools with @tool decorator
- Defining agents with @agent decorator
- Running agents with async/await
- Accessing agent results (status, output, tokens, cost)

Requirements:
    pip install openagentflow[anthropic]
    export ANTHROPIC_API_KEY=your-api-key
"""

import asyncio
import os

from openagentflow import agent, tool


# Define tools
@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    # In a real implementation, this would call a weather API
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "cloudy", "humidity": 65},
        "New York": {"temp": 18, "condition": "sunny", "humidity": 45},
        "London": {"temp": 15, "condition": "rainy", "humidity": 80},
        "Paris": {"temp": 20, "condition": "partly cloudy", "humidity": 55},
    }
    return weather_data.get(city, {"temp": 20, "condition": "unknown", "humidity": 50})


@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    # Only allow safe operations
    allowed_chars = set("0123456789+-*/.(). ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression")
    try:
        return float(eval(expression))
    except Exception as e:
        raise ValueError(f"Cannot evaluate: {e}")


# Define agent
@agent(
    model="claude-sonnet-4-20250514",
    tools=[get_weather, calculate],
    max_iterations=5,
    system_prompt="You are a helpful assistant that can check weather and do calculations.",
)
async def assistant(question: str) -> str:
    """
    A helpful assistant that can answer questions about weather and math.

    The agent will:
    1. Understand the user's question
    2. Decide which tool(s) to use
    3. Execute the tools
    4. Synthesize a response
    """
    pass  # The reasoning loop is handled by the framework


async def main():
    """Run the example agent."""
    print("=" * 60)
    print("Open Agent Flow - Basic Agent Example")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nNote: ANTHROPIC_API_KEY not set.")
        print("Set it with: export ANTHROPIC_API_KEY=your-key")
        print("\nRunning in demo mode (showing agent structure only)...")

        # Show what the agent looks like
        from openagentflow import get_agent

        spec = get_agent("assistant")
        print(f"\nAgent: {spec.name}")
        print(f"Model: {spec.model.model_id}")
        print(f"Tools: {[t.name for t in spec.tools]}")
        print(f"Max iterations: {spec.max_iterations}")
        return

    # Example 1: Weather query
    print("\n[Query 1]: What's the weather in Tokyo?")
    result = await assistant("What's the weather in Tokyo?")
    print(f"[Status]: {result.status.name}")
    print(f"[Output]: {result.output}")
    print(f"[Tokens]: {result.total_tokens}")
    print(f"[Cost]: ${result.total_cost:.6f}")

    # Example 2: Calculation
    print("\n[Query 2]: What is 15 * 7 + 23?")
    result = await assistant("What is 15 * 7 + 23?")
    print(f"[Status]: {result.status.name}")
    print(f"[Output]: {result.output}")

    # Example 3: Combined query
    print("\n[Query 3]: Is it warmer in Tokyo or London? By how many degrees?")
    result = await assistant("Is it warmer in Tokyo or London? By how many degrees?")
    print(f"[Status]: {result.status.name}")
    print(f"[Output]: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
