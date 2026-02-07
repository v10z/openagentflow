"""
Demo: Open Agent Flow - Works without API keys!

This demo shows the agent framework in action using the MockProvider.
No API keys needed - just run it!

Usage:
    python examples/demo.py
"""

import asyncio

from openagentflow import agent, tool, AgentStatus


# Define tools
@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "cloudy", "humidity": 65},
        "New York": {"temp": 18, "condition": "sunny", "humidity": 45},
        "London": {"temp": 15, "condition": "rainy", "humidity": 80},
        "Paris": {"temp": 20, "condition": "partly cloudy", "humidity": 55},
    }
    return weather_data.get(city, {"temp": 20, "condition": "unknown", "humidity": 50})


@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    allowed = set("0123456789+-*/.(). ")
    if all(c in allowed for c in expression):
        try:
            return float(eval(expression))
        except Exception:
            pass
    return 0.0


@tool
def search_products(query: str) -> list[dict]:
    """Search for products in the catalog."""
    products = [
        {"name": "Laptop Pro", "price": 1299, "category": "electronics"},
        {"name": "Wireless Mouse", "price": 49, "category": "electronics"},
        {"name": "Coffee Maker", "price": 89, "category": "kitchen"},
        {"name": "Running Shoes", "price": 129, "category": "sports"},
    ]
    query_lower = query.lower()
    return [p for p in products if query_lower in p["name"].lower() or query_lower in p["category"]]


# Define agent - will auto-use MockProvider since no API key
@agent(
    model="claude-sonnet-4-20250514",
    tools=[get_weather, calculate, search_products],
    max_iterations=5,
    system_prompt="You are a helpful assistant with access to weather, calculator, and product search tools.",
)
async def assistant(question: str) -> str:
    """Multi-purpose assistant agent."""
    pass


async def main():
    print("=" * 60)
    print("Open Agent Flow - Demo (using MockProvider)")
    print("=" * 60)
    print("\nNo API keys needed! The MockProvider simulates LLM responses.")
    print()

    # Example 1: Weather
    print("-" * 60)
    print("[Query 1]: What's the weather in Tokyo?")
    print("-" * 60)
    result = await assistant("What's the weather in Tokyo?")
    print(f"\n[Status]: {result.status.name}")
    print(f"[Output]: {result.output}")
    print(f"[Tokens]: {result.total_tokens}")
    print(f"[Duration]: {result.duration_ms:.1f}ms")

    # Example 2: Calculation
    print("\n" + "-" * 60)
    print("[Query 2]: Calculate 15 * 7 + 23")
    print("-" * 60)
    result = await assistant("Calculate 15 * 7 + 23")
    print(f"\n[Status]: {result.status.name}")
    print(f"[Output]: {result.output}")

    # Example 3: Product search
    print("\n" + "-" * 60)
    print("[Query 3]: Search for electronics products")
    print("-" * 60)
    result = await assistant("Search for electronics products")
    print(f"\n[Status]: {result.status.name}")
    print(f"[Output]: {result.output}")

    print("\n" + "=" * 60)
    print("Demo complete! To use real LLM:")
    print("  export ANTHROPIC_API_KEY=your-key")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
