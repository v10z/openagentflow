"""
OpenAgentFlow Example 02: Autonomous Agent

Demonstrates:
- Creating an agent with @agent decorator
- Giving agents tools
- Running the ReAct loop
- Inspecting AgentResult (output, status, tokens)
"""
import asyncio
from openagentflow import agent, tool


@tool
def search_knowledge(query: str) -> list[dict]:
    """Search a knowledge base for relevant information."""
    # Simulated knowledge base
    knowledge = {
        "python": [
            {"title": "Python 3.12 Released", "content": "Python 3.12 brings improved error messages and performance."},
            {"title": "Type Hints Guide", "content": "Use type hints for better code documentation and IDE support."},
        ],
        "ai": [
            {"title": "LLM Trends 2025", "content": "Multi-agent systems and reasoning engines are growing rapidly."},
            {"title": "RAG Architecture", "content": "Retrieval-augmented generation combines search with LLM output."},
        ],
        "web": [
            {"title": "HTMX Rising", "content": "HTMX enables dynamic web apps without heavy JavaScript frameworks."},
        ],
    }
    results = []
    for topic, entries in knowledge.items():
        if topic in query.lower() or any(query.lower() in e["title"].lower() for e in entries):
            results.extend(entries)
    return results if results else [{"title": "No results", "content": f"No matches for: {query}"}]


@tool
def summarize(text: str, max_words: int = 50) -> str:
    """Summarize text to a maximum word count."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


@agent(model="claude-opus-4-6", tools=[search_knowledge, summarize])
async def research_assistant(question: str) -> str:
    """An AI research assistant that searches for information and provides concise answers."""
    pass  # ReAct loop handles execution


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 02: Autonomous Agent")
    print("=" * 60)

    print("\nAsking the research assistant about AI trends...")
    print("(The agent will use search_knowledge and summarize tools autonomously)\n")

    try:
        result = await research_assistant("What are the latest AI trends?")
        print(f"Status: {result.status}")
        print(f"Output: {result.output}")
        print(f"Total tokens: {result.total_tokens}")
        print(f"Tool calls: {len(result.tool_calls) if hasattr(result, 'tool_calls') else 'N/A'}")
    except Exception as e:
        print(f"Agent execution requires an LLM provider.")
        print(f"Set ANTHROPIC_API_KEY or install Ollama for local inference.")
        print(f"Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
