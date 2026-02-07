"""
OpenAgentFlow Example 12: Local AI with Ollama

Demonstrates:
- Using Ollama for local LLM inference (no API key needed)
- Checking Ollama availability
- Creating agents with Ollama provider
- Running reasoning engines locally
"""
import asyncio
from openagentflow import agent, tool
from openagentflow.llm.providers import OllamaProvider, is_ollama_available


@tool
def calculate(expression: str) -> float:
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        raise ValueError(f"Invalid characters in expression: {expression}")
    return float(eval(expression))


@tool
def lookup(topic: str) -> str:
    """Look up information about a topic."""
    knowledge = {
        "python": "Python is a high-level programming language known for readability.",
        "rust": "Rust is a systems programming language focused on safety and performance.",
        "ollama": "Ollama is a tool for running large language models locally.",
        "openagentflow": "OpenAgentFlow is a Python framework for building AI agents.",
    }
    return knowledge.get(topic.lower(), f"No information found for: {topic}")


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 12: Local AI with Ollama")
    print("=" * 60)

    # Check if Ollama is running
    # Note: is_ollama_available() is a synchronous function (no await needed)
    print("\n--- Checking Ollama availability ---")
    available = is_ollama_available()
    print(f"Ollama available: {available}")

    if not available:
        print("\nOllama is not running. To use this example:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull a model: ollama pull llama3")
        print("  3. Start Ollama: ollama serve")
        print("  4. Run this example again")
        print("\nShowing what the code looks like anyway:\n")

    # Create an Ollama provider
    print("--- Creating Ollama Provider ---")
    print('provider = OllamaProvider(base_url="http://localhost:11434")')
    print("# Note: OllamaProvider takes base_url, NOT model")
    print("# The model is specified per-request or in the agent decorator\n")

    # Define an agent with Ollama
    @agent(model="llama3", tools=[calculate, lookup])
    async def local_assistant(question: str) -> str:
        """A local AI assistant powered by Ollama."""
        pass

    print("--- Agent defined with Ollama model ---")
    print("@agent(model='llama3', tools=[calculate, lookup])")
    print("async def local_assistant(question: str) -> str: ...")

    if available:
        print("\n--- Running local agent ---")
        try:
            result = await local_assistant("What is 42 * 17?")
            print(f"Result: {result.output}")
        except Exception as e:
            print(f"Error: {e}")

    # Show Ollama provider features
    print("\n--- Ollama Provider Features ---")
    print("  - Zero API key required (runs locally)")
    print("  - Supports all Ollama models (llama3, mistral, codellama, etc.)")
    print("  - Tool calling support (function calling)")
    print("  - Token counting via Ollama API")
    print("  - Cost estimation: always $0.00 (it's local!)")
    print("  - Custom base_url for remote Ollama servers")

    # Show provider API
    print("\n--- Provider API ---")
    print("""
from openagentflow.core.types import LLMProvider, Message, ModelConfig

provider = OllamaProvider(base_url="http://localhost:11434")

# Generate a response
response = await provider.generate(
    messages=[Message(role="user", content="Hello!")],
    config=ModelConfig(provider=LLMProvider.OLLAMA, model_id="llama3"),
)
print(response.content)

# Count tokens (synchronous)
tokens = provider.count_tokens("Hello world", model_id="llama3")

# Estimate cost (always free!)
cost = provider.estimate_cost(1000, 500, "llama3")
# cost == 0.0 (local inference is free!)
""")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
