"""
OpenAgentFlow Example 05: Reasoning Engines

Demonstrates:
- Using multiple reasoning engines
- Inspecting ReasoningTrace (steps, LLM calls, DAG)
- Comparing engine outputs
"""
import asyncio
from openagentflow.reasoning import (
    DialecticalSpiral,
    DreamWakeCycle,
    MetaCognitiveLoop,
    AdversarialSelfPlay,
    EvolutionaryThought,
    FractalRecursion,
    ResonanceNetwork,
    TemporalRecursion,
    SimulatedAnnealing,
    SocraticInterrogation,
)


async def demo_engine(engine, name, prompt, provider):
    """Run a reasoning engine and print the trace summary."""
    print(f"\n--- {name} ---")
    try:
        trace = await engine.reason(prompt, provider)
        print(f"  Steps: {len(trace.steps)}")
        print(f"  LLM calls: {trace.total_llm_calls}")
        print(f"  Duration: {trace.duration_ms:.0f}ms")
        print(f"  Output: {trace.final_output[:200]}...")

        # Show step types
        step_types = [s.step_type for s in trace.steps]
        print(f"  Step types: {step_types}")
    except Exception as e:
        print(f"  Error: {e}")


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 05: Reasoning Engines")
    print("=" * 60)

    # You need an LLM provider to run reasoning engines
    # Option 1: Anthropic
    # from openagentflow.llm.providers.anthropic_ import AnthropicProvider
    # provider = AnthropicProvider(api_key="sk-ant-...")

    # Option 2: Ollama (local, no API key)
    # from openagentflow.llm.providers.ollama_ import OllamaProvider
    # provider = OllamaProvider(base_url="http://localhost:11434")

    # Option 3: Mock (for demonstration)
    from openagentflow.llm.providers.mock import MockProvider
    provider = MockProvider()

    prompt = "Should we use microservices or a monolith for our new e-commerce platform?"

    print(f"\nPrompt: {prompt}")
    print(f"Provider: {type(provider).__name__}")

    engines = [
        (DialecticalSpiral(max_depth=2), "Dialectical Spiral"),
        (AdversarialSelfPlay(max_rounds=2), "Adversarial Self-Play"),
        (DreamWakeCycle(max_cycles=2), "Dream-Wake Cycle"),
        (SocraticInterrogation(max_rounds=3), "Socratic Interrogation"),
        (TemporalRecursion(future_perspectives=["1 month", "1 year"]), "Temporal Recursion"),
    ]

    for engine, name in engines:
        await demo_engine(engine, name, prompt, provider)

    print("\n\nAll 10 engines available:")
    all_engines = [
        "DialecticalSpiral", "DreamWakeCycle", "MetaCognitiveLoop",
        "AdversarialSelfPlay", "EvolutionaryThought", "FractalRecursion",
        "ResonanceNetwork", "TemporalRecursion", "SimulatedAnnealing",
        "SocraticInterrogation",
    ]
    for i, name in enumerate(all_engines, 1):
        print(f"  {i:2d}. {name}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
