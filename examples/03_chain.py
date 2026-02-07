"""
OpenAgentFlow Example 03: Chain (Sequential Pipeline)

Demonstrates:
- Chaining agents with @chain
- Sequential execution: output of one agent feeds into the next
- Building multi-step processing pipelines
"""
import asyncio
from openagentflow import agent, chain, tool


@tool
def extract_requirements(text: str) -> list[str]:
    """Extract requirements from a project description."""
    # Simple keyword-based extraction
    requirements = []
    keywords = ["must", "should", "need", "require", "want"]
    for sentence in text.split("."):
        if any(kw in sentence.lower() for kw in keywords):
            requirements.append(sentence.strip())
    return requirements if requirements else ["No explicit requirements found in text."]


@tool
def estimate_complexity(requirements: list[str]) -> dict:
    """Estimate project complexity from requirements."""
    return {
        "total_requirements": len(requirements),
        "complexity": "high" if len(requirements) > 5 else "medium" if len(requirements) > 2 else "low",
        "estimated_sprints": max(1, len(requirements) // 3),
    }


@agent(model="claude-opus-4-6", tools=[extract_requirements])
async def analyst(project_description: str) -> str:
    """Analyze a project and extract key requirements."""
    pass


@agent(model="claude-opus-4-6", tools=[estimate_complexity])
async def planner(analysis: str) -> str:
    """Create a project plan from an analysis."""
    pass


@agent(model="claude-opus-4-6")
async def reviewer(plan: str) -> str:
    """Review a project plan and provide feedback."""
    pass


@chain(agents=["analyst", "planner", "reviewer"])
async def project_pipeline(project_description: str) -> str:
    """Analyze -> Plan -> Review pipeline."""
    pass


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 03: Chain (Sequential Pipeline)")
    print("=" * 60)

    project = """
    We need to build a real-time chat application. It must support
    multiple rooms and private messaging. Users should be able to
    share files up to 10MB. The system needs to handle 10,000
    concurrent users. We require end-to-end encryption for all
    messages. The frontend should be responsive and work on mobile.
    """

    print(f"\nProject description:\n{project.strip()}")
    print("\nRunning pipeline: Analyst -> Planner -> Reviewer\n")

    try:
        result = await project_pipeline(project)
        print(f"Pipeline output:\n{result.output}")
    except Exception as e:
        print(f"Chain execution requires an LLM provider.")
        print(f"Set ANTHROPIC_API_KEY or install Ollama for local inference.")
        print(f"Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
