"""
OpenAgentFlow Example 04: Swarm (Parallel + Consensus)

Demonstrates:
- Running multiple agents in parallel with @swarm
- Three consensus strategies: voting, synthesis, first
- Multi-perspective analysis
"""
import asyncio
from openagentflow import agent, swarm, tool
from openagentflow.tools import code


# Wrap built-in tools with @tool so the agent can discover them
@tool
def get_complexity(source: str) -> int:
    """Calculate cyclomatic complexity of source code."""
    return code.calculate_complexity(source)


@tool
def get_todos(source: str) -> list[dict]:
    """Find TODO/FIXME comments in source code."""
    return code.find_todos(source)


@agent(model="claude-opus-4-6", tools=[get_complexity, get_todos])
async def quality_reviewer(source_code: str) -> str:
    """Review code quality: complexity, readability, maintainability."""
    pass


@agent(model="claude-opus-4-6")
async def security_reviewer(source_code: str) -> str:
    """Review code for security vulnerabilities: injection, XSS, secrets."""
    pass


@agent(model="claude-opus-4-6")
async def performance_reviewer(source_code: str) -> str:
    """Review code for performance issues: N+1 queries, memory leaks, blocking I/O."""
    pass


# Synthesis strategy: combine all perspectives into one report
@swarm(
    agents=["quality_reviewer", "security_reviewer", "performance_reviewer"],
    strategy="synthesis",
)
async def comprehensive_review(source_code: str) -> str:
    """Run all reviewers in parallel and synthesize their findings."""
    pass


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 04: Swarm (Parallel + Consensus)")
    print("=" * 60)

    sample_code = '''
import sqlite3
import os

def get_user(user_id):
    conn = sqlite3.connect("app.db")
    # WARNING: SQL injection vulnerability
    cursor = conn.execute(f"SELECT * FROM users WHERE id = {user_id}")
    user = cursor.fetchone()
    conn.close()
    return user

def process_all_users():
    conn = sqlite3.connect("app.db")
    users = conn.execute("SELECT id FROM users").fetchall()
    results = []
    for user in users:
        # N+1 query problem
        details = conn.execute(f"SELECT * FROM user_details WHERE user_id = {user[0]}").fetchone()
        results.append(details)
    conn.close()
    return results

API_KEY = "sk-secret-key-12345"  # TODO: Move to env var
'''

    print(f"\nCode to review:\n{sample_code}")
    print("Running 3 reviewers in parallel (quality, security, performance)...\n")

    try:
        result = await comprehensive_review(sample_code)
        print(f"Synthesized review:\n{result.output}")
    except Exception as e:
        print(f"Swarm execution requires an LLM provider.")
        print(f"Set ANTHROPIC_API_KEY or install Ollama for local inference.")
        print(f"Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
