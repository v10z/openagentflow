"""
OpenAgentFlow Example 11: Code Review Swarm

Demonstrates a real-world use case: multi-agent code review
using specialized agents from the agents module.
"""
import asyncio
from openagentflow import agent, swarm, tool
from openagentflow.tools import code, text


@tool
def get_complexity(source: str) -> dict:
    """Get code complexity metrics."""
    return {
        "cyclomatic": code.calculate_complexity(source),
        "functions": [f["name"] for f in code.extract_functions(source)],
        "todos": code.find_todos(source),
        "lines": code.count_lines_of_code(source),
    }


@tool
def get_naming_issues(source: str) -> list:
    """Check for naming convention violations."""
    return code.check_naming_convention(source)


@agent(model="claude-opus-4-6", tools=[get_complexity])
async def quality_agent(source_code: str) -> str:
    """Analyze code quality: complexity, readability, structure."""
    pass


@agent(model="claude-opus-4-6", tools=[get_naming_issues])
async def style_agent(source_code: str) -> str:
    """Check code style: naming conventions, formatting, consistency."""
    pass


@agent(model="claude-opus-4-6")
async def security_agent(source_code: str) -> str:
    """Scan for security vulnerabilities: injection, XSS, hardcoded secrets."""
    pass


@agent(model="claude-opus-4-6")
async def docs_agent(source_code: str) -> str:
    """Evaluate documentation: docstrings, comments, type hints."""
    pass


@swarm(
    agents=["quality_agent", "style_agent", "security_agent", "docs_agent"],
    strategy="synthesis",
)
async def code_review_panel(source_code: str) -> str:
    """4-agent code review panel. Runs in parallel, synthesizes findings."""
    pass


async def main():
    print("=" * 60)
    print("OpenAgentFlow Example 11: Code Review Swarm")
    print("=" * 60)

    source = '''
import os
import hashlib
from typing import Optional

DB_PASSWORD = "admin123"  # TODO: Move to environment variable

class UserManager:
    def __init__(self):
        self.users = {}
        self.conn = None

    def get_user(self, user_id: str) -> Optional[dict]:
        """Get user by ID."""
        query = f"SELECT * FROM users WHERE id = '{user_id}'"
        # Execute query...
        return self.users.get(user_id)

    def CreateUser(self, name, email, password):
        hashed = hashlib.md5(password.encode()).hexdigest()
        self.users[name] = {
            "name": name,
            "email": email,
            "password_hash": hashed,
        }
        return True

    def delete_all_users(self):
        self.users = {}
        os.system("rm -rf /var/data/users/*")
        return True

    def find_users(self, search_term):
        results = []
        for uid, user in self.users.items():
            if search_term in str(user):
                results.append(user)
        return results
'''

    print(f"\nCode to review ({code.count_lines_of_code(source)} lines):")
    print(source)

    # Run tools directly to show what the agents would use
    print("--- Direct Tool Results ---")
    metrics = get_complexity(source)
    print(f"Complexity: {metrics}")

    naming = get_naming_issues(source)
    print(f"Naming issues: {naming}")

    todos = code.find_todos(source)
    print(f"TODOs: {todos}")

    print("\n--- Swarm Review ---")
    print("(Requires LLM provider - set ANTHROPIC_API_KEY)")

    try:
        result = await code_review_panel(source)
        print(f"\nSynthesized Review:\n{result.output}")
    except Exception as e:
        print(f"Swarm requires LLM provider: {e}")
        print("\nThe swarm would run 4 agents in parallel:")
        print("  1. quality_agent -> complexity, structure analysis")
        print("  2. style_agent -> naming conventions, formatting")
        print("  3. security_agent -> SQL injection, hardcoded secrets, OS commands")
        print("  4. docs_agent -> missing docstrings, type hints")
        print("  -> Synthesis combines all findings into one report")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
