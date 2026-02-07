"""
OpenAgentFlow Agents - 20 Specialized AI Agents

A collection of specialized agents for code review, improvement, and creative tasks.

Agent Categories:
- Code Quality (5): style_enforcer, complexity_analyzer, dead_code_hunter, pattern_detector, consistency_checker
- Security (2): vulnerability_scanner, secrets_detector
- Documentation (2): docstring_generator, readme_writer
- Testing (2): test_generator, coverage_analyzer
- Refactoring (2): code_modernizer, architecture_advisor
- Creative (4): code_explainer, idea_generator, code_translator, name_suggester
- Research (3): dependency_researcher, performance_profiler, best_practices_advisor

Usage:
    from openagentflow.agents import code_quality, security

    # Run a code review swarm
    @swarm(agents=[
        "style_enforcer",
        "complexity_analyzer",
        "vulnerability_scanner"
    ], strategy="synthesis")
    async def code_review(code: str) -> dict:
        pass

    result = await code_review(my_source_code)
"""

from openagentflow.agents import code_quality
from openagentflow.agents import security
from openagentflow.agents import documentation
from openagentflow.agents import testing
from openagentflow.agents import refactoring
from openagentflow.agents import creative
from openagentflow.agents import research

__all__ = [
    "code_quality",
    "security",
    "documentation",
    "testing",
    "refactoring",
    "creative",
    "research",
]
