# Agents Reference

## Overview

OpenAgentFlow includes 20 specialized agents for code review, analysis, and improvement. All agents use the `@agent` decorator and execute via the ReAct (Reason + Act) loop, which enables iterative tool use, self-correction, and structured output.

Agents are organized into 8 categories. Each agent can run independently or be composed into a swarm for multi-agent workflows.

---

## Code Quality (5 agents)

| Agent | Purpose |
|-------|---------|
| `style_enforcer` | PEP 8 compliance, naming conventions, formatting rules |
| `complexity_analyzer` | Cyclomatic complexity, nesting depth, cognitive complexity |
| `dead_code_hunter` | Unused imports, unreachable code, unused variables and functions |
| `pattern_detector` | Anti-patterns, code smells, common pitfalls |
| `consistency_checker` | Style consistency across the entire codebase |

### style_enforcer

Checks source code against PEP 8 and configurable style rules. Reports violations with line numbers and suggested fixes.

### complexity_analyzer

Computes cyclomatic complexity, nesting depth, and cognitive complexity metrics for functions and classes. Flags code that exceeds configurable thresholds.

### dead_code_hunter

Identifies unused imports, variables, functions, and unreachable branches. Produces a report with safe-to-remove candidates and confidence levels.

### pattern_detector

Scans for known anti-patterns and code smells such as god classes, long parameter lists, feature envy, and duplicated logic.

### consistency_checker

Analyzes naming conventions, import ordering, docstring styles, and formatting choices across multiple files to detect inconsistencies.

---

## Security (2 agents)

| Agent | Purpose |
|-------|---------|
| `vulnerability_scanner` | SQL injection, XSS, command injection, path traversal |
| `secrets_detector` | Hardcoded secrets, API keys, passwords, tokens |

### vulnerability_scanner

Analyzes code for common vulnerability classes including SQL injection, cross-site scripting (XSS), command injection, and path traversal. Reports findings with severity ratings and remediation guidance.

### secrets_detector

Scans source files for hardcoded secrets, API keys, passwords, tokens, and other sensitive data that should not appear in version control.

---

## Documentation (2 agents)

| Agent | Purpose |
|-------|---------|
| `docstring_generator` | Generate or improve function and class docstrings |
| `readme_writer` | Generate README files, changelogs, and API documentation |

### docstring_generator

Generates or improves docstrings for functions, classes, and modules. Supports Google, NumPy, and Sphinx docstring formats. Infers parameter types and return values from code context.

### readme_writer

Produces README files, changelogs, and API documentation by analyzing the project structure, public interfaces, and existing documentation.

---

## Testing (2 agents)

| Agent | Purpose |
|-------|---------|
| `test_generator` | Unit test generation with assertions and edge cases |
| `coverage_analyzer` | Test coverage analysis and gap identification |

### test_generator

Generates unit tests for functions and classes. Produces tests with meaningful assertions, edge case coverage, and appropriate use of mocks and fixtures.

### coverage_analyzer

Analyzes existing test coverage, identifies untested code paths, and prioritizes coverage gaps by risk and complexity.

---

## Refactoring (2 agents)

| Agent | Purpose |
|-------|---------|
| `code_modernizer` | Python 3.10+ syntax updates, modern idioms |
| `architecture_advisor` | Design patterns, SOLID principles, structural improvements |

### code_modernizer

Suggests updates to modern Python syntax including structural pattern matching, type union syntax (`X | Y`), walrus operator usage, and other Python 3.10+ features.

### architecture_advisor

Reviews code structure against SOLID principles and common design patterns. Recommends refactoring strategies to improve maintainability, testability, and separation of concerns.

---

## Creative (4 agents)

| Agent | Purpose |
|-------|---------|
| `code_explainer` | Explain code in plain English |
| `idea_generator` | Feature ideas, improvements, alternative approaches |
| `code_translator` | Convert between paradigms (async, OOP, functional) |
| `name_suggester` | Better variable, function, and class names |

### code_explainer

Produces clear, plain-English explanations of code at any level of detail. Useful for onboarding, code reviews, and documentation.

### idea_generator

Analyzes existing code and suggests feature ideas, architectural improvements, and alternative implementation approaches.

### code_translator

Converts code between programming paradigms. Transforms synchronous code to async, procedural code to object-oriented, or imperative code to functional style.

### name_suggester

Proposes improved names for variables, functions, classes, and modules based on their usage context, domain conventions, and readability.

---

## Research (3 agents)

| Agent | Purpose |
|-------|---------|
| `dependency_researcher` | Analyze and suggest dependencies |
| `performance_profiler` | Performance analysis and optimization |
| `best_practices_advisor` | Industry best practices and standards |

### dependency_researcher

Analyzes project dependencies for version conflicts, security advisories, licensing issues, and lighter-weight alternatives.

### performance_profiler

Identifies performance bottlenecks, unnecessary allocations, inefficient algorithms, and suggests optimizations with estimated impact.

### best_practices_advisor

Reviews code against industry best practices, language idioms, and framework-specific conventions. Provides actionable recommendations with references.

---

## Usage Examples

### Running a Single Agent

```python
from openagentflow.agents import code_quality, security

# Run the style enforcer on source code
result = await code_quality.style_enforcer(source_code)

# Run the vulnerability scanner
findings = await security.vulnerability_scanner(source_code)
```

### Composing Agents into a Swarm

Use the `@swarm` decorator to run multiple agents together with a defined aggregation strategy.

```python
from openagentflow import swarm

@swarm(agents=[
    "style_enforcer",
    "complexity_analyzer",
    "vulnerability_scanner",
    "secrets_detector"
], strategy="synthesis")
async def full_review(code: str) -> dict:
    """Comprehensive code review combining multiple agents."""
    pass
```

The `strategy` parameter controls how agent outputs are combined:

| Strategy | Behavior |
|----------|----------|
| `synthesis` | Merge all agent outputs into a unified report |
| `vote` | Agents vote on findings; majority rules |
| `pipeline` | Each agent's output feeds into the next |

### Accessing Individual Results

```python
result = await full_review(source_code)

# Aggregated output
print(result.summary)

# Per-agent results
for agent_name, output in result.agent_outputs.items():
    print(f"{agent_name}: {output.status}")
```

---

## Creating Custom Agents

Define custom agents using the `@agent` decorator. Attach tools with `@tool` to extend agent capabilities.

### Basic Custom Agent

```python
from openagentflow import agent

@agent(model="claude-sonnet-4-20250514")
async def my_reviewer(code: str) -> str:
    """Review code for a specific concern."""
    pass  # ReAct loop handles execution
```

### Custom Agent with Tools

```python
from openagentflow import agent, tool

@tool
def my_tool(input: str) -> str:
    """Custom tool for processing input."""
    return f"processed: {input}"

@agent(model="claude-sonnet-4-20250514", tools=[my_tool])
async def my_agent(query: str) -> str:
    """Custom agent with tools."""
    pass  # ReAct loop handles execution
```

### Custom Agent with Configuration

```python
from openagentflow import agent, tool

@tool
def fetch_lint_rules(config_path: str) -> dict:
    """Load lint rules from a configuration file."""
    import json
    with open(config_path) as f:
        return json.load(f)

@agent(
    model="claude-sonnet-4-20250514",
    tools=[fetch_lint_rules],
    max_iterations=10,
    temperature=0.0,
)
async def custom_linter(code: str, config: str = "lint.json") -> dict:
    """Lint code using project-specific rules."""
    pass
```

---

## Agent Configuration

All agents accept common configuration parameters through the `@agent` decorator:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model to use for reasoning |
| `tools` | `list` | `[]` | Tools available to the agent |
| `max_iterations` | `int` | `5` | Maximum ReAct loop iterations |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `stop_condition` | `callable` | `None` | Custom stop condition for the loop |

---

## Agent Lifecycle

Each agent follows the ReAct loop during execution:

1. **Reason** -- The agent analyzes the input and determines the next action.
2. **Act** -- The agent calls a tool or produces an intermediate result.
3. **Observe** -- The agent inspects the tool output.
4. **Repeat** -- Steps 1-3 repeat until the agent reaches a conclusion or hits `max_iterations`.
5. **Return** -- The agent produces a final structured output.

This loop enables agents to self-correct, gather additional context, and iteratively refine their analysis before returning a result.
