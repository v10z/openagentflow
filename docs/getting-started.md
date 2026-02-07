# Getting Started with OpenAgentFlow

OpenAgentFlow is a Python framework for building autonomous AI agents. It provides
decorators, orchestration primitives, and a rich library of built-in tools so you
can go from zero to a working agent pipeline in minutes.

This guide walks you through installation, creating your first tool and agent,
composing agents into chains and swarms, and configuring provider credentials.

---

## Installation

Install the core package or choose an extras bundle that pulls in the provider
SDK you need.

```bash
# Core package
pip install openagentflow

# With Anthropic Claude support
pip install openagentflow[anthropic]

# With OpenAI support
pip install openagentflow[openai]

# With all providers
pip install openagentflow[all]

# Development (editable install with test and lint tooling)
pip install -e ".[dev]"
```

### Requirements

- **Python 3.10+**
- No mandatory API keys. OpenAgentFlow works out of the box with the Claude Code
  CLI or the built-in `MockProvider`, so you can start experimenting without
  configuring any credentials at all.

---

## Your First Tool

Tools are plain Python functions decorated with `@tool`. OpenAgentFlow
automatically generates a JSON Schema from the function's type hints and
docstring, making the tool available to any agent that references it.

```python
from openagentflow import tool

@tool
def analyze(code: str) -> dict:
    """Analyze code quality."""
    return {"lines": len(code.splitlines()), "chars": len(code)}

result = analyze("print('hello')")
print(analyze._tool_spec.name)          # "analyze"
print(analyze._tool_spec.input_schema)  # JSON Schema from type hints
```

Key points:

- The decorator introspects the function signature and builds an
  `input_schema` that conforms to JSON Schema.
- The tool remains callable as a normal Python function, so you can unit-test
  it without spinning up an agent.
- Docstrings become the tool description that the language model sees.

---

## Your First Agent

An agent wraps a language model call inside a ReAct (Reason + Act) loop. You
declare the model, attach tools, and let the framework handle tool dispatch,
retries, and result aggregation.

```python
import asyncio
from openagentflow import agent, tool

@tool
def search(query: str) -> list[dict]:
    """Search for information."""
    return [{"title": "Result", "url": "https://example.com"}]

@agent(model="claude-sonnet-4-20250514", tools=[search])
async def researcher(question: str) -> str:
    """Research agent that searches and synthesizes."""
    pass  # ReAct loop handles execution

async def main():
    result = await researcher("What are AI trends?")
    print(result.output)
    print(result.status)        # AgentStatus.SUCCEEDED
    print(result.total_tokens)  # Token usage

asyncio.run(main())
```

The function body is intentionally `pass`. The `@agent` decorator replaces
the body with the ReAct execution loop. The docstring tells the model what
role it should play, and the `tools` list determines which actions it can
take during reasoning.

---

## Chaining Agents

Chains let you compose agents into sequential pipelines. The output of one
agent becomes the input of the next.

```python
from openagentflow import agent, chain

@agent(model="claude-sonnet-4-20250514")
async def planner(task: str) -> str:
    """Break down a task."""
    pass

@agent(model="claude-sonnet-4-20250514")
async def executor(plan: str) -> str:
    """Execute the plan."""
    pass

@chain(agents=["planner", "executor"])
async def pipeline(task: str) -> str:
    """Plan then execute."""
    pass
```

When you call `await pipeline("Build a REST API")`, the framework:

1. Runs `planner` with the original task.
2. Passes the planner's output into `executor` as its input.
3. Returns the executor's final output as the pipeline result.

Chains can include as many agents as you need, and each agent in the chain
can have its own tools.

---

## Swarming Agents (Parallel + Consensus)

Swarms run multiple agents in parallel over the same input and then merge
their outputs using a configurable strategy.

```python
from openagentflow import agent, swarm

@agent(model="claude-sonnet-4-20250514")
async def reviewer_1(code: str) -> str:
    """First reviewer."""
    pass

@agent(model="claude-sonnet-4-20250514")
async def reviewer_2(code: str) -> str:
    """Second reviewer."""
    pass

@swarm(agents=["reviewer_1", "reviewer_2"], strategy="synthesis")
async def code_review(code: str) -> str:
    """Consensus from multiple reviewers."""
    pass
```

Available strategies:

| Strategy       | Behavior                                                  |
|----------------|-----------------------------------------------------------|
| `"synthesis"`  | Merge all outputs into a single cohesive response.        |
| `"majority"`   | Return the answer agreed upon by the majority of agents.  |
| `"first"`      | Return whichever agent finishes first.                    |
| `"all"`        | Return every agent's output as a list.                    |

Swarms are useful for code review, fact-checking, brainstorming, and any
scenario where multiple perspectives improve the final answer.

---

## Zero API Key Mode

OpenAgentFlow resolves provider credentials through the following priority
chain. If a higher-priority source is available, lower-priority sources are
ignored.

1. **Direct parameter** -- pass `api_key` when constructing a provider or agent.
2. **`configure()` call** -- call `configure(anthropic_api_key="...")` at
   startup.
3. **Environment variable** -- set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`)
   in your shell.
4. **Claude Code CLI** -- if none of the above are present and the Claude Code
   CLI is installed, OpenAgentFlow automatically delegates to it. No key
   needed.
5. **MockProvider** -- for unit tests and CI pipelines, the built-in
   `MockProvider` returns deterministic responses without any network calls.

This means you can clone the repository, run the examples, and build agents
without ever touching an API key, as long as you have the Claude Code CLI
available.

---

## Configuration

### Programmatic

```python
from openagentflow import configure

configure(
    anthropic_api_key="sk-ant-...",
    openai_api_key="sk-...",
)
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Set these in your shell profile, a `.env` file (loaded by your application),
or your CI secrets store. The `configure()` function and environment variables
can be mixed: explicit values from `configure()` take precedence over
environment variables.

---

## Next Steps

Now that you have a working agent, explore the rest of the framework:

- [Agents](agents.md) -- 20 specialized agents for common workflows.
- [Tools](tools.md) -- 99 built-in tools covering file I/O, HTTP, parsing, and
  more.
- [Reasoning Engines](reasoning.md) -- 10 advanced reasoning strategies
  (chain-of-thought, tree-of-thought, self-consistency, and others).
- [Memory System](memory.md) -- 3-tier memory hierarchy (working, episodic,
  semantic) for long-running agents.
- [Graph Tracing](graph.md) -- Execution DAG tracing for debugging and
  observability.
- [Meta-Agent](meta-agent.md) -- JIT tool creation: agents that build their
  own tools at runtime.
- [Distributed Compute](distributed.md) -- Scale out with Kubernetes, Docker,
  and SSH backends.
