# Claude Code Skill for OpenAgentFlow

## What is a Claude Code Skill?

A Claude Code skill is a markdown file with YAML frontmatter that teaches Claude how to use a framework or tool during its work. When installed, it becomes available as a slash command (e.g., `/openagentflow`) that Claude can reference.

## Installation

### Option 1: Copy to your home directory (recommended)

```bash
# From the openagentflow repo root:
mkdir -p ~/.claude/commands
cp .claude/commands/openagentflow.md ~/.claude/commands/openagentflow.md
```

### Option 2: Project-level (auto-detected)

If you have the `.claude/commands/openagentflow.md` file in your project root, Claude Code will automatically detect it when working in that directory.

## Usage

Once installed, use the `/openagentflow` slash command in Claude Code to activate the skill. Claude will then know how to:

- Import and use all openagentflow decorators (`@agent`, `@tool`, `@chain`, `@swarm`)
- Run any of the 30 reasoning engines
- Use the 3-tier memory system
- Create graph traces with SQLite backend
- Dynamically create tools with the JIT Meta-Agent
- Work with distributed compute clusters
- Use all 99 built-in tools across 9 categories

## Skill File Structure

The skill file at `.claude/commands/openagentflow.md` follows this format:

```yaml
---
description: Short description of what the skill enables
allowed-tools: Bash(python:*), Read, Glob, Grep, Write, Edit
---

# Skill content (markdown)

## Quick Imports
...

## Usage Patterns
...
```

### YAML Frontmatter Fields

| Field | Description |
|-------|-------------|
| `description` | Shown in the command palette when browsing skills |
| `allowed-tools` | Which Claude Code tools the skill can use |

## Creating Your Own Skills

You can create custom skills for your own openagentflow agents:

### Example: Custom Agent Skill

Create `.claude/commands/my-agent.md`:

```yaml
---
description: Run my custom analysis agent
allowed-tools: Bash(python:*), Read, Write
---

# My Analysis Agent

Run the analysis agent from the project:

` ` `python
import asyncio
from myproject.agents import analysis_agent

async def run():
    result = await analysis_agent("Analyze this code for bugs")
    print(result.output)

asyncio.run(run())
` ` `
```

### Example: Reasoning Engine Skill

Create `.claude/commands/reason.md`:

```yaml
---
description: Use adversarial reasoning to stress-test a solution
allowed-tools: Bash(python:*), Read
---

# Adversarial Reasoning

Use when you need to stress-test a solution through Red/Blue/Judge debate.

` ` `python
from openagentflow.reasoning import AdversarialSelfPlay

engine = AdversarialSelfPlay(max_rounds=5)
trace = await engine.reason(prompt, llm_provider)

# Red team attacks, Blue team defends, Judge decides
for step in trace.steps:
    print(f"[{step.step_type}] {step.content[:100]}")
` ` `
```

## Tips

- Keep skills focused on one capability or workflow
- Include common gotchas and error patterns
- Provide copy-pasteable code examples
- Reference file paths relative to the project root
