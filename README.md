<p align="center">
  <img src="https://raw.githubusercontent.com/v10z/openagentflow/main/docs/logo.svg" alt="OpenAgentFlow Logo" width="400">
</p>

<h1 align="center">Open Agent Flow</h1>

<p align="center">
  <strong>Distributed Agentic AI Workflows with Graph-Native Reasoning Traces</strong>
</p>

<p align="center">
  <a href="https://github.com/v10z/openagentflow/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
  <a href="https://github.com/v10z/openagentflow">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
  </a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#agents">20 Agents</a> •
  <a href="#tools">107 Tools</a> •
  <a href="#architecture">Architecture</a>
</p>

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ╔═══════════════════════════════════════════════════════════════════╗    │
│   ║                      OPEN AGENT FLOW                              ║    │
│   ╠═══════════════════════════════════════════════════════════════════╣    │
│   ║                                                                   ║    │
│   ║    @agent ──────► @tool ──────► @chain ──────► @swarm            ║    │
│   ║       │             │             │              │                ║    │
│   ║       ▼             ▼             ▼              ▼                ║    │
│   ║   ┌───────┐    ┌───────┐    ┌───────┐    ┌────────────┐          ║    │
│   ║   │ LLM   │    │ Pure  │    │Serial │    │  Parallel  │          ║    │
│   ║   │Powered│    │Python │    │ Exec  │    │ + Consensus│          ║    │
│   ║   └───────┘    └───────┘    └───────┘    └────────────┘          ║    │
│   ║                                                                   ║    │
│   ╠═══════════════════════════════════════════════════════════════════╣    │
│   ║   20 Specialized Agents  │  107 Pure Python Tools                 ║    │
│   ╚═══════════════════════════════════════════════════════════════════╝    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Open Agent Flow is a Python framework for building autonomous AI agents with:

- **Graph-native reasoning traces** - Every thought, tool call, and decision recorded
- **Multi-LLM support** - Anthropic Claude, OpenAI GPT, AWS Bedrock, Ollama
- **20 specialized agents** - Code review, security, testing, documentation, and more
- **107 pure Python tools** - Text, code, data, web, math, crypto, media, AI utilities
- **Zero API key mode** - Works with Claude Code CLI (no key needed!)

## Features

<table>
<tr>
<td width="50%">

### Decorators
```python
@agent    # Autonomous AI agent
@tool     # Pure Python function
@chain    # Sequential pipeline
@swarm    # Parallel + consensus
```

</td>
<td width="50%">

### LLM Providers
```python
claude-sonnet-4-20250514  # Anthropic
gpt-4o                    # OpenAI
claude-code               # CLI (free!)
mock                      # Testing
```

</td>
</tr>
</table>

## Installation

```bash
# Core package
pip install openagentflow

# With Anthropic Claude
pip install openagentflow[anthropic]

# With all providers
pip install openagentflow[all]

# Development
pip install -e ".[dev]"
```

## Quick Start

### Define a Tool

```python
from openagentflow import tool

@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression safely."""
    return eval(expression)  # Use safe evaluator in production
```

### Define an Agent

```python
from openagentflow import agent, tool

@tool
def search(query: str) -> list[dict]:
    """Search for information."""
    return [{"title": "Result", "url": "https://..."}]

@agent(model="claude-sonnet-4-20250514", tools=[search])
async def researcher(question: str) -> str:
    """Research agent that searches and synthesizes."""
    pass
```

### Run the Agent

```python
import asyncio

async def main():
    result = await researcher("What are AI trends in 2025?")
    print(result.output)

asyncio.run(main())
```

### Chain Agents (Sequential)

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

### Swarm Agents (Parallel + Consensus)

```python
from openagentflow import agent, swarm

@agent(model="claude-sonnet-4-20250514")
async def reviewer_1(code: str) -> str:
    """First reviewer perspective."""
    pass

@agent(model="claude-sonnet-4-20250514")
async def reviewer_2(code: str) -> str:
    """Second reviewer perspective."""
    pass

@swarm(agents=["reviewer_1", "reviewer_2"], strategy="synthesis")
async def code_review(code: str) -> str:
    """Get consensus from multiple reviewers."""
    pass
```

## Agents

Open Agent Flow includes **20 specialized agents** for code review and improvement:

### Code Quality (5 agents)
| Agent | Purpose |
|-------|---------|
| `style_enforcer` | PEP 8, naming conventions, formatting |
| `complexity_analyzer` | Cyclomatic complexity, nesting depth |
| `dead_code_hunter` | Unused imports, variables, functions |
| `pattern_detector` | Anti-patterns, code smells |
| `consistency_checker` | Style consistency across codebase |

### Security (2 agents)
| Agent | Purpose |
|-------|---------|
| `vulnerability_scanner` | SQL injection, XSS, command injection |
| `secrets_detector` | Hardcoded secrets, API keys, passwords |

### Documentation (2 agents)
| Agent | Purpose |
|-------|---------|
| `docstring_generator` | Generate/improve docstrings |
| `readme_writer` | README, changelog, API docs |

### Testing (2 agents)
| Agent | Purpose |
|-------|---------|
| `test_generator` | Unit test generation |
| `coverage_analyzer` | Test coverage analysis |

### Refactoring (2 agents)
| Agent | Purpose |
|-------|---------|
| `code_modernizer` | Python 3.10+ syntax updates |
| `architecture_advisor` | Design patterns, SOLID principles |

### Creative (4 agents)
| Agent | Purpose |
|-------|---------|
| `code_explainer` | Explain code in plain English |
| `idea_generator` | Feature ideas, improvements |
| `code_translator` | Convert between styles (async, OOP, functional) |
| `name_suggester` | Better variable/function names |

### Research (3 agents)
| Agent | Purpose |
|-------|---------|
| `dependency_researcher` | Analyze and suggest dependencies |
| `performance_profiler` | Performance analysis |
| `best_practices_advisor` | Industry best practices |

### Usage

```python
from openagentflow.agents import code_quality, security

# Use in a swarm for comprehensive review
@swarm(agents=[
    "style_enforcer",
    "complexity_analyzer",
    "vulnerability_scanner",
    "secrets_detector"
], strategy="synthesis")
async def full_review(code: str) -> dict:
    """Comprehensive code review."""
    pass
```

## Tools

Open Agent Flow includes **107 pure Python tools** across 10 categories:

### Text Processing (15 tools)
```python
from openagentflow.tools import text

text.extract_emails("Contact: hello@example.com")  # ['hello@example.com']
text.text_to_slug("Hello World!")                   # 'hello-world'
text.text_to_morse("SOS")                           # '... --- ...'
text.detect_language("Bonjour le monde")            # 'french'
text.find_palindromes("A man a plan a canal")       # ['a', 'a', 'a']
```

### Code Analysis (15 tools)
```python
from openagentflow.tools import code

code.calculate_complexity(source)      # Cyclomatic complexity
code.extract_functions(source)         # List of functions with signatures
code.find_todos(source)                # TODO/FIXME/XXX comments
code.find_magic_numbers(source)        # Hardcoded numbers
code.check_naming_convention(source)   # PEP 8 naming violations
```

### Data Transformation (15 tools)
```python
from openagentflow.tools import data

data.json_to_csv(json_str)     # Convert JSON to CSV
data.csv_to_json(csv_str)      # Convert CSV to JSON
data.flatten_json(nested)      # Flatten nested JSON
data.yaml_to_json(yaml_str)    # YAML to JSON
data.xml_to_dict(xml_str)      # XML to dictionary
```

### Web/HTTP (10 tools)
```python
from openagentflow.tools import web

web.parse_url("https://example.com/path?q=1")  # URL components
web.extract_links(html)                         # All href links
web.html_to_markdown(html)                      # Convert to markdown
web.validate_email("user@example.com")          # Email validation
```

### Math/Science (10 tools)
```python
from openagentflow.tools import math

math.prime_factors(84)           # [2, 2, 3, 7]
math.is_prime(17)                # True
math.fibonacci(10)               # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
math.statistics_summary([1,2,3]) # mean, median, mode, std_dev
math.convert_units(100, "km", "miles")  # 62.137...
```

### Cryptography (8 tools)
```python
from openagentflow.tools import crypto

crypto.hash_sha256("hello")        # SHA-256 hash
crypto.generate_password(16)       # Secure random password
crypto.caesar_cipher("hello", 3)   # 'khoor'
crypto.generate_uuid()             # Random UUID
```

### Media (8 tools)
```python
from openagentflow.tools import media

media.color_hex_to_rgb("#FF5733")  # (255, 87, 51)
media.color_rgb_to_hex(255, 87, 51) # '#FF5733'
media.aspect_ratio(1920, 1080)      # '16:9'
media.generate_qr_data("https://...")  # QR code ASCII
```

### Date/Time (8 tools)
```python
from openagentflow.tools import datetime

datetime.parse_date("Jan 15, 2025")       # ISO format
datetime.date_difference("2025-01-01", "2025-12-31")  # Days between
datetime.get_weekday("2025-01-15")        # 'Wednesday'
datetime.timestamp_to_date(1704067200)    # ISO date
```

### AI/ML Helpers (8 tools)
```python
from openagentflow.tools import ai

ai.count_tokens("Hello world", model="gpt-4")  # Token estimate
ai.split_into_chunks(long_text, chunk_size=1000)  # Chunking
ai.extract_keywords(text, top_n=10)  # Key terms
ai.estimate_cost(1000, 500, "gpt-4")  # API cost estimate
```

### System/File (10 tools)
```python
from openagentflow.tools import system

system.human_readable_size(1048576)   # '1.00 MB'
system.sanitize_filename("file?.txt") # 'file.txt'
system.glob_to_regex("*.py")          # Regex pattern
system.parse_env_file(env_content)    # Dict of env vars
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Open Agent Flow                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │                    Decorators Layer                       │    │
│   │  @agent  │  @tool  │  @chain  │  @swarm  │  configure()  │    │
│   └──────────────────────────────────────────────────────────┘    │
│                              │                                     │
│   ┌──────────────────────────▼───────────────────────────────┐    │
│   │                   LLM Providers                           │    │
│   │  Anthropic │ OpenAI │ Bedrock │ Ollama │ Claude Code CLI │    │
│   └──────────────────────────────────────────────────────────┘    │
│                              │                                     │
│   ┌──────────────────────────▼───────────────────────────────┐    │
│   │               Specialized Agents (20)                     │    │
│   │  Code Quality │ Security │ Testing │ Documentation       │    │
│   │  Refactoring  │ Creative │ Research                       │    │
│   └──────────────────────────────────────────────────────────┘    │
│                              │                                     │
│   ┌──────────────────────────▼───────────────────────────────┐    │
│   │                Pure Python Tools (107)                    │    │
│   │  Text │ Code │ Data │ Web │ Math │ Crypto │ Media │ AI   │    │
│   └──────────────────────────────────────────────────────────┘    │
│                              │                                     │
│   ┌──────────────────────────▼───────────────────────────────┐    │
│   │                   Runtime Layer                           │    │
│   │     Executor │ Memory │ Traces │ Guardrails              │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Zero API Key Mode

Open Agent Flow automatically detects and uses the Claude Code CLI if installed:

```python
# No API key needed! Just have Claude Code CLI installed.
@agent(model="claude-sonnet-4-20250514")
async def my_agent(query: str) -> str:
    """This works without any API key if Claude Code CLI is available."""
    pass
```

Provider selection order:
1. Direct `api_key` parameter
2. `configure(anthropic_api_key="...")`
3. `ANTHROPIC_API_KEY` environment variable
4. **Claude Code CLI** (no key needed!)
5. MockProvider (for testing)

## Configuration

### Environment Variables
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Programmatic Configuration
```python
from openagentflow import configure

configure(
    anthropic_api_key="sk-ant-...",
    openai_api_key="sk-...",
)
```

## Contributing

Contributions welcome! Please read the contributing guidelines first.

```bash
# Clone and install
git clone https://github.com/v10z/openagentflow.git
cd openagentflow
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with intelligence by AI agents
</p>
