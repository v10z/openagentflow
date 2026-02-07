# OpenAgentFlow Examples

Runnable examples demonstrating every major feature of OpenAgentFlow.

## Prerequisites

```bash
pip install openagentflow[all]
```

Most examples use `claude-opus-4-6` as the model. Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or use Ollama for local inference (no API key needed):

```bash
ollama pull llama3
# Examples with Ollama will auto-detect the local server
```

## Examples

| File | What it demonstrates |
|------|---------------------|
| [`01_tools.py`](01_tools.py) | Creating tools with `@tool`, using built-in tools |
| [`02_agent.py`](02_agent.py) | Building an autonomous agent with `@agent` |
| [`03_chain.py`](03_chain.py) | Sequential pipelines with `@chain` |
| [`04_swarm.py`](04_swarm.py) | Parallel execution and consensus with `@swarm` |
| [`05_reasoning.py`](05_reasoning.py) | Using all 30 reasoning engines |
| [`06_memory.py`](06_memory.py) | 3-tier memory system with GC |
| [`07_graph.py`](07_graph.py) | Execution tracing with SQLite graph |
| [`08_meta_agent.py`](08_meta_agent.py) | JIT tool creation with the Meta-Agent |
| [`09_distributed.py`](09_distributed.py) | Multi-node compute clusters |
| [`10_full_pipeline.py`](10_full_pipeline.py) | End-to-end: agents + reasoning + memory + tracing |
| [`11_code_review_swarm.py`](11_code_review_swarm.py) | Multi-agent code review with specialized agents |
| [`12_ollama_local.py`](12_ollama_local.py) | Local-first AI with Ollama (no API key) |

## Running

```bash
# Run any example:
python examples/01_tools.py

# Run all examples:
for f in examples/*.py; do python "$f"; done
```
