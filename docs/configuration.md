# Configuration

## Environment Variables

| Variable           | Description                                          |
|--------------------|------------------------------------------------------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key                            |
| `OPENAI_API_KEY`   | OpenAI GPT API key                                   |
| `OLLAMA_BASE_URL`  | Ollama server URL (default: `http://localhost:11434`) |

## Programmatic Configuration

Use the `configure` function to set API keys and defaults at the module level:

```python
from openagentflow import configure

configure(
    anthropic_api_key="sk-ant-...",
    openai_api_key="sk-...",
)
```

## Provider Selection Order

When an agent needs an LLM provider, OpenAgentFlow resolves credentials in the following order:

1. **Direct `api_key` parameter** -- passed explicitly when creating an agent or provider instance.
2. **`configure(anthropic_api_key="...")`** -- set via the programmatic configuration function.
3. **`ANTHROPIC_API_KEY` environment variable** -- read from the process environment.
4. **Claude Code CLI** -- if running inside Claude Code, no key is needed.
5. **MockProvider** -- used for testing when no real provider is available.

The first source that provides a non-empty value wins. This allows production deployments to rely on environment variables while local development can use `configure()` or the CLI.

## LLM Providers

| Provider        | Install                              | Key Required |
|-----------------|--------------------------------------|--------------|
| Anthropic       | `pip install openagentflow[anthropic]` | Yes          |
| OpenAI          | `pip install openagentflow[openai]`   | Yes          |
| Ollama          | Base install                         | No (local)   |
| Claude Code CLI | Base install                         | No           |
| Mock            | Base install                         | No           |

### Provider Examples

**Anthropic:**

```python
from openagentflow.providers import AnthropicProvider

provider = AnthropicProvider(api_key="sk-ant-...")
response = await provider.generate("Explain quicksort.")
```

**OpenAI:**

```python
from openagentflow.providers import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
response = await provider.generate("Explain quicksort.")
```

**Ollama (local):**

```python
from openagentflow.providers import OllamaProvider

provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")
response = await provider.generate("Explain quicksort.")
```

**Mock (testing):**

```python
from openagentflow.providers import MockProvider

provider = MockProvider(responses=["This is a test response."])
response = await provider.generate("Any prompt")
# Returns "This is a test response."
```

## Installation Extras

| Extra        | What it includes          |
|--------------|---------------------------|
| `anthropic`  | Anthropic SDK             |
| `openai`     | OpenAI SDK                |
| `gremlin`    | Gremlin graph backend     |
| `all`        | All of the above          |
| `dev`        | Testing and linting tools |

### Install Examples

```bash
# Base install (Ollama, Claude Code CLI, Mock providers)
pip install openagentflow

# With Anthropic support
pip install openagentflow[anthropic]

# With all providers and backends
pip install openagentflow[all]

# Development install
pip install openagentflow[dev]
```
