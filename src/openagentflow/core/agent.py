"""
@agent decorator for defining autonomous agents.

Follows the TwinGraph 1.0 patterns:
- Decorator works with and without framework overhead
- Source code capture for reproducibility
- Hash-based lineage tracking
- Multi-platform abstraction (same API for local, docker, k8s, lambda)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar, overload
from uuid import uuid4

from openagentflow.core.types import (
    AgentResult,
    AgentSpec,
    AgentState,
    AgentStatus,
    ExecutionHash,
    ExecutorType,
    LLMProvider,
    MemoryConfig,
    ModelConfig,
    ReasoningStrategy,
    ToolSpec,
)
from openagentflow.exceptions import AgentError, AgentTimeoutError, MaxIterationsError

F = TypeVar("F", bound=Callable[..., Any])

# Agent registry (global)
_agent_registry: dict[str, AgentSpec] = {}


def get_agent(name: str) -> AgentSpec | None:
    """Get a registered agent by name."""
    return _agent_registry.get(name)


def get_all_agents() -> dict[str, AgentSpec]:
    """Get all registered agents."""
    return _agent_registry.copy()


@overload
def agent(func: F) -> F: ...


@overload
def agent(
    *,
    model: str | ModelConfig = "claude-sonnet-4-20250514",
    tools: list[Callable[..., Any]] | None = None,
    reasoning_strategy: str | ReasoningStrategy = ReasoningStrategy.REACT,
    max_iterations: int = 10,
    timeout: float = 300.0,
    executor: str | ExecutorType = ExecutorType.LOCAL,
    memory: MemoryConfig | None = None,
    system_prompt: str | None = None,
    supervisor: str | None = None,
    git_tracking: bool = False,
    guardrails: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    api_key: str | None = None,
) -> Callable[[F], F]: ...


def agent(
    func: F | None = None,
    *,
    model: str | ModelConfig = "claude-sonnet-4-20250514",
    tools: list[Callable[..., Any]] | None = None,
    reasoning_strategy: str | ReasoningStrategy = ReasoningStrategy.REACT,
    max_iterations: int = 10,
    timeout: float = 300.0,
    executor: str | ExecutorType = ExecutorType.LOCAL,
    memory: MemoryConfig | None = None,
    system_prompt: str | None = None,
    supervisor: str | None = None,
    git_tracking: bool = False,
    guardrails: list[str] | None = None,
    name: str | None = None,
    description: str | None = None,
    api_key: str | None = None,
) -> F | Callable[[F], F]:
    """
    Decorator to define an autonomous agent.

    Can be used with or without arguments:

        @agent
        async def researcher(query: str) -> str:
            '''Research agent.'''
            ...

        @agent(model="claude-sonnet-4-20250514", tools=[search, calculate])
        async def smart_researcher(query: str) -> Report:
            '''Smart research agent with tools.'''
            ...

        @agent(model="claude-sonnet-4-20250514", api_key="sk-ant-...")
        async def researcher(query: str) -> str:
            '''Agent with direct API key.'''
            ...

    Args:
        model: LLM model to use (string or ModelConfig)
        tools: List of tool functions (decorated with @tool)
        reasoning_strategy: Strategy for agent reasoning (react, cot, tot)
        max_iterations: Maximum reasoning iterations
        timeout: Execution timeout in seconds
        executor: Execution backend (local, docker, kubernetes, lambda)
        memory: Memory configuration
        system_prompt: Custom system prompt
        supervisor: Name of supervisor agent
        git_tracking: Enable git commit tracking
        guardrails: List of guardrail names to apply
        name: Agent name (defaults to function name)
        description: Agent description (defaults to docstring)
        api_key: Direct API key for the LLM provider (optional)
    """

    def decorator(fn: F) -> F:
        agent_name = name or fn.__name__
        agent_description = description or fn.__doc__ or f"Agent: {agent_name}"

        # Parse model config
        if isinstance(model, str):
            model_config = _parse_model_string(model)
        else:
            model_config = model

        # Parse reasoning strategy
        if isinstance(reasoning_strategy, str):
            strategy = ReasoningStrategy(reasoning_strategy)
        else:
            strategy = reasoning_strategy

        # Parse executor
        if isinstance(executor, str):
            exec_type = ExecutorType(executor)
        else:
            exec_type = executor

        # Extract tool specs from decorated functions
        tool_specs: list[ToolSpec] = []
        if tools:
            for tool_func in tools:
                if hasattr(tool_func, "_tool_spec"):
                    tool_specs.append(tool_func._tool_spec)
                else:
                    # Warn: tool not decorated
                    pass

        # Capture source code (TwinGraph pattern)
        try:
            source_code = inspect.getsource(fn)
            source_file = inspect.getfile(fn)
        except (OSError, TypeError):
            source_code = None
            source_file = None

        # Create agent spec
        spec = AgentSpec(
            name=agent_name,
            description=agent_description.strip(),
            func=fn,
            model=model_config,
            tools=tool_specs,
            reasoning_strategy=strategy,
            memory_config=memory,
            max_iterations=max_iterations,
            timeout_seconds=timeout,
            executor_type=exec_type,
            supervisor=supervisor,
            system_prompt=system_prompt,
            guardrails=guardrails or [],
            source_code=source_code,
            source_file=source_file,
            git_tracking=git_tracking,
        )

        # Register agent
        _agent_registry[agent_name] = spec

        @functools.wraps(fn)
        async def async_wrapper(
            *args: Any,
            parent_hash: str | list[str] | None = None,
            **kwargs: Any,
        ) -> AgentResult:
            """
            Async wrapper that executes the agent with full tracing.

            Args:
                *args: Positional arguments for the agent
                parent_hash: Parent execution hash(es) for lineage
                **kwargs: Keyword arguments for the agent
            """
            run_id = str(uuid4())
            start_time = datetime.now(timezone.utc)

            # Create execution hash (TwinGraph pattern)
            parent_hashes = (
                [parent_hash] if isinstance(parent_hash, str) else (parent_hash or [])
            )
            execution_hash = ExecutionHash.generate(parent_hashes)

            # Initialize state
            state = AgentState(
                execution_hash=execution_hash,
                parent_hashes=parent_hashes,
                status=AgentStatus.THINKING,
            )

            try:
                # Build input data from args/kwargs
                sig = inspect.signature(fn)
                params = list(sig.parameters.keys())
                input_data = {}
                for i, arg in enumerate(args):
                    if i < len(params):
                        input_data[params[i]] = arg
                input_data.update(kwargs)

                # Get the LLM provider based on model config
                provider = _get_provider(model_config, api_key=api_key)

                # Run the ReAct loop via executor
                from openagentflow.runtime.executor import AgentExecutor

                executor = AgentExecutor(provider)
                result = await executor.run(
                    spec=spec,
                    input_data=input_data,
                    parent_hash=parent_hash,
                )

                return result

            except Exception as e:
                # The executor handles most errors internally
                # This catches any unexpected issues
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                return AgentResult(
                    agent_name=agent_name,
                    run_id=run_id,
                    status=AgentStatus.FAILED,
                    error=str(e),
                    duration_ms=duration,
                    execution_hash=execution_hash,
                )

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync wrapper for direct function calls (without framework)."""
            return fn(*args, **kwargs)

        # Store wrapper in spec for chain/swarm access
        spec.wrapper = async_wrapper

        # For async functions, return async wrapper
        # For sync functions, return sync wrapper with async_call attached
        if asyncio.iscoroutinefunction(fn):
            async_wrapper._agent_spec = spec  # type: ignore
            return async_wrapper  # type: ignore
        else:
            sync_wrapper._agent_spec = spec  # type: ignore
            sync_wrapper._async_call = async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def _get_provider(config: ModelConfig, api_key: str | None = None) -> "BaseLLMProvider":
    """Get the appropriate LLM provider for a model config.

    Provider selection order:
    1. Direct api_key parameter -> Use API directly
    2. configure() / environment variable -> Use API directly
    3. Claude Code CLI installed -> Use Claude Code CLI (no key needed!)
    4. Falls back to MockProvider for testing

    Args:
        config: Model configuration
        api_key: Direct API key (optional, overrides all other sources)

    Returns:
        LLM provider instance
    """
    from openagentflow.config.settings import get_api_key
    from openagentflow.llm.base import BaseLLMProvider
    from openagentflow.llm.providers.claude_code import is_claude_code_available

    # Check for mock/local model
    if config.model_id in ("mock", "mock-provider", "test", "local"):
        from openagentflow.llm.providers.mock import MockProvider

        return MockProvider(verbose=True)

    # Check for claude-code model explicitly
    if config.model_id == "claude-code":
        if is_claude_code_available():
            from openagentflow.llm.providers.claude_code import ClaudeCodeProvider

            return ClaudeCodeProvider()
        else:
            from openagentflow.llm.providers.mock import MockProvider

            print(
                "[OpenAgentFlow] Claude Code CLI not found. "
                "Install from: https://docs.anthropic.com/claude-code "
                "Using MockProvider for testing."
            )
            return MockProvider(verbose=True)

    if config.provider == LLMProvider.ANTHROPIC:
        # Check for API key: direct > configure() > env var
        key = api_key or get_api_key("anthropic")
        if key:
            from openagentflow.llm.providers.anthropic_ import AnthropicProvider

            return AnthropicProvider(api_key=key)

        # No API key - try Claude Code CLI
        if is_claude_code_available():
            from openagentflow.llm.providers.claude_code import ClaudeCodeProvider

            print(
                "[OpenAgentFlow] No API key found, but Claude Code CLI detected! "
                "Using Claude Code for LLM calls (no key needed)."
            )
            return ClaudeCodeProvider()

        # Try Ollama as a local fallback
        from openagentflow.llm.providers.ollama_ import OllamaProvider, is_ollama_available

        if is_ollama_available():
            print(
                "[OpenAgentFlow] No API key found and Claude Code CLI not installed, "
                "but Ollama is running locally! Using Ollama as fallback provider."
            )
            return OllamaProvider()

        # Fall back to mock provider
        from openagentflow.llm.providers.mock import MockProvider

        print(
            "[OpenAgentFlow] No Anthropic API key found, Claude Code CLI not installed, "
            "and Ollama not running. "
            "Options: 1) configure(anthropic_api_key='...') "
            "2) Install Claude Code CLI "
            "3) Start Ollama (ollama serve) "
            "4) Set ANTHROPIC_API_KEY env var. "
            "Using MockProvider for development/testing."
        )
        return MockProvider(verbose=True)

    elif config.provider == LLMProvider.OPENAI:
        key = api_key or get_api_key("openai")
        if key:
            from openagentflow.llm.providers.openai_ import OpenAIProvider

            return OpenAIProvider(api_key=key)
        else:
            from openagentflow.llm.providers.mock import MockProvider

            print(
                "[OpenAgentFlow] No OpenAI API key found. "
                "Use configure(openai_api_key='...') or set OPENAI_API_KEY. "
                "Using MockProvider for development/testing."
            )
            return MockProvider(verbose=True)

    elif config.provider == LLMProvider.AWS_BEDROCK:
        from openagentflow.llm.providers.mock import MockProvider

        print(
            "[OpenAgentFlow] AWS Bedrock not yet implemented. "
            "Using MockProvider for development/testing."
        )
        return MockProvider(verbose=True)

    elif config.provider == LLMProvider.OLLAMA:
        from openagentflow.llm.providers.ollama_ import OllamaProvider, is_ollama_available

        base_url = "http://localhost:11434"
        if is_ollama_available(base_url):
            return OllamaProvider(base_url=base_url)

        from openagentflow.llm.providers.mock import MockProvider

        print(
            "[OpenAgentFlow] Ollama server not reachable at "
            f"{base_url}. Start it with: ollama serve\n"
            "Using MockProvider for development/testing."
        )
        return MockProvider(verbose=True)

    else:
        from openagentflow.llm.providers.mock import MockProvider

        print(f"[OpenAgentFlow] Unknown provider: {config.provider}. Using MockProvider.")
        return MockProvider(verbose=True)


def _parse_model_string(model_str: str) -> ModelConfig:
    """Parse a model string into ModelConfig.

    Supports an explicit ``provider/model`` prefix syntax (e.g.
    ``"ollama/llama3"``) as well as heuristic inference from the model name.
    """
    # ----- Explicit provider prefix (e.g. "ollama/llama3") -----
    if model_str.startswith("ollama/"):
        return ModelConfig(provider=LLMProvider.OLLAMA, model_id=model_str)

    # Map common model strings to full configs
    model_map = {
        "claude-sonnet-4-20250514": (LLMProvider.ANTHROPIC, "claude-sonnet-4-20250514"),
        "claude-opus-4-20250514": (LLMProvider.ANTHROPIC, "claude-opus-4-20250514"),
        "claude-3-5-sonnet": (LLMProvider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
        "claude-3-opus": (LLMProvider.ANTHROPIC, "claude-3-opus-20240229"),
        "gpt-4": (LLMProvider.OPENAI, "gpt-4-turbo-preview"),
        "gpt-4o": (LLMProvider.OPENAI, "gpt-4o"),
        "gpt-4o-mini": (LLMProvider.OPENAI, "gpt-4o-mini"),
        "gpt-3.5-turbo": (LLMProvider.OPENAI, "gpt-3.5-turbo"),
    }

    if model_str in model_map:
        provider, model_id = model_map[model_str]
        return ModelConfig(provider=provider, model_id=model_id)

    # Try to infer provider from model string
    lower = model_str.lower()
    if "claude" in lower:
        return ModelConfig(provider=LLMProvider.ANTHROPIC, model_id=model_str)
    elif "gpt" in lower:
        return ModelConfig(provider=LLMProvider.OPENAI, model_id=model_str)
    elif any(
        name in lower
        for name in (
            "llama", "mistral", "mixtral", "codellama", "deepseek",
            "qwen", "phi", "gemma", "command-r", "vicuna", "orca",
        )
    ):
        return ModelConfig(provider=LLMProvider.OLLAMA, model_id=model_str)
    else:
        # Default to Anthropic
        return ModelConfig(provider=LLMProvider.ANTHROPIC, model_id=model_str)


async def run_agent(
    agent_name: str,
    input_data: dict[str, Any],
    parent_hash: str | list[str] | None = None,
) -> AgentResult:
    """
    Run a registered agent by name.

    Args:
        agent_name: Name of the registered agent
        input_data: Input data for the agent
        parent_hash: Parent execution hash(es) for lineage
    """
    spec = get_agent(agent_name)
    if spec is None:
        raise AgentError(f"Agent not found: {agent_name}", agent_name=agent_name)

    # Get the async wrapper
    func = spec.func
    if hasattr(func, "_async_call"):
        return await func._async_call(**input_data, parent_hash=parent_hash)
    elif asyncio.iscoroutinefunction(func):
        return await func(**input_data, parent_hash=parent_hash)
    else:
        # Wrap sync function
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(**input_data))
