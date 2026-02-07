"""
Core type definitions for Open Agent Flow.

This module defines all the fundamental types used throughout the framework:
- Agent specifications and state
- Tool definitions and results
- Message types for LLM communication
- Memory and reasoning configurations
- Multi-agent coordination types
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class AgentStatus(Enum):
    """Status of an agent execution."""

    IDLE = auto()
    THINKING = auto()
    TOOL_CALLING = auto()
    WAITING_HUMAN = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class ReasoningStrategy(Enum):
    """Available reasoning strategies for agents."""

    REACT = "react"  # Reason + Act loop
    CHAIN_OF_THOUGHT = "cot"  # Chain of Thought
    TREE_OF_THOUGHT = "tot"  # Tree of Thought
    REFLEXION = "reflexion"  # Self-reflection
    CUSTOM = "custom"  # User-defined


class MemoryType(Enum):
    """Types of memory backends."""

    SHORT_TERM = "short_term"  # Sliding window / summarization
    LONG_TERM = "long_term"  # Vector database
    EPISODIC = "episodic"  # Graph-based experiences
    WORKING = "working"  # Scratchpad


class LLMProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AWS_BEDROCK = "bedrock"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


class ExecutorType(Enum):
    """Execution backends for agents."""

    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS_LAMBDA = "lambda"
    AWS_BATCH = "batch"
    BEDROCK_AGENTS = "bedrock_agents"


# =============================================================================
# Execution Hash (from TwinGraph - content-addressable lineage)
# =============================================================================


@dataclass(frozen=True)
class ExecutionHash:
    """
    Immutable execution hash for lineage tracking.

    Preserves the hash-based lineage pattern from TwinGraph 1.0:
    - Each execution gets a unique hash
    - Child hashes are derived from parent hashes + timestamp
    - Enables graph-based audit trails
    """

    value: str
    parents: tuple[str, ...] = field(default_factory=tuple)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def generate(cls, parent_hashes: list[str] | None = None) -> ExecutionHash:
        """Generate a unique execution hash with optional parent lineage."""
        parents = tuple(parent_hashes) if parent_hashes else ()
        timestamp = datetime.now(timezone.utc)

        # Hash = MD5(concatenated_parent_hashes + timestamp + uuid for uniqueness)
        content = "".join(parents) + timestamp.isoformat() + str(uuid4())
        hash_value = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

        return cls(value=hash_value, parents=parents, timestamp=timestamp)

    @classmethod
    def from_content(cls, content: str, parent_hashes: list[str] | None = None) -> ExecutionHash:
        """
        Generate deterministic hash from content (for caching).

        Same content = same hash, enabling cache lookups.
        """
        parents = tuple(parent_hashes) if parent_hashes else ()
        hash_value = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
        return cls(value=hash_value, parents=parents)

    def __str__(self) -> str:
        return self.value[:16]  # Short form for display


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    provider: LLMProvider
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    timeout_seconds: float = 120.0
    # Provider-specific settings
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def claude_sonnet(cls, **kwargs: Any) -> ModelConfig:
        """Create config for Claude Sonnet."""
        return cls(
            provider=LLMProvider.ANTHROPIC,
            model_id="claude-sonnet-4-20250514",
            **kwargs,
        )

    @classmethod
    def claude_opus(cls, **kwargs: Any) -> ModelConfig:
        """Create config for Claude Opus."""
        return cls(
            provider=LLMProvider.ANTHROPIC,
            model_id="claude-opus-4-20250514",
            **kwargs,
        )

    @classmethod
    def gpt4(cls, **kwargs: Any) -> ModelConfig:
        """Create config for GPT-4."""
        return cls(
            provider=LLMProvider.OPENAI,
            model_id="gpt-4-turbo-preview",
            **kwargs,
        )


# =============================================================================
# Tool Types
# =============================================================================


@dataclass
class ToolSpec:
    """Specification for an agent-callable tool."""

    name: str
    description: str
    func: Callable[..., Any]
    input_schema: dict[str, Any]  # JSON Schema
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    requires_confirmation: bool = False  # Human confirmation needed
    cost_estimate: float | None = None  # Estimated cost per call
    # Source tracking (from TwinGraph pattern)
    source_code: str | None = None
    source_file: str | None = None


@dataclass
class ToolCall:
    """A tool invocation by an agent."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_call: ToolCall | None = None
    success: bool = False
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    _call_id: str | None = field(default=None, repr=False)

    @property
    def call_id(self) -> str:
        """Get the call ID from the tool call or _call_id."""
        if self.tool_call:
            return self.tool_call.id
        return self._call_id or ""

    @classmethod
    def from_call_id(
        cls,
        call_id: str,
        success: bool,
        output: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        """Create a ToolResult with just a call_id (no full ToolCall)."""
        return cls(
            tool_call=None,
            success=success,
            output=output,
            error=error,
            duration_ms=duration_ms,
            metadata=metadata or {},
            _call_id=call_id,
        )


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class Message:
    """A message in an agent conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None  # For tool messages
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # For tool result messages
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call_id: str, content: str, name: str | None = None) -> Message:
        """Create a tool result message."""
        return cls(role="tool", content=content, tool_call_id=call_id, name=name)


# =============================================================================
# Memory Configuration
# =============================================================================


@dataclass
class MemoryConfig:
    """Configuration for agent memory."""

    # Short-term memory
    short_term_enabled: bool = True
    short_term_strategy: Literal["sliding_window", "summarization"] = "sliding_window"
    short_term_max_tokens: int = 8000

    # Long-term memory (vector store)
    long_term_enabled: bool = False
    long_term_backend: str | None = None  # pgvector, chroma, pinecone
    long_term_config: dict[str, Any] = field(default_factory=dict)

    # Episodic memory (graph-based)
    episodic_enabled: bool = False
    episodic_backend: str | None = None  # tinkergraph, neptune

    # Working memory (scratchpad)
    working_enabled: bool = True


# =============================================================================
# Agent Types
# =============================================================================


@dataclass
class AgentSpec:
    """Specification for an autonomous agent."""

    name: str
    description: str
    func: Callable[..., Any]
    model: ModelConfig
    tools: list[ToolSpec] = field(default_factory=list)
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT
    memory_config: MemoryConfig | None = None
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    executor_type: ExecutorType = ExecutorType.LOCAL
    supervisor: str | None = None  # Supervisor agent name
    system_prompt: str | None = None
    guardrails: list[str] = field(default_factory=list)
    # Source tracking (TwinGraph pattern)
    source_code: str | None = None
    source_file: str | None = None
    git_tracking: bool = False
    # Additional attributes
    metadata: dict[str, Any] = field(default_factory=dict)
    # Wrapper function with parent_hash support (set by decorator)
    wrapper: Callable[..., Any] | None = None

    async def __call__(self, *args: Any, parent_hash: str | list[str] | None = None, **kwargs: Any) -> Any:
        """Call the agent with optional parent hash for lineage."""
        if self.wrapper:
            return await self.wrapper(*args, parent_hash=parent_hash, **kwargs)
        else:
            # Fallback to original function
            import asyncio
            if asyncio.iscoroutinefunction(self.func):
                return await self.func(*args, **kwargs)
            else:
                return self.func(*args, **kwargs)


@dataclass
class AgentState:
    """Mutable state maintained across ReAct iterations."""

    iteration: int = 0
    messages: list[Message] = field(default_factory=list)
    tool_calls_history: list[ToolCall] = field(default_factory=list)
    tool_results_history: list[ToolResult] = field(default_factory=list)
    accumulated_tokens: int = 0
    accumulated_cost: float = 0.0
    status: AgentStatus = AgentStatus.IDLE
    last_observation: str | None = None
    scratchpad: dict[str, Any] = field(default_factory=dict)
    # Lineage tracking
    execution_hash: ExecutionHash | None = None
    parent_hashes: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result of an agent execution."""

    agent_name: str
    run_id: str
    status: AgentStatus
    output: Any = None
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_ms: float = 0.0
    error: str | None = None
    trace_id: str | None = None
    execution_hash: ExecutionHash | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Multi-Agent Coordination Types
# =============================================================================


@dataclass
class ChainSpec:
    """Specification for a sequential agent chain."""

    name: str
    agents: list[str]  # Ordered agent names
    pass_output: bool = True  # Pass output to next agent
    stop_on_failure: bool = True


@dataclass
class SwarmSpec:
    """Specification for parallel multi-agent coordination."""

    name: str
    agents: list[str]  # Participating agent names
    consensus_strategy: Literal["voting", "debate", "hierarchical"] = "voting"
    min_agreement: float = 0.5  # For voting
    debate_rounds: int = 2  # For debate
    coordinator: str | None = None  # For hierarchical
    timeout_seconds: float = 600.0


# =============================================================================
# Protocols (for type checking)
# =============================================================================


@runtime_checkable
class Executor(Protocol):
    """Protocol for agent executors."""

    async def execute(
        self,
        spec: AgentSpec,
        input_data: dict[str, Any],
    ) -> AgentResult:
        """Execute an agent."""
        ...


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph database backends."""

    async def add_vertex(self, vertex: dict[str, Any]) -> str:
        """Add a vertex to the graph."""
        ...

    async def add_edge(
        self,
        from_vertex: str,
        to_vertex: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add an edge to the graph."""
        ...

    async def query(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute a query."""
        ...


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for memory backends."""

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a value."""
        ...

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories."""
        ...
