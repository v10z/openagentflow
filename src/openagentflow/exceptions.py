"""
Exception hierarchy for Open Agent Flow.

All exceptions inherit from OpenAgentFlowError for easy catching.
"""

from __future__ import annotations

from typing import Any


class OpenAgentFlowError(Exception):
    """Base exception for all Open Agent Flow errors."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


# Configuration Errors
class ConfigurationError(OpenAgentFlowError):
    """Error in configuration."""

    pass


class ValidationError(OpenAgentFlowError):
    """Validation error for inputs or outputs."""

    pass


# Agent Errors
class AgentError(OpenAgentFlowError):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        *,
        agent_name: str | None = None,
        run_id: str | None = None,
        iteration: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.agent_name = agent_name
        self.run_id = run_id
        self.iteration = iteration


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    pass


class AgentCancelledError(AgentError):
    """Agent execution was cancelled."""

    pass


class MaxIterationsError(AgentError):
    """Agent reached maximum iterations without completing."""

    pass


# LLM Errors
class LLMError(OpenAgentFlowError):
    """Base exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model


class LLMConnectionError(LLMError):
    """Failed to connect to LLM provider."""

    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMContentFilterError(LLMError):
    """Content was filtered by provider safety systems."""

    pass


class LLMContextLengthError(LLMError):
    """Context length exceeded model limits."""

    def __init__(
        self,
        message: str,
        *,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens


# Tool Errors
class ToolError(OpenAgentFlowError):
    """Base exception for tool-related errors."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.tool_input = tool_input


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""

    pass


class ToolExecutionError(ToolError):
    """Error during tool execution."""

    pass


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""

    pass


class ToolValidationError(ToolError):
    """Tool input or output validation failed."""

    pass


# Memory Errors
class MemoryError(OpenAgentFlowError):
    """Base exception for memory-related errors."""

    pass


class MemoryConnectionError(MemoryError):
    """Failed to connect to memory backend."""

    pass


class MemoryQueryError(MemoryError):
    """Error querying memory."""

    pass


# Graph/Trace Errors
class GraphError(OpenAgentFlowError):
    """Base exception for graph-related errors."""

    pass


class GraphConnectionError(GraphError):
    """Failed to connect to graph database."""

    pass


class GraphQueryError(GraphError):
    """Error executing graph query."""

    pass


class TraceError(OpenAgentFlowError):
    """Error recording or querying traces."""

    pass


# Coordination Errors
class CoordinationError(OpenAgentFlowError):
    """Base exception for multi-agent coordination errors."""

    pass


class HandoffError(CoordinationError):
    """Error during agent handoff."""

    pass


class ConsensusError(CoordinationError):
    """Failed to reach consensus in swarm."""

    pass


# Guardrail Errors
class GuardrailError(OpenAgentFlowError):
    """Guardrail check failed."""

    def __init__(
        self,
        message: str,
        *,
        guardrail_name: str | None = None,
        blocked_content: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.guardrail_name = guardrail_name
        self.blocked_content = blocked_content


class TokenLimitError(GuardrailError):
    """Token limit exceeded."""

    pass


class CostLimitError(GuardrailError):
    """Cost limit exceeded."""

    pass


class ContentFilteredError(GuardrailError):
    """Content blocked by filter."""

    pass


# Auth/API Errors (used by LLM providers)
class AuthenticationError(LLMError):
    """Authentication failed (invalid API key, etc.)."""

    pass


class RateLimitError(LLMError):
    """Rate limit exceeded (alias for LLMRateLimitError for simpler use)."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class TimeoutError(OpenAgentFlowError):
    """Operation timed out."""

    pass
