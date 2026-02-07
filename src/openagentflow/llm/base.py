"""Base protocol for LLM providers.

All LLM providers (Anthropic, OpenAI, Bedrock, Ollama) implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from openagentflow.core.types import Message, ModelConfig, ToolCall, ToolSpec


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    """The text content of the response."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """Tool calls requested by the model."""

    stop_reason: str = "end_turn"
    """Why the model stopped: 'end_turn', 'tool_use', 'max_tokens', etc."""

    input_tokens: int = 0
    """Number of input tokens used."""

    output_tokens: int = 0
    """Number of output tokens generated."""

    model_id: str = ""
    """The actual model ID that was used."""

    raw_response: Any = None
    """The raw response from the provider (for debugging)."""

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Whether the response includes tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    text: str = ""
    """Text content in this chunk."""

    is_tool_call_start: bool = False
    """Whether this chunk starts a tool call."""

    tool_call_id: str | None = None
    """ID of the tool call if this is a tool chunk."""

    tool_name: str | None = None
    """Name of the tool being called."""

    tool_input_delta: str = ""
    """Partial JSON input for the tool call."""

    is_final: bool = False
    """Whether this is the final chunk."""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations:
        - AnthropicProvider (Claude models)
        - OpenAIProvider (GPT models)
        - BedrockProvider (AWS Bedrock models)
        - OllamaProvider (Local models)

    Example:
        provider = AnthropicProvider(api_key="...")
        response = await provider.generate(
            messages=[Message(role="user", content="Hello!")],
            config=ModelConfig(model_id="claude-sonnet-4-20250514"),
        )
        print(response.content)
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history.
            config: Model configuration (model_id, temperature, etc.).
            tools: Available tools for the model to call.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with content and optional tool calls.

        Raises:
            LLMError: If the API call fails.
            RateLimitError: If rate limited by the provider.
            AuthenticationError: If API key is invalid.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the LLM.

        Yields chunks of text and tool call information as they arrive.

        Args:
            messages: Conversation history.
            config: Model configuration.
            tools: Available tools for the model to call.
            system_prompt: Optional system prompt.

        Yields:
            StreamChunk objects with text deltas and tool call info.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.
            model_id: The model to use for tokenization.

        Returns:
            Number of tokens.
        """
        ...

    @abstractmethod
    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model_id: str
    ) -> float:
        """Estimate cost in USD for a generation.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model_id: The model used.

        Returns:
            Estimated cost in USD.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of this provider (e.g., 'anthropic', 'openai')."""
        ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List of model IDs supported by this provider."""
        ...

    def supports_model(self, model_id: str) -> bool:
        """Check if this provider supports a model."""
        return model_id in self.supported_models
