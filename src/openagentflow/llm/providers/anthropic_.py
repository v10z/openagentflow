"""Anthropic Claude provider for Open Agent Flow.

Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and future models.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, AsyncIterator

from openagentflow.exceptions import (
    AuthenticationError,
    LLMError,
    RateLimitError,
)
from openagentflow.llm.base import BaseLLMProvider, LLMResponse, StreamChunk

if TYPE_CHECKING:
    from openagentflow.core.types import Message, ModelConfig, ToolSpec

# Model pricing per 1M tokens (USD) - as of late 2024
ANTHROPIC_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
}

SUPPORTED_MODELS = list(ANTHROPIC_PRICING.keys())


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider.

    Example:
        from openagentflow.llm.providers import AnthropicProvider
        from openagentflow.core.types import Message, ModelConfig

        provider = AnthropicProvider()  # Uses ANTHROPIC_API_KEY env var
        response = await provider.generate(
            messages=[Message(role="user", content="Hello!")],
            config=ModelConfig(model_id="claude-sonnet-4-20250514"),
        )
        print(response.content)
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise LLMError(
                    "anthropic package not installed. "
                    "Install with: pip install openagentflow[anthropic]"
                )

            if not self._api_key:
                raise AuthenticationError(
                    "Anthropic API key not found. "
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
                )

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

        return self._client

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert openagentflow Messages to Anthropic format."""
        result = []
        for msg in messages:
            if msg.role == "system":
                continue  # System messages handled separately

            anthropic_msg: dict[str, Any] = {"role": msg.role}

            # Handle tool results specially
            if msg.role == "user" and msg.tool_call_id:
                anthropic_msg["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                ]
            elif msg.tool_calls:
                # Assistant message with tool calls
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anthropic_msg["content"] = content_blocks
            else:
                anthropic_msg["content"] = msg.content

            result.append(anthropic_msg)

        return result

    def _convert_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert openagentflow ToolSpecs to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response from Claude."""
        client = self._get_client()

        # Build request
        request_params: dict[str, Any] = {
            "model": config.model_id,
            "max_tokens": config.max_tokens,
            "messages": self._convert_messages(messages),
        }

        if config.temperature is not None:
            request_params["temperature"] = config.temperature

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        try:
            import anthropic

            response = await client.messages.create(**request_params)
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}")

        # Parse response
        from openagentflow.core.types import ToolCall

        content_text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        tool_name=block.name,
                        arguments=dict(block.input),
                    )
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model_id=response.model,
            raw_response=response,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: ModelConfig,
        tools: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from Claude."""
        client = self._get_client()

        # Build request
        request_params: dict[str, Any] = {
            "model": config.model_id,
            "max_tokens": config.max_tokens,
            "messages": self._convert_messages(messages),
        }

        if config.temperature is not None:
            request_params["temperature"] = config.temperature

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        try:
            import anthropic

            async with client.messages.stream(**request_params) as stream:
                current_tool_id: str | None = None
                current_tool_name: str | None = None

                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if hasattr(block, "type"):
                            if block.type == "tool_use":
                                current_tool_id = block.id
                                current_tool_name = block.name
                                yield StreamChunk(
                                    is_tool_call_start=True,
                                    tool_call_id=block.id,
                                    tool_name=block.name,
                                )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            yield StreamChunk(text=delta.text)
                        elif hasattr(delta, "partial_json"):
                            yield StreamChunk(
                                tool_call_id=current_tool_id,
                                tool_name=current_tool_name,
                                tool_input_delta=delta.partial_json,
                            )
                    elif event.type == "message_stop":
                        yield StreamChunk(is_final=True)

        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}")

    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens using Anthropic's tokenizer.

        Falls back to approximate count if tokenizer unavailable.
        """
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key or "dummy")
            return client.count_tokens(text)
        except Exception:
            # Rough approximation: ~4 chars per token for English
            return len(text) // 4

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model_id: str
    ) -> float:
        """Estimate cost in USD."""
        pricing = ANTHROPIC_PRICING.get(
            model_id, {"input": 3.00, "output": 15.00}  # Default to Sonnet pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        return SUPPORTED_MODELS
