"""OpenAI provider for Open Agent Flow.

Supports GPT-4, GPT-4o, GPT-4o-mini, and future models.
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
OPENAI_PRICING = {
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

SUPPORTED_MODELS = list(OPENAI_PRICING.keys())


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider.

    Example:
        from openagentflow.llm.providers import OpenAIProvider
        from openagentflow.core.types import Message, ModelConfig

        provider = OpenAIProvider()  # Uses OPENAI_API_KEY env var
        response = await provider.generate(
            messages=[Message(role="user", content="Hello!")],
            config=ModelConfig(model_id="gpt-4o"),
        )
        print(response.content)
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise LLMError(
                    "openai package not installed. "
                    "Install with: pip install openagentflow[openai]"
                )

            if not self._api_key:
                raise AuthenticationError(
                    "OpenAI API key not found. "
                    "Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )

            self._client = openai.AsyncOpenAI(api_key=self._api_key)

        return self._client

    def _convert_messages(
        self, messages: list[Message], system_prompt: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert openagentflow Messages to OpenAI format."""
        result = []

        # Add system prompt first if provided
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            elif msg.role == "tool":
                # Tool result message
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                # Assistant message with tool calls
                openai_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": self._serialize_args(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": openai_tool_calls,
                })
            else:
                result.append({"role": msg.role, "content": msg.content})

        return result

    def _serialize_args(self, args: dict[str, Any]) -> str:
        """Serialize tool arguments to JSON string."""
        import json
        return json.dumps(args)

    def _convert_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert openagentflow ToolSpecs to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
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
        """Generate a response from GPT."""
        import json

        client = self._get_client()

        # Build request
        request_params: dict[str, Any] = {
            "model": config.model_id,
            "max_tokens": config.max_tokens,
            "messages": self._convert_messages(messages, system_prompt),
        }

        if config.temperature is not None:
            request_params["temperature"] = config.temperature

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        try:
            import openai

            response = await client.chat.completions.create(**request_params)
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}")

        # Parse response
        from openagentflow.core.types import ToolCall

        message = response.choices[0].message
        content_text = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        tool_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason or "stop",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
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
        """Stream a response from GPT."""
        client = self._get_client()

        # Build request
        request_params: dict[str, Any] = {
            "model": config.model_id,
            "max_tokens": config.max_tokens,
            "messages": self._convert_messages(messages, system_prompt),
            "stream": True,
        }

        if config.temperature is not None:
            request_params["temperature"] = config.temperature

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        try:
            import openai

            stream = await client.chat.completions.create(**request_params)

            current_tool_calls: dict[int, dict[str, Any]] = {}

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Text content
                if delta.content:
                    yield StreamChunk(text=delta.content)

                # Tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                            if tc_delta.id:
                                yield StreamChunk(
                                    is_tool_call_start=True,
                                    tool_call_id=tc_delta.id,
                                )

                        if tc_delta.function:
                            if tc_delta.function.name:
                                current_tool_calls[idx]["name"] = tc_delta.function.name
                                yield StreamChunk(
                                    tool_call_id=current_tool_calls[idx]["id"],
                                    tool_name=tc_delta.function.name,
                                )
                            if tc_delta.function.arguments:
                                current_tool_calls[idx]["arguments"] += tc_delta.function.arguments
                                yield StreamChunk(
                                    tool_call_id=current_tool_calls[idx]["id"],
                                    tool_input_delta=tc_delta.function.arguments,
                                )

                # Check for end
                if chunk.choices[0].finish_reason:
                    yield StreamChunk(is_final=True)

        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}")

    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens using tiktoken.

        Falls back to approximate count if tiktoken unavailable.
        """
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(model_id)
            return len(encoding.encode(text))
        except Exception:
            # Rough approximation: ~4 chars per token for English
            return len(text) // 4

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model_id: str
    ) -> float:
        """Estimate cost in USD."""
        pricing = OPENAI_PRICING.get(
            model_id, {"input": 5.00, "output": 15.00}  # Default to gpt-4o pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return SUPPORTED_MODELS
